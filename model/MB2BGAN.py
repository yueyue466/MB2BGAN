import torch
import torch.nn.functional as F
import yaml
# from models.explore.kernel_encoding.kernel_wizard import KernelWizard
from models.explore.kernel_encoding.kernel_wizard import KernelWizard
from . import networks
from .base_model import BaseModel
from .losses import GANLoss, VGGLoss, cal_gradient_penalty


class Blur2BlurModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="batch", netG="unet_256")
        if is_train:
            parser.set_defaults(pool_size=0)
            parser.add_argument("--lambda_Perc", type=float, default=0.8, help="weight for Perc loss")##重构损失
            parser.add_argument("--lambda_gp", type=float, default=0.0001, help="weight for GP loss")##惩罚项

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.loss_names = ["G_GAN", "G_Perc", "D_real", "D_fake"]

        if self.isTrain:
            self.model_names = ["G", "D"]
            self.visual_names = ["real_A", "blur_known", "sharp_known", "fake_B_"]
        else:
            self.model_names = ["G"]
            self.visual_names = ["real_A", "fake_B_"]
        self.opt = opt

        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        self.upscale = torch.nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)##不依赖于学习的上采样
        self.device = torch.device("cuda:{}".format(self.gpu_ids[0])) if self.gpu_ids else torch.device("cpu")
        if self.isTrain:
            self.netD = networks.define_D(
                opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids
            )

            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
            self.criterionPerc = VGGLoss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.netD.to(self.device)
            self.netD = torch.nn.DataParallel(self.netD)
        with open("options/generate_blur/augmentation.yml", "r") as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)["KernelWizard"]
            model_path = opt["pretrained"]
        self.genblur = KernelWizard(opt)
        print("Loading KernelWizard...")
        self.genblur.eval()
        self.genblur.load_state_dict(torch.load(model_path))
        self.genblur = self.genblur.to(self.device)

    def set_input(self, input):

        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.Avis = input["Avis"].to(self.device)
        self.sizeA = input["sizeA"]
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

        if self.isTrain:
            self.real_B = input["B" if AtoB else "A"].to(self.device)

            self.blur_known = input["C"].to(self.device)
            self.sharp_known = input["D"].to(self.device)
            kernel_mean, kernel_sigma = self.genblur(self.sharp_known, self.blur_known)
            self.kernel_real = kernel_mean + kernel_sigma * torch.randn_like(kernel_mean)
            print(f"real_B: {self.real_B.shape}, kernel_real: {self.kernel_real.shape}")
            self.real_B = self.genblur.adaptKernel(self.real_B, self.kernel_real)

    def deblurring_step(self, x):
        nbatch = x.shape[0]
        chunk_size = 4
        outs = []
        with torch.no_grad():
            for idx in range(0, nbatch, chunk_size):
                pred = self.deblur(x[idx : idx + chunk_size])
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
        return torch.cat(outs, dim=0).to(self.device)

    def forward(self):
        self.fake_B = self.netG(self.real_A, self.Avis)
        # self.fake_B = self.netG(self.real_A)
        self.fake_B_ = self.fake_B[2]

    def backward_D(self, iters):

        fake_B = [x.detach() for x in self.fake_B]
        pred_fake = self.netD(fake_B)
        self.loss_D_fake = self.criterionGAN(iters, pred_fake, False, dis_update=True)

        real_B0 = F.interpolate(self.real_B, scale_factor=0.25, mode="bilinear")
        real_B1 = F.interpolate(self.real_B, scale_factor=0.5, mode="bilinear")
        real_B = [real_B0, real_B1, self.real_B]

        pred_real = self.netD(real_B)
        self.loss_D_real = self.criterionGAN(0, pred_real, True, dis_update=True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D += cal_gradient_penalty(self.netD, real_B[2], fake_B[2], self.real_B.device, self.opt.lambda_gp)[0]##计算惩罚项梯度损失

        self.loss_D.backward()

    def backward_G(self, iters):

        pred_fake = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(0, pred_fake, True, dis_update=False)

        real_A0 = F.interpolate(self.real_A, scale_factor=0.25, mode="bilinear")
        real_A1 = F.interpolate(self.real_A, scale_factor=0.5, mode="bilinear")
        perc1 = self.criterionPerc.forward(self.fake_B[0], real_A0)
        perc2 = self.criterionPerc.forward(self.fake_B[1], real_A1)
        perc3 = self.criterionPerc.forward(self.fake_B[2], self.real_A)
        self.loss_G_Perc = (perc1 + perc2 + perc3) * self.opt.lambda_Perc

        self.loss_G = self.loss_G_GAN + self.loss_G_Perc
        self.loss_G.backward()

    def optimize_parameters(self, iters):
        self.forward()  # #计算G（A）

        # update D_kernel
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D(iters)
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G(iters)
        self.optimizer_G.step()

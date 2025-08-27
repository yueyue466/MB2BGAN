import os
import random

import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class UnalignedDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.isTrain = opt.phase == "train"
        self.ratio = opt.ratio if self.isTrain else 1.0

        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  ## create a path '/path/to/data/trainA'
        self.dir_Avis = os.path.join(opt.dataroot, opt.phase + "Avis")
        if self.isTrain:
            self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  ## create a path '/path/to/data/trainB'
            self.dir_C = os.path.join(opt.dataroot, opt.phase + "C")
            self.dir_D = os.path.join(opt.dataroot, opt.phase + "D")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # 5
        self.Avis_paths = sorted(make_dataset(self.dir_Avis, opt.max_dataset_size))
        if self.isTrain:
            self.B_paths = sorted(
                make_dataset(self.dir_B, opt.max_dataset_size)
            )  # load images from '/path/to/data/trainB'
            self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))
            self.D_paths = sorted(make_dataset(self.dir_D, opt.max_dataset_size))

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.Avi_size = len(self.Avis_paths)
        print(self.A_size)
        if self.isTrain:
            self.B_size = len(self.B_paths)  # get the size of dataset B
            self.C_size = len(self.C_paths)
            self.D_size = len(self.D_paths)

        self.BtoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if self.BtoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = (
            self.opt.input_nc if self.BtoA else self.opt.output_nc
        )  # get the number of channels of output image
        if self.BtoA:
            self.transform_A = get_transform(self.opt, self.opt.preprocessB, "B", grayscale=(input_nc == 1))
            self.transform_Avis = get_transform(self.opt, self.opt.preprocessB, "Avis", grayscale=(input_nc == 1))
            if self.isTrain:
                self.transform_B = get_transform(self.opt, self.opt.preprocessB, "A", grayscale=(output_nc == 1))
                self.transform_C = get_transform(self.opt, self.opt.preprocessA, "C", grayscale=(output_nc == 1))
                self.transform_D = get_transform(self.opt, self.opt.preprocessA, "D", grayscale=(output_nc == 1))
        else:
            self.transform_A = get_transform(self.opt, self.opt.preprocessA, "A", grayscale=(input_nc == 1))
            self.transform_Avis = get_transform(self.opt, self.opt.preprocessA, "Avis", grayscale=(input_nc == 1))
            if self.isTrain:
                self.transform_B = get_transform(self.opt, self.opt.preprocessA, "B", grayscale=(output_nc == 1))
                self.transform_C = get_transform(self.opt, self.opt.preprocessB, "C", grayscale=(output_nc == 1))
                self.transform_D = get_transform(self.opt, self.opt.preprocessB, "D", grayscale=(output_nc == 1))

    def __getitem__(self, index):
        nir_path = self.A_paths[index % self.A_size]
        nir_filename = os.path.basename(nir_path)  # 获取文件名，例如 "123nir.png"
        nir_base, ext = os.path.splitext(nir_filename)  # 分离基础名和扩展名，例如 "123nir" 和 ".png"
        nir_prefix = nir_base.replace("nir", "")  # 提取前缀，例如 "123"

        # 构造对应的可见光文件名
        vis_filename = nir_prefix + "vis" + ext  # 例如 "123vis.png"
        vis_image_path = os.path.join(self.dir_Avis, vis_filename)

        # 检查文件是否存在
        if not os.path.exists(vis_image_path):
            raise ValueError(f"未找到与红外图像 {nir_filename} 对应的可见光图像 {vis_filename}")

        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        Avis_path = vis_image_path
        A_img = Image.open(A_path).convert("RGB")
        Avis_img = Image.open(Avis_path).convert("RGB")

        sizeA = A_img.size

        # apply image transformation
        transform_list = []
        transform_list += [transforms.ToTensor()]
        f = transforms.Compose(transform_list)

        # 生成随机种子确保变换同步
        seed = random.randint(0, 0xFFFFFFFF)

        # 处理A图像
        random.seed(seed)
        torch.manual_seed(seed)
        A = self.transform_A(A_img)

        # 处理Avis图像（使用相同种子）
        random.seed(seed)
        torch.manual_seed(seed)
        Avis = self.transform_Avis(Avis_img)

        if self.isTrain:
            if self.opt.serial_batches:  # make sure index is within then range
                index_B = index % self.B_size
            else:  # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]

            C_path = self.C_paths[index_B]
            D_path = self.D_paths[index_B]

            B_img = Image.open(B_path).convert("RGB")
            C_img = Image.open(C_path).convert("RGB")
            D_img = Image.open(D_path).convert("RGB")
            Avis_img = Image.open(Avis_path).convert("RGB")

            if "padding" in self.opt.preprocessA and not self.BtoA:
                w, h = make_power_2(A_img.size[0], A_img.size[1], 256)
                result = Image.new(A_img.mode, (w, h), (0, 0, 0))
                result.paste(A_img, (0, 0))
                A_img = result

                w, h = make_power_2(Avis_img.size[0], Avis_img.size[1], 256)
                result = Image.new(Avis_img.mode, (w, h), (0, 0, 0))
                result.paste(Avis_img, (0, 0))
                Avis_img = result

            elif "padding" in self.opt.preprocessB and self.BtoA:
                w, h = make_power_2(B_img.size[0], B_img.size[1], 256)
                result = Image.new(B_img.mode, (w, h), (0, 0, 0))
                result.paste(B_img, (0, 0))
                B_img = result

                w, h = make_power_2(Avis_img.size[0], Avis_img.size[1], 256)
                result = Image.new(Avis_img.mode, (w, h), (0, 0, 0))
                result.paste(Avis_img, (0, 0))
                Avis_img = result

            # apply image transformation
            transform_list = []
            transform_list += [transforms.ToTensor()]
            f = transforms.Compose(transform_list)

            B = self.transform_B(B_img)

            crop_w, crop_h = A.shape[1], A.shape[2]
            crop_indices = transforms.RandomCrop.get_params(C_img, output_size=(crop_w, crop_h))
            i, j, _, _ = crop_indices

            blur_tensor = transforms.functional.crop(C_img, i, j, crop_w, crop_h)
            sharp_tensor = transforms.functional.crop(D_img, i, j, crop_w, crop_h)
            blur_tensor = f(blur_tensor)
            sharp_tensor = f(sharp_tensor)

            return {
                "A": A,
                "Avis": Avis,
                "B": B,
                "C": blur_tensor,
                "D": sharp_tensor,
                "A_paths": A_path,
                "Avis_paths": Avis_path,
                "B_paths": B_path,
                "C_paths": C_path,
                "D_paths": D_path,
                "sizeA": sizeA,
            }
        else:
            return {"A": A, "Avis":Avis,"A_paths": A_path,"Avis_paths": Avis_path, "sizeA": sizeA}

    def random_masking(self, x, mask_ratio):
        # breakpoint()
        _x = x.clone()
        # p = self.opt.patch_size
        x = x.reshape(16 * 16, 16 * 16 * 3)
        L, D = x.shape  # batch, length, dim
        len_keep = int(L * mask_ratio)

        noise = torch.rand(L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([L], device=x.device)
        mask[:len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=0, index=ids_restore)
        mask = mask.unsqueeze(-1).repeat(1, D)
        mask = mask.reshape(16, 16, 16, 16, 3).permute(4, 0, 2, 1, 3).reshape(3, 256, 256)
        print(mask)
        print(_x)

        return mask * _x

    def __len__(self):
        if self.isTrain:
            return max(self.A_size, self.B_size)
        return self.A_size


def make_power_2(ow, oh, base, method=transforms.InterpolationMode.BICUBIC):
    # method = __transforms2pil_resize(method)
    if oh % base != 0:
        h = int((int(oh / base) + 1) * base)
    else:
        h = oh

    if ow % base != 0:
        w = int((int(ow / base) + 1) * base)
    else:
        w = ow
    return (w, h)

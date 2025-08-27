#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'LiuYue Xu'
__email__ = 'xlygy410@gmail.com'
__project__ = 'MB2BGAN: Unsupervised Cross-Modal Feature-Guided Infrared Image Deblurring'
__date__ = '2025-08-27'

"""
MB2BGAN: A multi-modal GAN for unsupervised infrared image deblurring.
Implements Modal Adaptive Fusion (MAF) module to combine infrared (IR) and visible (VIS) features.
Based on MIMO-UNet with enhancements for cross-modal feature fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ResBlock, BasicConv


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()
        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()
        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane - 3, kernel_size=1, stride=1, relu=True),
        )
        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class MAF(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(MAF, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # Channel attention for modality fusion
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # Learnable modality weight

        # Spatial attention: multi-scale convolution
        self.conv3x3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, padding=2, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x_ir, x_vis=None):
        # Modal adaptive fusion: Eq. (1) in paper
        if x_vis is not None:
            x = self.alpha * x_ir + (1 - self.alpha) * x_vis
        else:
            x = x_ir

        # Residual connection
        identity = x

        # Channel attention
        channel_max = F.max_pool2d(x, (x.size(2), x.size(3)))
        channel_max = channel_max.view(channel_max.size(0), -1)
        channel_weight = self.fc2(F.relu(self.fc1(channel_max)))
        channel_weight = torch.sigmoid(channel_weight).view(x.size(0), x.size(1), 1, 1)
        x_ch = x * channel_weight

        # Spatial attention: Eq. (2) in paper
        scale_3x3 = self.conv3x3(x_ch)
        scale_5x5 = self.conv5x5(x_ch)
        spatial_concat = torch.cat([scale_3x3, scale_5x5], dim=1)
        spatial_weight = torch.sigmoid(self.conv1x1(spatial_concat))
        x_sp = x_ch * spatial_weight

        # Final output with residual: Eq. (2)
        out = F.relu(x_sp + identity)
        return out


class MB2BGAN(nn.Module):
    def __init__(self, num_res=8):
        super(MB2BGAN, self).__init__()
        base_channel = 32

        # Encoder
        self.Encoder = nn.ModuleList(
            [
                EBlock(base_channel, num_res),
                EBlock(base_channel * 2, num_res),
                EBlock(base_channel * 4, num_res),
            ]
        )
        self.feat_extract = nn.ModuleList(
            [
                BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
                BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
                BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
                BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
                BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
                BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.MAF1 = MAF(base_channel)
        self.MAF2 = MAF(base_channel * 2)
        self.MAF3 = MAF(base_channel * 4)

        # Decoder
        self.Decoder = nn.ModuleList(
            [DBlock(base_channel * 4, num_res), DBlock(base_channel * 2, num_res), DBlock(base_channel, num_res)]
        )

        self.Convs = nn.ModuleList(
            [
                BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
                BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
            ]
        )

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([AFF(base_channel * 7, base_channel * 1), AFF(base_channel * 7, base_channel * 2)])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x_ir, x_vis):
        """Forward pass for MB2BGAN generator.
        Args:
            x_ir: Infrared image (batch, 3, height, width).
            x_vis: Visible image (batch, 3, height, width).
        Returns:
            List of multi-scale outputs (coarse to fine).
        """
        x_vis = F.interpolate(x_vis, size=x_ir.shape[2:], mode='bilinear', align_corners=False)

        x_2 = F.interpolate(x_ir, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        outputs = list()

        # Feature extraction and fusion
        x_ir_feat = self.feat_extract[0](x_ir)
        x_vis_feat = self.feat_extract[0](x_vis)
        x_fused = self.MAF1(x_ir_feat, x_vis_feat)
        res1 = self.Encoder[0](x_fused)

        z_ir = self.feat_extract[1](res1)
        z_vis = self.feat_extract[1](x_vis_feat)
        z_fused = self.MAF2(z_ir, z_vis)
        res2 = self.Encoder[1](z_fused)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.MAF3(z)
        z = self.Encoder[2](z)

        # Multi-scale feature aggregation
        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        # Decoder
        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + x_ir)

        return outputs


class MB2BGANPlus(nn.Module):
    def __init__(self, num_res=20):
        super(MB2BGANPlus, self).__init__()
        base_channel = 32

        # Encoder
        self.Encoder = nn.ModuleList(
            [
                EBlock(base_channel, num_res),
                EBlock(base_channel * 2, num_res),
                EBlock(base_channel * 4, num_res),
            ]
        )

        self.feat_extract = nn.ModuleList(
            [
                BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
                BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
                BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
                BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
                BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
                BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.Decoder = nn.ModuleList(
            [DBlock(base_channel * 4, num_res), DBlock(base_channel * 2, num_res), DBlock(base_channel, num_res)]
        )

        self.Convs = nn.ModuleList(
            [
                BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
                BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
            ]
        )

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([AFF(base_channel * 7, base_channel * 1), AFF(base_channel * 7, base_channel * 2)])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(self, x):
        """Forward pass for MB2BGANPlus (single-modal input).
        Args:
            x: Input image (batch, 3, height, width).
        Returns:
            List of multi-scale outputs (coarse to fine).
        """
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + x)

        return outputs


def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MB2BGANPlus":
        return MB2BGANPlus()
    elif model_name == "MB2BGAN":
        return MB2BGAN()
    raise ModelError("Wrong Model!\nYou should choose MB2BGANPlus or MB2BGAN.")

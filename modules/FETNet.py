# !usr/bin/env python

# -*- coding:utf-8 _*-

"""
@Author:GuangtaoLyu
@Github:https://github.com/GuangtaoLyu
@time: 2021/5/13 13:21
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels,kernel_size=3,dilation=1,padding=1, use_res=False,down=True):
        super().__init__()
        self.use_res = use_res
        self.use_down = down
        self.down = nn.MaxPool2d(2)
        self.resblock = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,dilation=dilation, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        if self.use_down:
            x = self.down(x)
        return self.resblock(x)+self.conv2(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels,kernel_size=3, up=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return  self.conv2(x1)


class Gate_Conv_Module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(Gate_Conv_Module, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x, x_mask = torch.split(x, self.out_channels - 1, dim=1)
        x = self.relu(x)
        x_mask = self.sigmoid(x_mask)
        x = x * x_mask
        return x, x_mask


class Text_Texture_Erase_And_Enhance_Module(nn.Module):
    def __init__(self, in_channels):
        super(Text_Texture_Erase_And_Enhance_Module, self).__init__()
        self.gconv1 = Gate_Conv_Module(in_channels + 1, in_channels + 1, kernel_size=7, stride=1, padding=3)
        self.gconv2 = Gate_Conv_Module(in_channels + 1, in_channels + 1, kernel_size=5, stride=1, padding=2)
        self.gconv3 = Gate_Conv_Module(in_channels + 1, in_channels + 1, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)
        self.maxpool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels, in_channels , kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

        self.conv4 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels , in_channels, kernel_size=1)


    def forward(self, x1, x2, x3, x_mask):
        m_batchsize, c, width, height = x3.size()
        epson = 1e-7
        x1_1 = self.maxpool(x1)
        x1_1 = self.maxpool(x1_1)
        x1_1 = self.conv4(x1_1)
        x2_1 = self.maxpool(x2)
        x2_1 = self.conv5(x2_1)
        f1 = torch.cat([x1_1, x2_1, x3], dim=1)
        f1 = self.conv(f1)
        f1 = f1 * x_mask
        f1 = torch.cat([f1, x_mask], dim=1)
        x_1, x_mask_1 = self.gconv3(f1)
        x_1, x_mask_1 = self.gconv2(torch.cat([x_1,x_mask_1],dim=1))
        x_1, x_mask_1 = self.gconv1(torch.cat([x_1,x_mask_1],dim=1))
        f1 = x_1.view(m_batchsize, -1, width * height)
        b_att = torch.bmm(f1.permute(0, 2, 1), f1)
        f1 = x_mask.view(m_batchsize, -1, width * height)
        mask_att = torch.bmm(f1.permute(0, 2, 1), f1)
        b_att = b_att * mask_att
        b_att = b_att.view(m_batchsize, -1, width, height)

        b_att = self.softmax(b_att)
        b_att =F.avg_pool2d(b_att, 3, 1, padding = 1)
        b_att_1 = b_att.view(m_batchsize, -1, height * width)

        x_out = x_1.view(m_batchsize, -1, width * height)
        x_out = (torch.bmm(x_out, b_att_1.permute(0, 2, 1))).view(m_batchsize, c, width, height)
        out = x_1 * x_mask + (1 - x_mask) * x_out

        x_3 = x3.view(m_batchsize, -1, width * height)
        x_3 = (torch.bmm(x_3, b_att_1.permute(0, 2, 1))).view(m_batchsize, c, width, height)
        o3 = x3 * x_mask + (1 - x_mask) * x_3

        x_mask = self.up(x_mask)
        x2_1 = x2_1.view(m_batchsize, -1, width * height)
        x2_1 = (torch.bmm(x2_1, b_att_1.permute(0, 2, 1))).view(m_batchsize, -1, width, height)
        x2_1 = self.up(x2_1)
        x2_1 = self.conv2(x2_1)
        x2_1 = self.relu(x2_1)
        o2 = x2 * x_mask + (1 - x_mask) * x2_1

        x_mask = self.up(x_mask)
        x1_1 = x1_1.view(m_batchsize, -1, width * height)
        x1_1 = (torch.bmm(x1_1, b_att_1.permute(0, 2, 1))).view(m_batchsize, -1, width, height)
        x1_1 = self.up(x1_1)
        x1_1 = self.up(x1_1)
        x1_1 = self.conv3(x1_1)
        x1_1 = self.relu(x1_1)
        o1 = x1 * x_mask + (1 - x_mask) * x1_1

        return o1, o2, o3, out, x_mask


class Text_Structure_Erase_And_Enhance_Module(nn.Module):
    def __init__(self, in_channels):
        super(Text_Structure_Erase_And_Enhance_Module, self).__init__()
        self.ca = ChannelAttention(in_channels)

    def forward(self, x, x_mask):
        # erase
        x1 = x * x_mask
        # fill
        att_2 = self.ca(x1)
        # enhance
        x = x*att_2
        return x

class Get_mask(nn.Module):
    def __init__(self):
        super(Get_mask, self).__init__()
        self.conv1 = Up(768,256)
        self.conv2 = Up(384,128)
        self.mask_get = nn.Sequential(
            nn.Conv2d(128,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x4, x5,x6):
        x5 = self.conv1(x6,x5)
        x4 = self.conv2(x5,x4)
        return self.mask_get(x4)


class FETNet(nn.Module):
    def __init__(self, n_channels):
        super(FETNet, self).__init__()
        # encode
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.down1 = Down(64, 128,5,padding=2)
        self.down2 = Down(128, 128,3)
        self.down3 = Down(128, 256,3)
        self.down4 = Down(256, 512,3)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.maxpool = nn.MaxPool2d(2)

        self.mask_get = Get_mask()

        self.tfe_c = Text_Texture_Erase_And_Enhance_Module(128)
        self.tfe4 = Text_Structure_Erase_And_Enhance_Module(256)
        self.tfe5 = Text_Structure_Erase_And_Enhance_Module(512)
        # decode
        self.up1 = Up(896, 256)
        self.up2 = Up(384, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # encode
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # encode

        # mask
        x_mask_out = self.mask_get(x3,x4,x5)
        # mask

        # transfer
        x_1, x_2, x_3, out, mask_out = self.tfe_c(x1, x2, x3, x_mask_out)
        out = self.maxpool(out)
        out = self.maxpool(out)
        x_mask = self.maxpool(x_mask_out)
        x_4 = self.tfe4(x4, x_mask)
        x_mask = self.maxpool(x_mask)
        x_5 = self.tfe5(x5, x_mask)
        x_5_c = torch.cat([x_5,out],dim=1)
        # transfer

        x_out = self.up1(x_5_c, x_4)
        x_out = self.up2(x_out, x_3)
        x_out = self.up3(x_out, x_2)
        x_out = self.up4(x_out, x_1)
        x_out_1 = self.out_conv(x_out)


        return x_out_1,mask_out


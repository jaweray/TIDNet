"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import cv2


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat * 3, scale_unetfeats * 3)
        self.up_dec1 = UpSample(n_feat * 3, scale_unetfeats * 3)

        self.up_enc2 = nn.Sequential(UpSample(n_feat * 3 + scale_unetfeats * 3, scale_unetfeats * 3),
                                     UpSample(n_feat * 3, scale_unetfeats * 3))
        self.up_dec2 = nn.Sequential(UpSample(n_feat * 3 + scale_unetfeats * 3, scale_unetfeats * 3),
                                     UpSample(n_feat * 3, scale_unetfeats * 3))

        self.conv_enc1 = nn.Conv2d(3 * n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(3 * n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(3 * n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(3 * n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(3 * n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(3 * n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


##########################################################################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


##########################################################################
class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.down12 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 64)
        self.up34 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = DoubleConv(64+64, out_channels, mid_channels=32)

    def forward(self, x):
        x1 = self.conv1(x)
        temp = self.down12(x1)
        x2 = self.conv2(temp)
        x3 = self.conv3(x2)
        temp = self.up34(x3)
        x4 = self.conv4(torch.cat([x1, temp], dim=1))
        return x4


##########################################################################
class TIDNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3,
                 reduction=4, bias=False):
        super(TIDNet, self).__init__()

        ## sobel kernel
        k_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        k_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        # self.soble_kernel_x = k_x.unsqueeze(0).repeat(1, 1, 1, 1)
        # self.soble_kernel_y = k_y.unsqueeze(0).repeat(1, 1, 1, 1)
        self.register_buffer('soble_kernel_x', k_x.unsqueeze(0).repeat(1, 1, 1, 1))
        self.register_buffer('soble_kernel_y', k_y.unsqueeze(0).repeat(1, 1, 1, 1))

        act = nn.PReLU()
        self.shallow_feat1_g = nn.Sequential(conv(5 * in_c // 3, n_feat, kernel_size, bias=bias))
        self.shallow_feat1_b = nn.Sequential(conv(5 * in_c // 3, n_feat, kernel_size, bias=bias))
        self.shallow_feat1_r = nn.Sequential(conv(5 * in_c // 3, n_feat, kernel_size, bias=bias))

        self.conv_g = nn.Sequential(conv(1, 10, kernel_size, bias=bias), conv(10, 10, kernel_size, bias=bias),
                                    conv(10, 1, kernel_size, bias=bias), nn.BatchNorm2d(1))
        self.conv_b = nn.Sequential(conv(1, 10, kernel_size, bias=bias), conv(10, 10, kernel_size, bias=bias),
                                    conv(10, 1, kernel_size, bias=bias), nn.BatchNorm2d(1))
        self.conv_r = nn.Sequential(conv(1, 10, kernel_size, bias=bias), conv(10, 10, kernel_size, bias=bias),
                                    conv(10, 1, kernel_size, bias=bias), nn.BatchNorm2d(1))

        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder_b = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder_b = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage1_encoder_g = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder_g = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage1_encoder_r = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder_r = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats,
                                    num_cab)

        self.sam1 = SAM(n_feat * 3, kernel_size=1, bias=bias)

        self.concat12 = conv(n_feat * 4, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail = conv(n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)
        # self.tail2 = nn.Sequential(DoubleConv(2 * out_c, 64),
        #                            DoubleConv(64, 32),
        #                            conv(32, out_c, kernel_size=3))
        self.tail2 = UBlock(2 * out_c, out_c)
        # self.tail2 = conv(2 * out_c, out_c, kernel_size, bias=bias)


    def forward(self, x_img):

        x_b, x_g, x_r = torch.chunk(x_img, 3, dim=1)
        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------

        ## compute edge and diffrence of each channel
        edge_b = 0.5 * abs(sum(self.Sobel(x_b)))
        edge_g = 0.5 * abs(sum(self.Sobel(x_g)))
        edge_r = 0.5 * abs(sum(self.Sobel(x_r)))

        edge = torch.cat((abs(edge_b - edge_g), abs(edge_r - edge_g)), dim=1)

        d_r = self.conv_r(x_r)
        d_g = self.conv_g(x_g)
        d_b = self.conv_b(x_b)

        sobel_d_bg = 0.5 * abs(sum(self.Sobel(d_b - d_g)))
        sobel_d_rg = 0.5 * abs(sum(self.Sobel(d_r - d_g)))

        diff = torch.cat((d_b - d_g, d_r - d_g), dim=1)
        # sv(d_r, 'R')
        # sv(d_g, 'G')
        # sv(d_b, 'B')
        # sv(abs(d_b - d_g), 'B-G')
        # sv(abs(d_r - d_g), 'R-G')

        ## cat edge and difference to each channel
        x_b_cat = torch.cat((x_b, edge, diff), dim=1)
        x_g_cat = torch.cat((x_g, edge, diff), dim=1)
        x_r_cat = torch.cat((x_r, edge, diff), dim=1)

        ## Compute Shallow Features
        shallow_feat1_b = self.shallow_feat1_b(x_b_cat)
        shallow_feat1_g = self.shallow_feat1_g(x_g_cat)
        shallow_feat1_r = self.shallow_feat1_r(x_r_cat)

        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_b = self.stage1_encoder_b(shallow_feat1_b)
        feat1_g = self.stage1_encoder_g(shallow_feat1_g)
        feat1_r = self.stage1_encoder_r(shallow_feat1_r)

        ## Pass features through Decoder of Stage 1
        res1_b = self.stage1_decoder_b(feat1_b)
        res1_g = self.stage1_decoder_g(feat1_g)
        res1_r = self.stage1_decoder_r(feat1_r)

        feat1 = [torch.cat((k, v, l), 1) for k, v, l in zip(feat1_b, feat1_g, feat1_r)]
        res1 = [torch.cat((k, v, l), 1) for k, v, l in zip(res1_b, res1_g, res1_r)]

        ## Apply Supervised Attention Module (SAM)
        ## Output image at Stage 1
        x2_samfeats, stage1_img = self.sam1(res1[0], x_img)

        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2 = self.shallow_feat2(x_img)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x2_cat = self.concat12(torch.cat([x2, x2_samfeats], 1))

        x2_cat = self.stage2_orsnet(x2_cat, feat1, res1)

        residual = self.tail(x2_cat)
        # stage2_img = self.tail2(torch.cat((residual + x_img, x_img), dim=1))

        return [residual + x_img, stage1_img], [d_b - d_g, d_r - d_g, x_b-x_g, x_r-x_g, (d_b, d_g, d_r)]

    def Sobel(self, img):
        _, _, kw, kh = self.soble_kernel_x.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.soble_kernel_x, groups=1), F.conv2d(img, self.soble_kernel_y, groups=1)


def sv(ocr_area, name):
    ocr_area = torch.clamp(ocr_area, 0, 1)
    img = ocr_area.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255
    cv2.imwrite('./tt/{}.png'.format(name), img)

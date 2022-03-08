# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#08.09 change pad

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer, equal_lr
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SEACEResnetBlock as SEACEResnetBlock
from models.networks.architecture import Ada_SPADEResnetBlock as Ada_SPADEResnetBlock
from models.networks.architecture import Attention
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm1d

class SEACEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = 64
        self.sw, self.sh = self.compute_latent_vector_size(opt)

        ic = opt.semantic_nc
        self.ref1_init = nn.Conv2d(3, ic, 3, stride=1, padding=1)

        self.fc = nn.Conv2d(8 * nf, 16 * nf, 3, stride=1, padding=1)
        # , 256, 512, 256, 256
        self.G_head_0 = SEACEResnetBlock(16 * nf, 16 * nf, opt, feat_nc=256)

        self.G_middle_0 = SEACEResnetBlock(16 * nf, 16 * nf, opt, feat_nc=256)
        self.G_middle_1 = SEACEResnetBlock(16 * nf, 16 * nf, opt, feat_nc=512)

        self.G_up_0 = SEACEResnetBlock(16 * nf, 8 * nf, opt, feat_nc=256)
        self.G_up_1 = SEACEResnetBlock(8 * nf, 4 * nf, opt, feat_nc=256)
        self.attn = Attention(4 * nf, 'spectral' in opt.norm_G)

        self.G_out_0 = SEACEResnetBlock(4 * nf, 2 * nf, opt, feat_nc=128)
        self.G_out_1 = SEACEResnetBlock(2 * nf, 1 * nf, opt, feat_nc=ic)

        self.conv_img1 = nn.Conv2d(1 * nf, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        # self.conv_confi1 = nn.Conv2d(1 * nf, 3, 3, padding=1)

    def compute_latent_vector_size(self, opt):
        num_up_layers = 5
        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)
        return sw, sh

    def forward(self, warp_out=None):

        seg_feat1, seg_feat2, seg_feat3, seg_feat4, seg_feat5, \
        ref_feat1, ref_feat2, ref_feat3, ref_feat4, ref_feat5, cmap = warp_out
        #  3, 128, 256, 256, 256
        ref_feat1 = self.ref1_init(ref_feat1)

        cmap1 = F.interpolate(cmap.repeat(1, seg_feat1.shape[1], 1, 1), size=seg_feat1.size()[2:], mode='nearest')  # 8
        cmap2 = F.interpolate(cmap.repeat(1, seg_feat2.shape[1], 1, 1), size=seg_feat2.size()[2:], mode='nearest')  # 16
        cmap3 = F.interpolate(cmap.repeat(1, seg_feat3.shape[1], 1, 1), size=seg_feat3.size()[2:], mode='nearest')  # 32
        cmap4 = F.interpolate(cmap.repeat(1, seg_feat4.shape[1], 1, 1), size=seg_feat4.size()[2:], mode='nearest')  #
        cmap5 = F.interpolate(cmap.repeat(1, seg_feat5.shape[1], 1, 1), size=seg_feat5.size()[2:], mode='nearest')  # 8

        ref_feat1 = F.interpolate(ref_feat1, size=seg_feat1.size()[2:])
        ref_feat2 = F.interpolate(ref_feat2, size=seg_feat2.size()[2:])
        ref_feat3 = F.interpolate(ref_feat3, size=seg_feat3.size()[2:])
        ref_feat4 = F.interpolate(ref_feat4, size=seg_feat4.size()[2:])
        ref_feat5 = F.interpolate(ref_feat5, size=seg_feat5.size()[2:])

        fusion1 = seg_feat1 * (1-cmap1) + ref_feat1 * cmap1
        fusion2 = seg_feat2 * (1-cmap2) + ref_feat2 * cmap2
        fusion3 = seg_feat3 * (1-cmap3) + ref_feat3 * cmap3
        fusion4 = seg_feat4 * (1-cmap4) + ref_feat4 * cmap4
        fusion5 = seg_feat5 * (1-cmap5) + ref_feat5 * cmap5

        x = torch.cat((seg_feat5, ref_feat5), 1)
        x = F.interpolate(x, size=(self.sh, self.sw))
        x = self.fc(x)

        x = self.G_head_0(x, fusion5)  # 8
        x = self.up(x)
        x = self.G_middle_0(x, fusion5) # 16
        x = self.G_middle_1(x, fusion4)
        x = self.up(x)

        x = self.G_up_0(x, fusion3) # 32
        x = self.up(x)
        x = self.G_up_1(x, fusion3) # 64
        x = self.attn(x)
        x = self.up(x) # 128
        x = self.G_out_0(x, fusion2) # 128
        x = self.up(x)
        x = self.G_out_1(x, fusion1) # 256

        x = self.conv_img1(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x





class AdaptiveFeatureGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks")
        return parser

    def __init__(self, opt):
        # TODO: kernel=4, concat noise, or change architecture to vgg feature pyramid
        super().__init__()
        self.opt = opt
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 64
        nf = 64
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(opt.spade_ic, ndf, kw, stride=1, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, opt.adaptor_kernel, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=1, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, opt.adaptor_kernel, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt
        self.head_0 = Ada_SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)

        # self.head_1 = Ada_SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)
        if opt.adaptor_nonlocal:
            self.attn = Attention(8 * nf, False)
        self.G_middle_0 = Ada_SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)
        self.G_middle_1 = Ada_SPADEResnetBlock(8 * nf, 4 * nf, opt, use_se=opt.adaptor_se)

        self.deeper2 = Ada_SPADEResnetBlock(4 * nf, 4 * nf, opt, dilation=4)
        self.degridding0 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=2, dilation=2))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, input, seg, multi=False):
        x = self.layer1(input)
        x = self.layer2(self.actvn(x))
        x2 = x  # 128

        x = self.layer3(self.actvn(x))
        x3 = x  # 128

        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.head_0(x, seg)
        x4 = x  # 64

        # x = self.head_1(x, seg)
        if self.opt.adaptor_nonlocal:
            x = self.attn(x)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x5 = x  # 64

        # x = self.deeper0(x, seg)
        # x = self.deeper1(x, seg)
        x = self.deeper2(self.up(x), seg)
        x = self.degridding0(x)

        if multi == True:
            return x2, x3, x4, x5, x     #  , 256, 512, 256, 256
        else:
            return x5, x


class DomainClassifier(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        nf = opt.ngf
        kw = 4 if opt.domain_rela else 3
        pw = int((kw - 1.0) / 2)
        self.feature = nn.Sequential(nn.Conv2d(4 * nf, 2 * nf, kw, stride=2, padding=pw),
                                SynchronizedBatchNorm2d(2 * nf, affine=True),
                                nn.LeakyReLU(0.2, False),
                                nn.Conv2d(2 * nf, nf, kw, stride=2, padding=pw),
                                SynchronizedBatchNorm2d(nf, affine=True),
                                nn.LeakyReLU(0.2, False),
                                nn.Conv2d(nf, int(nf // 2), kw, stride=2, padding=pw),
                                SynchronizedBatchNorm2d(int(nf // 2), affine=True),
                                nn.LeakyReLU(0.2, False))  #32*8*8
        model = [nn.Linear(int(nf // 2) * 8 * 8, 100),
                SynchronizedBatchNorm1d(100, affine=True),
                nn.ReLU()]
        if opt.domain_rela:
            model += [nn.Linear(100, 1)]
        else:
            model += [nn.Linear(100, 2),
                      nn.LogSoftmax(dim=1)]
        self.classifier = nn.Sequential(*model)

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x.view(x.shape[0], -1))
        return x

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                decay = self.mu
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
                
    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]

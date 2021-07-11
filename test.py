# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import data
import numpy as np
from util.util import masktorgb
from options.test_options import TestOptions
# from options.train_options import TrainOptions
from models.pix2pix_model import Pix2PixModel

opt = TestOptions().parse()
   
torch.manual_seed(0)
dataloader = data.create_dataloader(opt)
dataloader.dataset[0]

model = Pix2PixModel(opt)
model.eval()

# save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output')
# save_root = '/data/vdd/fangneng.zfn/SFERT5/' + opt.name
save_root = opt.checkpoints_dir.split('checkpoints')[0] + 'results/' + opt.name + '1/'

if not os.path.exists(save_root + '/pre'):
    os.makedirs(save_root + '/pre')
    os.makedirs(save_root + '/gt')
    os.makedirs(save_root + '/warp')

    # print (1/0)

# test
for i, data_i in enumerate(dataloader):
    print('{} / {}'.format(i, len(dataloader)))
    # if i * opt.batchSize >= 4993:
    # if i * opt.batchSize >= 400:
    #     break
    imgs_num = data_i['label'].shape[0]
    # out = model(data_i, mode='inference')
    out = model(data_i, mode='inference')

    pre = out['fake_image'].data.cpu()
    gt = data_i['image'].cpu()
    ref = data_i['ref'].cpu()
    label = data_i['label'][:, :1, :, :] + 0.5
    warp = out['warp128'].data.cpu()

    batch_size = pre.shape[0]

    for j in range(batch_size):
        pre_ = pre[j]
        gt_ = gt[j]
        ref_ = ref[j]
        label_ = label
        warp_ = warp[j]

        pre_ = (pre_ + 1) / 2
        vutils.save_image(pre_, save_root + '/pre/' + str(i) + '_' + str(j) + '.jpg',
                nrow=imgs_num, padding=0, normalize=False)

        gt_ = (gt_ + 1) / 2
        vutils.save_image(gt_, save_root + '/gt/' + str(i) + '_' + str(j) + '.jpg',
                          nrow=imgs_num, padding=0, normalize=False)

        # ref_ = (ref_ + 1) / 2
        # vutils.save_image(ref_, save_root + '/pre/' + str(i) + '_' + str(j) + '_ref.jpg',
        #                   nrow=imgs_num, padding=0, normalize=False)
        #
        # print (label_.shape)
        # label_ = masktorgb(label_.cpu().numpy())
        # label_ = torch.from_numpy(label_).float() / 128 - 1
        # vutils.save_image(label_[j], save_root + '/pre/' + str(i) + '_' + str(j) + '_label.png',
        #                   nrow=imgs_num, padding=0, normalize=False)

        warp_ = (warp_ + 1) / 2
        vutils.save_image(warp_, save_root + '/warp/' + str(i) + '_' + str(j) + '.jpg',
                          nrow=imgs_num, padding=0, normalize=False)

        # print(1/0)


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from data.base_dataset import BaseDataset, get_params, get_transform
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import util.util as util
import os
import random
import numpy as np
#from scipy.ndimage.filters import gaussian_filter


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths = self.get_paths(opt)

        if opt.dataset_mode != 'celebahq' and opt.dataset_mode != 'deepfashion':
            util.natural_sort(label_paths)
            util.natural_sort(image_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths

        size = len(self.label_paths)
        self.dataset_size = size

        self.real_reference_probability = 1 if opt.phase == 'test' else opt.real_reference_probability
        self.hard_reference_probability = 0 if opt.phase == 'test' else opt.hard_reference_probability
        self.ref_dict, self.train_test_folder = self.get_ref(opt)

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def get_label_tensor(self, path):
        label = Image.open(path)
        params1 = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params1, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0

        # print (label_tensor)
        spe = self.semantic_position_encoding(label_tensor)
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        return spe, label_tensor, params1

    def semantic_position_encoding(self, label):
        label = label.unsqueeze(0)
        label = F.interpolate(label, size=(64, 64), mode='nearest')
        label = label.squeeze()

        h, w = label.shape
        coord = torch.zeros((3, h, w))
        x = torch.arange(0, w, 1.0) / 255.0
        y = torch.arange(0, h, 1.0) / 255.0
        xx, yy = torch.meshgrid(x, y)

        ids = torch.unique(label)
        for id in ids:
            mask = (label == id).float()
            xx_masked = xx * mask
            yy_masked = yy * mask

            num, xx_sum, yy_sum = mask.sum(), xx_masked.sum(), yy_masked.sum()
            xx_center, yy_center = xx_sum / num, yy_sum / num

            # print(id, xx_center, yy_center)
            xx_relative = (xx_masked - xx_center) * mask
            yy_relative = (yy_masked - yy_center) * mask
            rr_relative = torch.sqrt((torch.pow(xx_relative, 2) + torch.pow(yy_relative, 2)).float())

            # print(mask.shape)
            coord[0, :, :] += xx_relative.float()
            coord[1, :, :] += yy_relative.float()
            coord[2, :, :] += rr_relative.float()

        return coord


    def __getitem__(self, index):
        # LabelImage
        label_path = self.label_paths[index]
        image_path = self.image_paths[index]

        # label_path_ = '/home/fangneng.zfn/projects/ICCV2021/Editing/util/seg.png'

        label_spe, label_tensor, params1 = self.get_label_tensor(label_path)

        if not self.opt.no_pairing_check:
            assert self.paths_match(label_path, image_path), \
                "The label_path %s and image_path %s don't match." % \
                (label_path, image_path)
        image = Image.open(image_path).convert('RGB')
        transform_image = get_transform(self.opt, params1)
        image_tensor = transform_image(image)

        ref_tensor = 0
        label_ref_tensor = 0

        random_p = random.random()
        if random_p < self.real_reference_probability or self.opt.phase == 'test':
        # if 0:
            key = image_path.replace('\\', '/').split('DeepFashion/')[-1] if self.opt.dataset_mode == 'deepfashion' else os.path.basename(image_path)
            val = self.ref_dict[key]
            if random_p < self.hard_reference_probability:
                path_ref = val[1]  #hard reference
            else:
                path_ref = val[0] #easy reference
            if self.opt.dataset_mode == 'deepfashion':
                path_ref = os.path.join(self.opt.dataroot, path_ref)
            else:
                path_ref = os.path.dirname(image_path).replace(self.train_test_folder[1], self.train_test_folder[0]) + '/' + path_ref

            image_ref = Image.open(path_ref).convert('RGB')
            if self.opt.dataset_mode == 'ade20k' or self.opt.dataset_mode == 'celebahqedge' or self.opt.dataset_mode == 'ade20klayout':
                path_ref_label = path_ref.replace('.jpg', '.png')
                path_ref_label = self.imgpath_to_labelpath(path_ref_label)
            elif self.opt.dataset_mode == 'coco' or self.opt.dataset_mode == 'cocolayout':
                path_ref_label = path_ref.replace('.jpg', '.png').replace('image', 'label')
                path_ref_label = self.imgpath_to_labelpath(path_ref_label)
            else:
                path_ref_label = self.imgpath_to_labelpath(path_ref)

            ref_spe, label_ref_tensor, params = self.get_label_tensor(path_ref_label)
            transform_image = get_transform(self.opt, params)
            ref_tensor = transform_image(image_ref)
            #ref_tensor = self.reference_transform(image_ref)
            self_ref_flag = torch.zeros_like(ref_tensor)
        else:
            pair = False
            if self.opt.dataset_mode == 'deepfashion' and self.opt.video_like:
                # if self.opt.hdfs:
                #     key = image_path.split('DeepFashion.zip@/')[-1]
                # else:
                #     key = image_path.split('DeepFashion/')[-1]
                key = image_path.replace('\\', '/').split('DeepFashion/')[-1]
                val = self.ref_dict[key]
                ref_name = val[0]

                key_name = key
                if os.path.dirname(ref_name) == os.path.dirname(key_name) and os.path.basename(ref_name).split('_')[0] == os.path.basename(key_name).split('_')[0]:
                    path_ref = os.path.join(self.opt.dataroot, ref_name)
                    image_ref = Image.open(path_ref).convert('RGB')
                    label_ref_path = self.imgpath_to_labelpath(path_ref)
                    ref_spe, label_ref_tensor, params = self.get_label_tensor(label_ref_path)
                    transform_image = get_transform(self.opt, params)
                    ref_tensor = transform_image(image_ref)
                    pair = True
            if not pair:
                ref_spe, label_ref_tensor, params = self.get_label_tensor(label_path)
                transform_image = get_transform(self.opt, params)
                ref_tensor = transform_image(image)
            #ref_tensor = self.reference_transform(image)
            self_ref_flag = torch.ones_like(ref_tensor)


        input_dict = {'label': label_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      'self_ref': self_ref_flag,
                      'ref': ref_tensor,
                      'label_ref': label_ref_tensor,
                      'label_spe': label_spe,
                      'ref_spe': ref_spe
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size

    def get_ref(self, opt):
        pass

    def imgpath_to_labelpath(self, path):
        return path

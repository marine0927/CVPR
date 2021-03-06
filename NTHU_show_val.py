# -*- coding:utf-8 -*-
import torch
import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
# from model.ada_deeplab_multi import Res_Deeplab
from dataset.mixed_dataset import MixedDataSet
from collections import OrderedDict
import os
from PIL import Image
import argparse
import json
from os.path import join

import matplotlib.pyplot as plt
import torch.nn as nn

MODEL_NAME = 'DeepLabV2'

if MODEL_NAME=='DeepLabV2':
    # from model.deeplab_multi_class import Res_Deeplab
    from model.deeplab_multi import Res_Deeplab
elif MODEL_NAME=='DeepLabV3':
    from model.deeplabv3 import Res_Deeplab

ROME_IMG_MEAN = np.array((116.12734586, 116.29007402, 111.07457307), dtype=np.float32)
RIO_IMG_MEAN = np.array((116.56478136, 117.95337218, 113.50515509), dtype=np.float32)
TOKYO_IMG_MEAN = np.array((112.59267422, 113.11990604, 109.67289213), dtype=np.float32)
TAIPEI_IMG_MEAN = np.array((105.58076453, 106.50068834, 104.46193236), dtype=np.float32)
IMG_MEAN=TOKYO_IMG_MEAN   #city 1
DATA_DIRECTORY = 'G:/data/NTHU_Datasets/Tokyo/Images'  #city 2
DATA_LIST_PATH = './dataset/NTHU_list/TOKYO_test.txt'  #city 3
gtDir='G:/data/NTHU_Datasets/Tokyo/Labels/Test'  #city 4
devkitDir='./dataset/NTHU_list'

NUM_CLASSES = 19
# RESTORE_FROM='./snapshots/retrain_mixed3/GTA5_88000.pth'
# SET = 'train'
SET = 'Test'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


# def label_mapping(input, mapping):
#     output = np.copy(input)
#     for ind in range(len(mapping)):
#         output[input == mapping[ind][0]] = mapping[ind][1]
#     return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):
    """
    Compute IoU given the predicted colorized images and
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])
    name_classes = np.array(info['label'], dtype=np.str)
    # mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'TOKYO_test.txt') #city 5
    label_path_list = join(devkit_dir, 'TOKYO_test.txt')  #city 6
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x[:-4]+'_city.png') for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x) for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        # pred = np.array(Image.open(pred_imgs[ind][:-4] + '.png'))
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        # label = label_mapping(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
        #     print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
        #                                                                           len(pred.flatten()), gt_imgs[ind],
        #                                                                           pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # if ind > 0 and ind % 10 == 0:
        #     print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    print (mIoUs)
    syn_mIoUs = np.delete(mIoUs, [3, 4, 5, 9, 14, 16])
    print (syn_mIoUs)
    return mIoUs,syn_mIoUs

def show_val(seg_state_dict, show_pred_sv_dir, city='ROME'):
    """Create the model and start the evaluation process."""

    # args = get_arguments()
    save_dir=show_pred_sv_dir
    gpu0 = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = Res_Deeplab(num_classes=NUM_CLASSES)
    model.load_state_dict(seg_state_dict)

    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(MixedDataSet(DATA_DIRECTORY, DATA_LIST_PATH, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=SET),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear')
    # n=0
    for index, batch in enumerate(testloader):
        # n+=1
        # if n>3:
        #     continue
        image, _, name = batch
        # _, output1, output2 = model(Variable(image, requires_grad=True).cuda(gpu0)) #ada_deeplab_multi
        output1, output2 = model(Variable(image, requires_grad=True).cuda(gpu0))
        output = interp(output2).cpu().data[0].numpy()

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)

        name = name[0].split('/')[-1]
        output_col.save('%s/%s.png' % (save_dir, name.split('.')[0]))
    print('colored pictures saving is done')
    mIoUs, syn_mIoUs=compute_mIoU(gtDir, save_dir, devkitDir)
    return mIoUs, syn_mIoUs


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
from model.deeplab_multi import Res_Deeplab
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.gta5_dataset import GTA5DataSet
from collections import OrderedDict
import os
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
SAVE_PATH = '../result/adapt_slective/adapt_iter2/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# cityscapes settings
CITY_IMG_MEAN= np.array((73.15835921, 82.90891754,72.39239876), dtype=np.float32)
CITY_DATA_DIRECTORY = '/home/zq/dl-test/ZQAdaptSegNet-master/data/Cityscapes/data/'
CITY_DATA_LIST_PATH = '../dataset/cityscapes_list/train.txt'
citys_feat_distr_path=SAVE_PATH+'citys_feat_distr.npy'  #cityscapes
citys_imgpaths_path=SAVE_PATH+'citys_img_paths.npy'

# GTA5 settings
GTA_IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
GTA_DATA_DIRECTORY = '/home/zq/dl-test/ZQAdaptSegNet-master/data/GTA5/'
GTA_DATA_LIST_PATH = '../dataset/gta5_list/train-full.txt'
gta_feat_distr_path=SAVE_PATH+'gta_feat_distr.npy'
gta_imgpaths_path=SAVE_PATH+'gta_img_paths.npy'

closest_imgs_path=os.path.join(SAVE_PATH,'closest.npy')
src_train_imgs_txt=os.path.join(SAVE_PATH,'train_closest.txt')

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM='/mnt/sda/zq2/ZQAdaptSegNet/snapshots/gta2citys_pt/GTA5_265000.pth'
SET = 'train'
# SET = 'val'

# palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
#            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
#            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
# zero_pad = 256 * 3 - len(palette)
# for i in range(zero_pad):
#     palette.append(0)
#
#
#
# def colorize_mask(mask):
#     # mask: numpy array of the mask
#     new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
#     new_mask.putpalette(palette)
#
#     return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--data-dir", type=str, default=CITY_DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=CITY_DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    gpu0 = args.gpu

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = Res_Deeplab(num_classes=args.num_classes)

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    if os.path.isfile(citys_feat_distr_path)==False:
        testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=CITY_IMG_MEAN, scale=False, mirror=False, set=args.set),
                                        batch_size=1, shuffle=False, pin_memory=True)

        # interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
        interp_down = nn.Upsample(size=(16, 32), mode='bilinear', align_corners=True)
        citys_feat_distrs=[]
        citys_img_paths=[]
        for index, batch in enumerate(testloader):
            if index % 100 == 0:
                print('%d processd of %d' % (index,len(testloader)))
            image, _, name = batch
            output1, output2 = model(Variable(image, volatile=True).cuda(gpu0))
            output = interp_down(output2).cpu().data[0].numpy()
            output = output.transpose(1,2,0)

            output = output[np.newaxis, :]  # add a dim
            citys_feat_distrs.extend(output)
            citys_img_paths.extend(name)

            #name: 'frankfurt/frankfurt_000001_007973_leftImg8bit.png'
            # name = name[0].split('/')[-1]
        citys_feat_distrs_np = np.array(citys_feat_distrs)
        citys_img_paths_np = np.array(citys_img_paths)
        np.save(citys_feat_distr_path, citys_feat_distrs_np)
        np.save(citys_imgpaths_path, citys_img_paths_np)
    else:
        citys_feat_distrs_np = np.load(citys_feat_distr_path)
        citys_img_paths_np = np.load(citys_imgpaths_path)

    if os.path.isfile(gta_feat_distr_path) == False:
        gtaloader = data.DataLoader(
            GTA5DataSet(GTA_DATA_DIRECTORY, GTA_DATA_LIST_PATH, crop_size=(1024, 512), mean=GTA_IMG_MEAN, scale=False,
                              mirror=False),
            batch_size=1, shuffle=False, pin_memory=True)

        interp_down = nn.Upsample(size=(16, 32), mode='bilinear', align_corners=True)
        gta_feat_distrs = []
        gta_img_paths = []
        for index, batch in enumerate(gtaloader):
            if index % 100 == 0:
                print('%d processd of %d' % (index, len(gtaloader)))
            image, _,_, name = batch
            output1, output2 = model(Variable(image, volatile=True).cuda(gpu0))
            output = interp_down(output2).cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)

            output = output[np.newaxis, :]  # add a dim
            gta_feat_distrs.extend(output)
            gta_img_paths.extend(name)

        gta_feat_distrs_np = np.array(gta_feat_distrs)
        gta_img_paths_np = np.array(gta_img_paths)
        np.save(gta_feat_distr_path, gta_feat_distrs_np)
        np.save(gta_imgpaths_path, gta_img_paths_np)
    else:
        gta_feat_distrs_np = np.load(gta_feat_distr_path)
        gta_img_paths_np = np.load(gta_imgpaths_path)

    if os.path.isfile(closest_imgs_path) == False:
        temp_feat=citys_feat_distrs_np[0, :]
        # [m,n,c]=temp_feat.shape
        pixel_amount=temp_feat.size
        closest_imgs_locs=[]
        for i in range(citys_img_paths_np.shape[0]):
            cur_citys_feat= citys_feat_distrs_np[i, :]
            distances=[]
            if i %10==0:
                print(i)
            for j in range(gta_img_paths_np.shape[0]):
                cur_gta_feat=gta_feat_distrs_np[j, :]
                dist_abs = abs(cur_citys_feat - cur_gta_feat)
                # e_dist = np.sqrt(np.square(dist_abs).sum(axis=1))
                dist_mean=np.sum(dist_abs)/pixel_amount
                distances.append(dist_mean)
            min_loc=np.argsort(distances)
            # need to check overlap
            top_ord=3
            closest_imgs_loc=min_loc[:top_ord]
            intersect_imgs= np.intersect1d(closest_imgs_loc,closest_imgs_locs)
            while intersect_imgs.size:
                inters_num=len(intersect_imgs)
                closest_imgs_loc_confirm=np.setdiff1d(closest_imgs_loc,intersect_imgs)  # find the difference
                closest_imgs_loc_candi=min_loc[top_ord:top_ord+inters_num]
                top_ord=top_ord+inters_num
                closest_imgs_loc_confirm=np.concatenate([closest_imgs_loc_confirm,closest_imgs_loc_candi])
                closest_imgs_loc=closest_imgs_loc_confirm
                intersect_imgs = np.intersect1d(closest_imgs_loc, closest_imgs_locs)

            closest_imgs_locs.extend(closest_imgs_loc)
        np.save(closest_imgs_path, closest_imgs_locs)
    else:
        closest_imgs_locs=np.load(closest_imgs_path)
    closest_imgs_locs_uni=np.unique(closest_imgs_locs)
    zq=1

    # get file_names
    with open(src_train_imgs_txt,'w') as f_train:
        for img_num in closest_imgs_locs_uni:
            line=gta_img_paths_np[img_num]+'\n'
            f_train.write(line)

if __name__ == '__main__':
    main()

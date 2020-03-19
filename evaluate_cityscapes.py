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
from collections import OrderedDict
import os
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

MODEL_NAME = 'DeepLabV2'

if MODEL_NAME=='DeepLabV2':
    from model.deeplab_multi import Res_Deeplab
elif MODEL_NAME=='DeepLabV3':
    from model.deeplabv3 import Res_Deeplab
CITY_IMG_MEAN= np.array((73.15835921, 82.90891754,72.39239876), dtype=np.float32)
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
IMG_MEAN=CITY_IMG_MEAN

DATA_DIRECTORY = '/home/zq/dl-test/ZQAdaptSegNet-master/data/Cityscapes/data/'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
RESTORE_FROM='/mnt/sda/zq2/ZQAdaptSegNet/snapshots/adapt_slective_pt/adapt_iter4/retrain_mixed4_mix/GTA5_{0}.pth'
#output
SAVE_PATH = './result/adapt_slective_pt/adapt_iter2_data/prediction/steps{0}'
SAVE_PATH = './result/adapt_slective_pt/adapt_iter4_data/prediction_train/steps{0}'


ITER_START=8000
ITER_END=8000
SPAN = 1000


# if not os.path.exists(SAVE_PATH):
#     os.makedirs(SAVE_PATH)
# gta_mp=np.load('./some_code/gta_cls_closest_m_pixels_percent.npy')
# citys_mp=np.load('./some_code/citys_cls_m_pixels_percent.npy')
# mp_weights=gta_mp/citys_mp
# weighted_classes=[4,5,6,7,16]
# for i in weighted_classes:
#     mp_weights[i] = 1.32 * (2 / (1 + np.exp(-2 * mp_weights[i])) - 1)

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.

SET = 'val'
# SET = 'train'

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

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
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

    for iter in range(ITER_START,ITER_END+1,SPAN):

        print('{0} /{1} processed'.format(iter,ITER_END))

        if not os.path.exists(args.save.format(iter)):
            os.makedirs(args.save.format(iter))

        model = Res_Deeplab(num_classes=args.num_classes)


        saved_state_dict = torch.load(args.restore_from.format(iter))
        # for k, v in saved_state_dict.items():
        #     if k.count('num_batches_tracked'):
        #         del saved_state_dict[k]
        model.load_state_dict(saved_state_dict)

        model.eval()
        model.cuda(gpu0)

        testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                        batch_size=1, shuffle=False, pin_memory=True)

        interp = nn.Upsample(size=(1024, 2048), mode='bilinear',align_corners=True)

        for index, batch in enumerate(testloader):
            if index % 100 == 0:
                print('%d processd of %d' % (index,len(testloader)))
            image, _, name = batch
            output1, output2 = model(Variable(image, volatile=True).cuda(gpu0))
            output = interp(output2)
            # output = F.softmax(output)
            output=output.cpu().data[0].numpy()


            output = output.transpose(1,2,0)
            # # a_mean = np.mean(output[:, :, 0])
            # # print(a_mean)
            # # output[:, :, 5] = output[:, :, 5] * 1.3
            # # output[:, :, 6] = output[:, :, 6] * 1.3
            # # output[:, :, 7] = output[:, :, 7] * 2
            # for mp_i in weighted_classes:
            #     output[:, :, mp_i] = output[:, :, mp_i] * mp_weights[mp_i]

            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            output_col = colorize_mask(output)
            # output = Image.fromarray(output)

            name = name[0].split('/')[-1]
            # output.save('%s/%s' % (args.save, name))
            # output_col.save('%s/%s_color.png' % (args.save.format(iter), name.split('.')[0]))
            output_col.save('%s/%s.png' % (args.save.format(iter), name.split('.')[0]))


if __name__ == '__main__':
    main()

import torch
import argparse
import scipy
from scipy import ndimage
import numpy as np

from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab_multi import Res_Deeplab
from dataset.cityscapes_dataset import cityscapesDataSet
import os
from PIL import Image
from model.discriminator import FCDiscriminator
import matplotlib.pyplot as plt
import torch.nn as nn
# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


device_ids = [0]
IMG_MEAN= np.array((73.15835921, 82.90891754,72.39239876), dtype=np.float32)
adapt_iter=1
CITYS_RETRAIN_TXT='./result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_pt/confidence_values/retrain.txt'
CITYS_VALUES_SV_PATH='./result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_pt/confidence_values/out_values.npy'
CITYS_FINE_VALUES_SV_PATH='./result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_pt/confidence_values/fine_out_values.npy'

if not os.path.exists('./result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_pt'):
    os.makedirs('./result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_pt')
if not os.path.exists('./result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_pt/confidence_values'):
    os.makedirs('./result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_pt/confidence_values')

DATA_DIRECTORY = 'G:/data/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/train.txt'
SAVE_PATH = './result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_pt/cityscapes_train/'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM='./model/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM='./model/GTA5_262500.pth'#''../snapshots/retrain_mixed2/GTA5_55000.pth'
RESTORE_FROM='./model/GTA5_8000.pth'
# DISC_RESTORE_FROM="/home/zq/dl-test/ZQAdaptSegNet-master/snapshots/GTA2Cityscapes_multi/GTA5_20000_D2.pth"
DISC_RESTORE_FROM="./model/GTA5_262500_D2.pth"
SET = 'train'
# SET = 'val'

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
    parser.add_argument("--dis-restore-from",type=str,default=DISC_RESTORE_FROM,
                        help="Where discriminator restore model parameters from.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    gpu0 = args.gpu

    model = Res_Deeplab(num_classes=args.num_classes)

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    #===========load discriminator model====begin===
    model_d2=FCDiscriminator(num_classes=args.num_classes)
    d2_state_dict=torch.load(args.dis_restore_from)
    model_d2.load_state_dict(d2_state_dict)
    model_d2.eval()
    model_d2.cuda(gpu0)

    #===========load discriminator model====end===
    testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear',align_corners=True)
    # interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear')

    out_values=[]
    fine_out_values=[]
    retrain_list=[]
    file = open(CITYS_RETRAIN_TXT, 'w')
    for index, batch in enumerate(testloader):
        if index % 20 == 0:
            print('%d processd of %d' % (index,len(testloader)))
        image, _, name = batch
        output1, output2 = model(Variable(image, volatile=True).cuda(gpu0))
        ini_output = interp(output2)
        d2_out1 = model_d2(F.softmax(ini_output,dim=1)) #.cpu().data[0].numpy()
        out_valu=d2_out1.mean()
        out_valu_img=np.array([[name[0]],out_valu.cpu().data.numpy()])

        out_values.extend(out_valu_img)
        if out_valu.cpu().data.numpy()>0.64:
            fine_out_valu_img = np.array([[name[0]], out_valu.cpu().data.numpy()])
            fine_out_values.extend(fine_out_valu_img)

            file.write(name[0]+'\n')
            output = interp(output2).cpu().data[0].numpy()

            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            output_col = colorize_mask(output)
            name = name[0].split('/')[-1]
            # output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))
            output_col.save('%s/%s.png' % (args.save, name.split('.')[0]))
            # print('its confidence value is %f' % out_valu)

            # plt.imshow(output_col)
            # plt.title(str(out_valu))
            # plt.show()

            # output = Image.fromarray(output)
            # output.save('%s/%s' % (args.save, name))


    out_values = np.array(out_values)

    np.save(CITYS_VALUES_SV_PATH, out_values)
    np.save(CITYS_FINE_VALUES_SV_PATH, fine_out_values)

    file.close()


if __name__ == '__main__':
    main()

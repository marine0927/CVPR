import torch
import argparse
import scipy
from scipy import ndimage
import numpy as np
import shutil

import sys
sys.path.append('../')

from torch.autograd import Variable
import torch.nn.functional as F
from model.deeplab_multi import Res_Deeplab
from dataset.cityscapes_dataset import cityscapesDataSet
import os
from PIL import Image
from model.discriminator import FCDiscriminator
import matplotlib.pyplot as plt
import torch.nn as nn

from torch.utils import data, model_zoo
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os.path as osp
from utils.loss import CrossEntropy2d
from dataset.mixed_dataset import MixedDataSet
from synthia_show_val import show_val

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

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

slic_str = ''  # empty means no slic
j_val = 0.4
top_r=150
start_iter=2
# parameters initialize
# ======parameters on calculation of outputs========begin=====
RESTORE_FROM='../model/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM='../model/syn_40000.pth'
RESTORE_FROM='../snapshots/syn_ss/40000_D40000_j0.4_t150/da_iter1_j0.4/da_iter1_steps6000.pth'
CITYS_DATA_DIRECTORY ='/home/zq/dl-test/ZQAdaptSegNet-master/data/Cityscapes/data/'
CITYS_DATA_LIST_PATH = '../dataset/cityscapes_list/train.txt'

DA_DIR='../result/syn_ss/40000_D40000_j0.4_t150/da_iter{0}_j{1}{2}'
SNAPSHOT_DIR = '../snapshots/syn_ss/40000_D40000_j0.4_t150/da_iter{0}_j{1}{2}'

CITYS_RETRAIN_TXT=DA_DIR+'/confidence_values/retrain.txt'
CITYS_VALUES_SV_PATH=DA_DIR+'/confidence_values/out_values.npy'
CITYS_FINE_VALUES_SV_PATH=DA_DIR+'/confidence_values/fine_out_values.npy'

CITYS_RETRAIN_SAVE_DIR = DA_DIR + '/cityscapes_train/'
# ======parameters on calculation of outputs========end=====

# ======parameters on generate retaining data========begin=====
# id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
#                               19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
#                               26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

GTA_GT_DIR = 'G:/Dataset2/SYNTHIA__RAND_CITYSCAPES/GT/LABELS_cityscapeID/'
GTA_IMG_DIR = 'G:/Dataset2/SYNTHIA__RAND_CITYSCAPES/RGB/'
CITYS_LBL_DIR = CITYS_RETRAIN_SAVE_DIR
CITYS_IMG_DIR = '/home/zq/dl-test/ZQAdaptSegNet-master/data/Cityscapes/data//leftImg8bit/train/'

GTA_TRAIN_CLOSEST_LIST_PATH= '../dataset/gta5_list/null.txt'
# citys_retrain_list_path=CITYS_RETRAIN_TXT

# output path
MIXED_IMG_DIR= DA_DIR + '/Mixeddata/images/'
MIXED_LBL_DIR= DA_DIR + '/Mixeddata/labels/'
TRAIN_MIXED_SV_PATH= DA_DIR + '/train_mixed.txt'

city_train_class_name={'aachen','bochum','bremen','cologne','darmstadt','dusseldorf','erfurt','hamburg','hanover','jena','krefeld',
            'monchengladbach','strasbourg','stuttgart','tubingen','ulm','weimar','zurich'}
# ======parameters on generate retaining data========end=====

#===========parameters on  retraining==========begin================
MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
RETRAIN_DIR = DA_DIR + '/Mixeddata'
IGNORE_LABEL = 255
INPUT_SIZE = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 6000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000

WEIGHT_DECAY = 0.0005
LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
#=============parameters on  retraining========end================

IGNORE_LABEL = 255
NUM_CLASSES = 19

# DIS_RESTORE_FROM='../keshan/GTA5_45000_D2.pth'
DIS_RESTORE_FROM='../model/syn_40000_D2.pth'
SET = 'train'
# SET = 'val'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)

for i in range(zero_pad):
    palette.append(0)

# def assign_params(iter,val,slic_flg):
#     a=DA_DIR.format(da_iter,j_val,slic_flg)
#     zq=1

def make_dirs(da_iter,j_val,slic_str):
    if not os.path.exists(DA_DIR.format(da_iter,j_val,slic_str)):
        os.makedirs(DA_DIR.format(da_iter,j_val,slic_str))
    if not os.path.exists(DA_DIR.format(da_iter,j_val,slic_str) + '/confidence_values'):
        os.makedirs(DA_DIR.format(da_iter,j_val,slic_str) + '/confidence_values')
    if not os.path.exists(MIXED_IMG_DIR.format(da_iter,j_val,slic_str)):
        os.makedirs(MIXED_IMG_DIR.format(da_iter,j_val,slic_str))
    if not os.path.exists(MIXED_LBL_DIR.format(da_iter,j_val,slic_str)):
        os.makedirs(MIXED_LBL_DIR.format(da_iter,j_val,slic_str))

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--data-dir", type=str, default=CITYS_DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=CITYS_DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=1,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=CITYS_RETRAIN_SAVE_DIR,
                        help="Path to save result.")
    parser.add_argument("--dis-restore-from",type=str,default=DIS_RESTORE_FROM,
                        help="Where discriminator restore model parameters from.")


    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    # parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
    #                     help="Path to the directory containing the source dataset.")
    # parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
    #                     help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def class_statistics(input_data, cls_num):
    data_stat = np.zeros(cls_num)
    for j in range(cls_num):
        cur_stat = np.sum(input_data == j)
        data_stat[j] = cur_stat
    return data_stat



def cal_output_value(pred_values,updated_model,citys_retrain_txt,citys_values_sv_path,citys_fine_values_sv_path,citys_retrain_save_path):


    if not os.path.exists(citys_retrain_save_path):
        os.makedirs(citys_retrain_save_path)

    gpu0 = args.gpu
    model = Res_Deeplab(num_classes=args.num_classes)
    saved_state_dict = torch.load(updated_model)
    # model.load_state_dict(saved_state_dict)
    saved_state_dict1=saved_state_dict

#    for k, v in saved_state_dict.items():
#        if k.count('num_batches_tracked'):
#            del saved_state_dict1[k]
    # saved_state_dict1={k.replace('bn1.num_batches_tracked', ''): v for k, v in saved_state_dict.items()}


    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(gpu0)

    # ===========load discriminator model====begin===
    model_d2 = FCDiscriminator(num_classes=args.num_classes)
    d2_state_dict = torch.load(args.dis_restore_from)
    model_d2.load_state_dict(d2_state_dict)
    model_d2.eval()
    model_d2.cuda(gpu0)
    # ===========load discriminator model====end===

    # ===========load class discriminator model====begin==



    # ===========load class discriminator model=====end===
    testloader = data.DataLoader(
        cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False,
                          mirror=False, set=args.set),
        batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear',align_corners=True)
    # interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear')

    out_values = []
    fine_out_values = []
    retrain_list = []
    file = open(citys_retrain_txt, 'w')
    for index, batch in enumerate(testloader):
        # if index > 30:
        #     break
        if index % 20 == 0:
            print('%d processd of %d' % (index, len(testloader)))
        image, _, name = batch
        output1, output2 = model(Variable(image, volatile=True).cuda(gpu0))
        ini_output = interp(output2)
        d2_out1 = model_d2(F.softmax(ini_output,dim=1))
        out_valu = d2_out1.mean()
        out_valu_img = np.array([name[0], out_valu.cpu().data.numpy()])
        # if out_valu.cpu().data.numpy()[0]<0:
        #     zq=1
        #     print(index)

        out_values.append(out_valu_img)
        # if out_valu.cpu().data.numpy()[0] < j_val:
        #     fine_out_valu_img = np.array([[name[0]], out_valu.cpu().data.numpy()])
        #     fine_out_values.extend(fine_out_valu_img)
        #     file.write(name[0] + '\n')
        output = interp(output2).cpu().data[0].numpy()

        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        name = name[0].split('/')[-1]
        output_col.save('%s/%s_color.png' % (citys_retrain_save_path, name.split('.')[0]))  # the word 'color' maybe need to delete

    out_values = np.array(out_values)
    np.save(citys_values_sv_path, out_values)

    # top 1000 or smaller than j_val
    values=out_values[:, 1]
    values=np.array(values).astype(float)
    out_values_sort_indx=np.argsort(values)
    for sort_i in range(top_r):
        val_ord=out_values_sort_indx[sort_i]
        if float(out_values[val_ord,1])< j_val:
            fine_out_valu_img = np.array([out_values[val_ord,0], out_values[val_ord,1]])
            fine_out_values.append(fine_out_valu_img)

            if float(out_values[val_ord,1])>pred_values[val_ord,0]:
                pred_values[val_ord,1]=0
                file.write(out_values[val_ord, 0] + ' 0\n')
            else:
                pred_values[val_ord, 1] = 1
                file.write(out_values[val_ord, 0] + '\n')
    np.save(citys_fine_values_sv_path, fine_out_values)
    pred_values[:,0]=values
    file.close()
    return pred_values


def find_closest_source():
    # useless here , probably be useful to transfer
    return 0

def gener_retrain_data(citys_fine_values_sv_path,citys_retrain_save_previous_dir,citys_retrain_save_dir,citys_retrain_txt,mixed_img_dir,mixed_lbl_dir,train_mixed_sv_path):
    citys_file = open(citys_retrain_txt, 'r')
    gta5_file = open(GTA_TRAIN_CLOSEST_LIST_PATH, 'r')
    mixed_file = open(train_mixed_sv_path, 'w')

    citys_fine_values=np.load(citys_fine_values_sv_path)
    line_num=0
    for line in citys_file.readlines():
        pre_flg=line.split(' ')[-1]
        if pre_flg== '0\n':
            citys_lbl_dir=citys_retrain_save_previous_dir
        else:
            citys_lbl_dir = citys_retrain_save_dir
        # line=line[0:len(line) - 1].split(' ')[0]
        line = line.strip().split(' ')[0]
        s_img_file_path = CITYS_IMG_DIR + line
        t_img_file_path = mixed_img_dir + line.split('/')[1]
        s_lbl_file_path = citys_lbl_dir + line.split('/')[1].split('.')[0] + '_color.png'
        t_lbl_file_path = mixed_lbl_dir + line.split('/')[1].split('.')[0] + '.png'
        shutil.copyfile(s_img_file_path, t_img_file_path)
        shutil.copyfile(s_lbl_file_path, t_lbl_file_path)
        mixed_file.write(line.split('/')[1]+' '+citys_fine_values[line_num,1]+'\n')
        # mixed_file.write(line.split('/')[1] + ' ' + '0\n')# set as 0 for weight 1
        line_num+=1

    for line in gta5_file.readlines():
        s_img_file_path = CITYS_IMG_DIR + line[0:len(line) - 1]
        t_img_file_path = mixed_img_dir + line[0:len(line) - 1]
        shutil.copyfile(s_img_file_path, t_img_file_path)
        t_lbl_file_path = mixed_lbl_dir + line[0:len(line) - 1]
        lbl_path = GTA_GT_DIR + line[0:len(line) - 1]
        lbl = np.array(Image.open(lbl_path))
        label_copy = 255 * np.ones(lbl.shape, dtype=np.float32)
        # for k, v in id_to_trainid.items():
        #     label_copy[lbl == k] = v
        lbl = label_copy
        lbl_col = colorize_mask(lbl)
        lbl_col.save(t_lbl_file_path)

        mixed_file.write(line)
    citys_file.close()
    gta5_file.close()
    mixed_file.close()

def retrain_citys(da_iter,updated_model,train_mixed_sv_path,retrain_dir,snapshot_dir,pred_sv_dir):
    """Create the model and start the training."""

    strat_snap_iter=0 #int(args.restore_from.split('_')[-1].split('.')[0])
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    gpu = args.gpu

    # Create network
    if args.model == 'DeepLab':
        model = Res_Deeplab(num_classes=args.num_classes)
        saved_state_dict = torch.load(updated_model)
#        for k, v in saved_state_dict.items():
#            if k.count('num_batches_tracked'):
#                del saved_state_dict[k]
        model.load_state_dict(saved_state_dict)

    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    # init D

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    retrain_list=train_mixed_sv_path

    trainloader = data.DataLoader(
        MixedDataSet(retrain_dir, retrain_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                     crop_size=input_size,
                     scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        # batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # bce_loss = torch.nn.BCEWithLogitsLoss()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)

    loss2_sum_per_epoch = 0
    loss2_per_epoch = 0
    epoch = 0
    loss2_epoch = ''
    lbl_list = open(args.data_list, 'r')
    lbl_num = len(lbl_list.readlines()) / 2
    mIoUs=[]
    syn_mIoUs=[]
    for i_iter in range(args.num_steps):

        loss_seg_value1 = 0

        loss_seg_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        for sub_i in range(args.iter_size):
            # train G

            # train with source

            _, batch = trainloader_iter.__next__()
            images, labels, _, img_nam, value = batch
            # print('%s\n' % img_nam[0])
            images = Variable(images).cuda(args.gpu)

            pred1, pred2 = model(images)
            pred1 = interp(pred1)
            pred2 = interp(pred2)

            loss_seg1 = loss_calc(pred1, labels, args.gpu)
            loss_seg2 = loss_calc(pred2, labels, args.gpu)
            loss = loss_seg2 + args.lambda_seg * loss_seg1

            # proper normalization
            loss_weight=1-float(value[0])
            if loss_weight>1:
                loss_weight=1
            loss = loss_weight*loss / args.iter_size
            loss.backward()
            loss_seg_value1 += loss_seg1.data.cpu().numpy() / args.iter_size
            loss_seg_value2 += loss_seg2.data.cpu().numpy() / args.iter_size

        optimizer.step()

        # print('exp = {}'.format(args.snapshot_dir))
        print(
            'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f}'.format(
                i_iter+ strat_snap_iter, args.num_steps, loss_seg_value1, loss_seg_value2))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(snapshot_dir, 'da_iter' +str(da_iter)+'_steps' + str(args.num_steps) + '.pth'))
            show_pred_sv_dir = pred_sv_dir + '/steps' + str(args.num_steps)
            mIoU, syn_mIoU= show_val(model.state_dict(), show_pred_sv_dir)
            mIoUs.append(str(round(np.nanmean(mIoU) * 100, 2)))
            syn_mIoUs.append(str(round(np.nanmean(syn_mIoU) * 100, 2)))
            for i_iou in range(len(mIoUs)):
                print(mIoUs[i_iou]+' '+syn_mIoUs[i_iou])
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(),
                       osp.join(snapshot_dir, 'da_iter' +str(da_iter)+'_steps'+ str(i_iter + strat_snap_iter) + '.pth'))
            show_pred_sv_dir =pred_sv_dir+'/steps'+str(i_iter + strat_snap_iter)
            mIoU,syn_mIoU = show_val(model.state_dict(), show_pred_sv_dir)
            mIoUs.append(str(round(np.nanmean(mIoU) * 100, 2)))
            syn_mIoUs.append(str(round(np.nanmean(syn_mIoU) * 100, 2)))
            for i_iou in range(len(mIoUs)):
                print(mIoUs[i_iou]+' '+syn_mIoUs[i_iou])

        loss2_sum_per_epoch += loss_seg2.data.cpu().numpy()
        if i_iter % lbl_num == 0 and i_iter != 0:
            epoch += 1
            loss2_per_epoch = loss2_sum_per_epoch / lbl_num
            loss2_epoch += 'epoch = {0}, loss_seg2 = {1:.3f} \n'.format(epoch, loss2_per_epoch)
            print(loss2_epoch)
            loss2_sum_per_epoch = 0
    return osp.join(snapshot_dir, 'da_iter' +str(da_iter)+'_steps'+ str(args.num_steps) + '.pth')

def main():
    """Create the model and start the evaluation process."""
    updated_model=RESTORE_FROM
    pred_values=np.ones((2975,2))
    for da_iter in range(start_iter,9):
        # parameters of calculating output value
        citys_retrain_txt=CITYS_RETRAIN_TXT.format(da_iter,j_val,slic_str)
        citys_values_sv_path=CITYS_VALUES_SV_PATH.format(da_iter,j_val,slic_str)
        citys_fine_values_sv_path=CITYS_FINE_VALUES_SV_PATH.format(da_iter,j_val,slic_str)
        citys_retrain_save_dir=CITYS_RETRAIN_SAVE_DIR.format(da_iter, j_val, slic_str)
        citys_retrain_save_previous_dir=CITYS_RETRAIN_SAVE_DIR.format(da_iter-1, j_val, slic_str)

        # parameters of generating retrain data
        mixed_img_dir=MIXED_IMG_DIR.format(da_iter,j_val,slic_str)
        mixed_lbl_dir=MIXED_LBL_DIR.format(da_iter,j_val,slic_str)
        train_mixed_sv_path=TRAIN_MIXED_SV_PATH.format(da_iter,j_val,slic_str)

        # parameters of retraining
        retrain_dir=RETRAIN_DIR.format(da_iter,j_val,slic_str)
        snapshot_dir=SNAPSHOT_DIR.format(da_iter,j_val,slic_str)
        pred_sv_dir=DA_DIR.format(da_iter, j_val, slic_str)

        make_dirs(da_iter,j_val,slic_str)

        pred_values=cal_output_value(pred_values,updated_model, citys_retrain_txt, citys_values_sv_path, citys_fine_values_sv_path, citys_retrain_save_dir)
        gener_retrain_data(citys_fine_values_sv_path,citys_retrain_save_previous_dir,citys_retrain_save_dir,citys_retrain_txt,mixed_img_dir,mixed_lbl_dir,train_mixed_sv_path)
        updated_model=retrain_citys(da_iter,updated_model, train_mixed_sv_path, retrain_dir, snapshot_dir, pred_sv_dir)

if __name__ == '__main__':
    main()

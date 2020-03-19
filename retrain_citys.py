import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random

from model.deeplab_multi import Res_Deeplab
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.mixed_dataset import MixedDataSet
from dataset.cityscapes_dataset import cityscapesDataSet

from show_val import show_val

# IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN= np.array((73.15835921, 82.90891754,72.39239876), dtype=np.float32)

strat_snap_iter = 0
adapt_iter = 6

MODEL = 'DeepLab'
BATCH_SIZE = 2
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_mixdata'
DATA_LIST_PATH = './result/adapt_iter' + str(adapt_iter) + '/train_mixed.txt'
DATA_LIST_PATH = './result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_mixdata/train_mixed.txt'
IGNORE_LABEL = 255
# INPUT_SIZE = '1280,720'
INPUT_SIZE = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 250000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
RESTORE_FROM = '/home/guest/ZQAdaptSegNet-master/model/GTA5_3000.pth' # from last iteration
# RESTORE_FROM='/mnt/sda/zq2/ZQAdaptSegNet/snapshots/gta2citys_pt/GTA5_262500.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
pre_sv_dir='./result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_mixdata/prediction/steps{0}'
SNAPSHOT_DIR = '/home/guest/ZQAdaptSegNet-master/snapshots/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'/retrain_mixed' + str(adapt_iter)+'_mix'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

SET = 'train'

device_ids = [0, 1, 2]

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
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
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
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
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
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
    optimizer.module.param_groups[0]['lr'] = lr
    if len(optimizer.module.param_groups) > 1:
        optimizer.module.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    gpu = args.gpu

    # Create network
    if args.model == 'DeepLab':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http':
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.load_state_dict(saved_state_dict)
        # ====the lines below comment by ZQ====
        # new_params = model.state_dict().copy()
        # for i in saved_state_dict:
        #     # Scale.layer5.conv2d_list.3.weight
        #     i_parts = i.split('.')
        #     # print i_parts
        #     if not args.num_classes == 19 or not i_parts[1] == 'layer5':
        #         new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        #         # print i_parts
        # model.load_state_dict(new_params)
        # ====end====

    model.train()
    
    model.cuda(args.gpu)

    cudnn.benchmark = True

    # init D

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        MixedDataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                     crop_size=input_size,
                     scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    # targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
    #                                                  max_iters=args.num_steps * args.iter_size * args.batch_size,
    #                                                  crop_size=input_size_target,
    #                                                  scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
    #                                                  set=args.set),
    #                                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    #                                pin_memory=True)
    #
    #
    # targetloader_iter = enumerate(targetloader)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
    optimizer.module.zero_grad()

    # optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    # optimizer_D1.zero_grad()
    #
    # optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    # optimizer_D2.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear',align_corners=True)
    # interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear')

    # labels for adversarial training
    # source_label = 0
    # target_label = 1

    loss2_sum_per_epoch = 0
    loss2_per_epoch = 0
    epoch = 0
    loss2_epoch = ''
    lbl_list = open(args.data_list, 'r')
    lbl_num = len(lbl_list.readlines()) /2
    # lbl_list = os.listdir(args.data_dir+'/labels')
    # lbl_num = len(lbl_list)/2
    mIoUs = []
    i_iters = []

    for i_iter in range(args.num_steps):

        loss_seg_value1 = 0
        # loss_adv_target_value1 = 0
        # loss_D_value1 = 0

        loss_seg_value2 = 0
        # loss_adv_target_value2 = 0
        # loss_D_value2 = 0

        optimizer.module.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        for sub_i in range(args.iter_size):
            # train G

            # train with source

            _, batch = trainloader_iter.__next__()
            images, labels, _, img_nam = batch
            # print('%s\n' % img_nam[0])
            images = Variable(images).cuda(args.gpu)

            pred1, pred2 = model(images)
            pred1 = interp(pred1)
            pred2 = interp(pred2)

            loss_seg1 = loss_calc(pred1, labels, args.gpu)
            loss_seg2 = loss_calc(pred2, labels, args.gpu)
            loss = loss_seg2 + args.lambda_seg * loss_seg1

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value1 += loss_seg1.data.cpu().numpy()/ args.iter_size
            loss_seg_value2 += loss_seg2.data.cpu().numpy()/ args.iter_size

            # train with target
        optimizer.module.step()

        # print('exp = {}'.format(args.snapshot_dir))
        print(
            'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value1, loss_seg_value2))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(),
                       osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter + strat_snap_iter) + '.pth'))
            show_pred_sv_dir = pre_sv_dir.format(i_iter)
            mIoU = show_val(model.state_dict(), show_pred_sv_dir,gpu)
            mIoUs.append(str(round(np.nanmean(mIoU) * 100, 2)))
            i_iters.append(i_iter)
            print_i = 0
            for miou in mIoUs:
                print('i{0}: {1}'.format(i_iters[print_i], miou))
                print_i = print_i + 1

        loss2_sum_per_epoch += loss_seg2.data.cpu().numpy()
        if i_iter % lbl_num == 0 and i_iter != 0:
            epoch +=1
            loss2_per_epoch = loss2_sum_per_epoch / lbl_num
            loss2_epoch+='epoch = {0}, loss_seg2 = {1:.3f} \n'.format(epoch, loss2_per_epoch)
            print(loss2_epoch)
            loss2_sum_per_epoch=0


if __name__ == '__main__':
    main()

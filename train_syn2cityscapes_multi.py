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
from PIL import Image
import matplotlib.pyplot as plt
import random
align_corners=True
from model.deeplab_multi import Res_Deeplab
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.synthia_dataset import synthiaDataSet
from dataset.cityscapes_dataset import cityscapesDataSet

from show_val import show_val

# 112.89155291
# 111.798947848
# 108.391345058
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
# IMG_MEAN = np.array((80.3504592789,70.8139310298,63.3137090403), dtype=np.float32) #from mine
CITY_IMG_MEAN = np.array((73.15835921, 82.90891754,72.39239876), dtype=np.float32)

iter_start=0

MODEL = 'DeepLab'
BATCH_SIZE = 2
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/home/zq/dl-test/ZQAdaptSegNet-master/data/SYNTHIA__RAND_CITYSCAPES/'
DATA_LIST_PATH = './dataset/synthia_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1024,512'#'1280,720'#
DATA_DIRECTORY_TARGET = '/home/zq/dl-test/ZQAdaptSegNet-master/data/Cityscapes/data'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 300001
NUM_STEPS_STOP = 300001  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
# RESTORE_FROM ='/mnt/sda/zq2/ZQAdaptSegNet/snapshots/gta2citys_pt/GTA5_187500.pth'
# D1_RESTORE_FROM = '/mnt/sda/zq2/ZQAdaptSegNet/snapshots/gta2citys_pt/GTA5_187500_D1.pth'
# D2_RESTORE_FROM = '/mnt/sda/zq2/ZQAdaptSegNet/snapshots/gta2citys_pt/GTA5_187500_D2.pth'
# OPTI_RESTORE_FROM ='/mnt/sda/zq2/ZQAdaptSegNet/snapshots/gta2citys_pt/GTA5_187500_optimizer.pth'
# OPTI_D1_RESTORE_FROM = '/mnt/sda/zq2/ZQAdaptSegNet/snapshots/gta2citys_pt/GTA5_187500_optimizer_D1.pth'
# OPTI_D2_RESTORE_FROM = '/mnt/sda/zq2/ZQAdaptSegNet/snapshots/gta2citys_pt/GTA5_187500_optimizer_D2.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = '/mnt/sda/zq2/ZQAdaptSegNet/snapshots/syn2citys_pt_temp2/'
pre_sv_dir='./result/syn2citys_pt_temp2/steps{0}'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0001
LAMBDA_ADV_TARGET2 = 0.001

TARGET = 'cityscapes'
SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
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
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
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
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)

    cudnn.enabled = True
    gpu = args.gpu

    # Create network
    if args.model == 'DeepLab':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params)
        # model.load_state_dict(saved_state_dict)

    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D2 = FCDiscriminator(num_classes=args.num_classes)

    # # load discriminator params
    # saved_state_dict_D1 = torch.load(D1_RESTORE_FROM)
    # saved_state_dict_D2 = torch.load(D2_RESTORE_FROM)
    # model_D1.load_state_dict(saved_state_dict_D1)
    # model_D2.load_state_dict(saved_state_dict_D2)

    model_D1.train()
    model_D1.cuda(args.gpu)

    model_D2.train()
    model_D2.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        synthiaDataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=CITY_IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)


    targetloader_iter = enumerate(targetloader)

    # implement model.optim_parameters(args) to handle different models' lr setting


    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.7, 0.99))
    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.7, 0.99))

    #BYZQ
    # opti_state_dict = torch.load(OPTI_RESTORE_FROM)
    # opti_state_dict_d1 = torch.load(OPTI_D1_RESTORE_FROM)
    # opti_state_dict_d2 = torch.load(OPTI_D2_RESTORE_FROM)
    # optimizer.load_state_dict(opti_state_dict)
    # optimizer_D1.load_state_dict(opti_state_dict_d1)
    # optimizer_D1.load_state_dict(opti_state_dict_d2)

    optimizer.zero_grad()
    optimizer_D1.zero_grad()
    optimizer_D2.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear',align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',align_corners=True)

    # labels for adversarial training
    source_label = 1
    target_label = 0
    mIoUs = []
    i_iters = []

    for i_iter in range(args.num_steps):
        if i_iter<=iter_start:
            continue

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        adjust_learning_rate_D(optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False

            # train with source

            _, batch = trainloader_iter.__next__()
            images, labels, _, _ = batch
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
            loss_seg_value1 += loss_seg1.data.cpu().numpy() / args.iter_size
            loss_seg_value2 += loss_seg2.data.cpu().numpy() / args.iter_size

            # train with target

            _, batch = targetloader_iter.__next__()
            images, _, name = batch
            images = Variable(images).cuda(args.gpu)

            pred_target1, pred_target2 = model(images)
            pred_target1 = interp_target(pred_target1)
            pred_target2 = interp_target(pred_target2)

            D_out1 = model_D1(F.softmax(pred_target1,dim=1))
            D_out2 = model_D2(F.softmax(pred_target2,dim=1))




            loss_adv_target1 = bce_loss(D_out1,
                                       Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(
                                           args.gpu))

            loss_adv_target2 = bce_loss(D_out2,
                                        Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda(
                                            args.gpu))

            loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2
            loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy() / args.iter_size
            loss_adv_target_value2 += loss_adv_target2.data.cpu().numpy() / args.iter_size

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred1 = pred1.detach()
            pred2 = pred2.detach()

            D_out1 = model_D1(F.softmax(pred1,dim=1))
            D_out2 = model_D2(F.softmax(pred2,dim=1))

            weight_s = float(D_out2.mean().data.cpu().numpy())

            loss_D1 = bce_loss(D_out1,
                              Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(args.gpu))

            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda(args.gpu))

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()
            loss_D_value2 += loss_D2.data.cpu().numpy()

            # train with target
            pred_target1 = pred_target1.detach()
            pred_target2 = pred_target2.detach()

            D_out1 = model_D1(F.softmax(pred_target1,dim=1))
            D_out2 = model_D2(F.softmax(pred_target2,dim=1))

            weight_t = float(D_out2.mean().data.cpu().numpy())
            # if weight_b>0.5 and i_iter>500:
            #     confidence_map = interp(D_out2).cpu().data[0][0].numpy()
            #     name = name[0].split('/')[-1]
            #     confidence_map=255*confidence_map
            #     confidence_output=Image.fromarray(confidence_map.astype(np.uint8))
            #     confidence_output.save('./result/confid_map/%s.png' % (name.split('.')[0]))
            #     zq=1
            print(weight_s,weight_t)

            loss_D1 = bce_loss(D_out1,
                              Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda(args.gpu))

            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda(args.gpu))

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()
            loss_D_value2 += loss_D2.data.cpu().numpy()

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()

        # print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f}'.format(
            i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2))

        if i_iter >= args.num_steps_stop - 1:
            print ('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '_D2.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print ('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D2.pth'))
            torch.save(optimizer.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_optimizer.pth'))
            torch.save(optimizer_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_optimizer_D1.pth'))
            torch.save(optimizer_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_optimizer_D2.pth'))
            show_pred_sv_dir = pre_sv_dir.format(i_iter)
            mIoU= show_val(model.state_dict(), show_pred_sv_dir,gpu)
            mIoUs.append(str(round(np.nanmean(mIoU) * 100, 2)))
            i_iters.append(i_iter)
            print_i = 0
            for miou in mIoUs:
                print('i{0}: {1}'.format(i_iters[print_i], miou))
                print_i = print_i + 1


if __name__ == '__main__':
    main()

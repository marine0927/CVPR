import numpy as np
import argparse
import json
from PIL import Image
from os.path import join

#NTHU
gtDir='./data/NTHU_Datasets/Tokyo/Labels/Test'
# predDir='./result/retrain_NTHU_rome/steps210'
predDir='./result/NTHU_tokyo_retrain_steps400_postcrf'
devkitDir='./dataset/NTHU_list'

ITER_START=1000
ITER_END=1000
SPAN = 1000
IMG_IOU_SHOW=False
CLS_IOU_SHOW=True

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):
    """
    Compute IoU given the predicted colorized images and
    """
    for iter in range(ITER_START, ITER_END + 1, SPAN):
        with open(join(devkit_dir, 'info.json'), 'r') as fp:
          info = json.load(fp)
        num_classes = np.int(info['classes'])
        if CLS_IOU_SHOW:
            print('Num classes', num_classes)
        name_classes = np.array(info['label'], dtype=np.str)
        mapping = np.array(info['label2train'], dtype=np.int)
        hist = np.zeros((num_classes, num_classes))

        image_path_list = join(devkit_dir, 'TOKYO_test.txt')  # city 5
        label_path_list = join(devkit_dir, 'TOKYO_test.txt')  # city 6
        gt_imgs = open(label_path_list, 'r').read().splitlines()
        gt_imgs = [join(gt_dir, x[:-4] + '_city.png') for x in gt_imgs]
        pred_imgs = open(image_path_list, 'r').read().splitlines()
        pred_imgs = [join(pred_dir.format(iter), x[:-3] + 'png') for x in pred_imgs]

        img_ious=[]

        for ind in range(len(gt_imgs)):
            # pred = np.array(Image.open(pred_imgs[ind][:-4]+'.png'))
            pred =Image.open(pred_imgs[ind])#[:-4] + '_color.png'))
            pred = pred.resize((2048, 1024), Image.NEAREST)
            pred= np.array(pred)
            label =Image.open(gt_imgs[ind])
            label=label.resize((2048,1024),Image.NEAREST)
            label = np.array(label)
            if len(label.flatten()) != len(pred.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
                continue
            hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
            #========calculate iou per image====begin==============
            per_cls_iou=per_class_iu(hist)
            per_cls_iou[np.isnan(per_cls_iou)]=0
            img_iou=100 * np.mean(per_cls_iou)
            img_iou_ords=[ind,img_iou]
            img_ious.append(img_iou_ords)
            # ========calculate iou per image====end==============
            if ind > 0 and ind % 10 == 0:
                if IMG_IOU_SHOW:
                    print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))

        mIoUs = per_class_iu(hist)
        np.save('./result/img_ious.npy',img_ious)
        for ind_class in range(num_classes):
            if CLS_IOU_SHOW:
                print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
        syn_mIoUs=np.delete(mIoUs,[3,4,5,9,14,16])
        print('===> iter: {0},mIoU: '.format(iter) + str(round(np.nanmean(mIoUs) * 100, 2))+' '+str(round(np.nanmean(syn_mIoUs) * 100, 2)))
    return mIoUs


def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str,default=gtDir, help='directory which stores CityScapes val gt images')
    parser.add_argument('--pred_dir', type=str, default=predDir, help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default=devkitDir, help='base directory of cityscapes')
    args = parser.parse_args()
    main(args)

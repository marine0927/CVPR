#coding=utf-8
import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image


class MixedDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(521, 521), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))#将图片list复制多段
        self.files = []
        # 元素替换，不是从某个数到某个数！！例：将7这个标签替换为0，下类同。冒号前为键key，冒号后为值value
        # self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
        #                       6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
        #                       13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "Images/Test/%s" % name.split(' ')[0])
            label_file = osp.join(self.root, "Labels/Test/%s_city.png" % (name.split(' ')[0]).split('.')[0])
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name.split(' ')[0],
                # "value": name.split(' ')[1] #byzq
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]
        # value = datafiles["value"] #byzq

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # # re-assign labels to match the format of Cityscapes
        # label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        # for k, v in self.id_to_trainid.items():
        #     label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name#,value byzq


if __name__ == '__main__':
    dst = MixedDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()

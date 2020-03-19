import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import os
import scipy.stats

def class_statistics(input_data, cls_num):
    data_stat = np.zeros(cls_num)
    for j in range(cls_num):
        cur_stat = np.sum(input_data == j)
        data_stat[j] = cur_stat
    return data_stat

#output
adp_dir='../result/adapt_slective_pt/adapt_iter5_pt/'
citys_cls_distr_path=adp_dir+'citys_cls_distr.npy'  #cityscapes
citys_imgpaths_path=adp_dir+'citys_img_paths.npy'
gta5_cls_distr_path=adp_dir+'gta5_cls_distr.npy'
gta5_imgpaths_path=adp_dir+'gta5_img_paths.npy'
citys_dir=adp_dir+'cityscapes_train/'
gta5_dir = '/home/zq/dl-test/ZQAdaptSegNet-master/data/GTA5/labels/'

# gta5_train_closest_sv_path= '../dataset/gta5_list/gta5_train_closest.txt'
gta5_train_closest_sv_path= adp_dir+'gta5_train_closest.txt'
citys_train_list='../dataset/cityscapes_list/train.txt'

id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

if os.path.isfile(gta5_train_closest_sv_path)==True:
    os.remove(gta5_train_closest_sv_path)
topen=open(citys_train_list,'r')  # open the target labels file
sopen=open(gta5_train_closest_sv_path, 'a')

if os.path.isfile(citys_cls_distr_path)==False:

    citys_list=os.listdir(citys_dir)
    citys_cls_distrs=[]
    citys_img_paths = []
    citys_list_num=len(citys_list)
    for i in range(0,citys_list_num):
        if i%100==0:
            print('%d of %d' % (i,citys_list_num))
        colored=citys_list[i][-9:-4]
        if colored!='color':
            png_path=citys_dir+citys_list[i]
            # png=mpimg.imread(png_path)
            png=np.array(Image.open(png_path))
            png_flat=png.flatten()
            # bin_min_max=np.array([0,21])
            # png_flat=np.concatenate((bin_min_max,png_flat)) #ensure the integral bins
            # n, bins, cls_distr= plt.hist(png_flat, bins=21,  facecolor='green', alpha=0.75)
            # the bins take the form [),until the last bin is for []
            citys_cls_distr=class_statistics(png_flat, 20) #including bg
            citys_cls_distr= citys_cls_distr[np.newaxis, :]#add a dim
            citys_cls_distrs.extend(citys_cls_distr)
            citys_img_paths.extend([citys_list[i]])
            # plt.imshow(png)
            # plt.show()
    citys_cls_distrs_np = np.array(citys_cls_distrs, dtype=int)
    citys_img_paths_np = np.array(citys_img_paths)
    np.save(citys_cls_distr_path, citys_cls_distrs_np)
    np.save(citys_imgpaths_path, citys_img_paths_np)
    # file = open('../result/citys_img_paths.txt', 'w')
    # file.write(str(citys_img_paths))
    # file.close
    print('Cityscapes color distribution calculation is done..')
else:
    citys_cls_distrs_np=np.load(citys_cls_distr_path)
    citys_img_paths_np = np.load(citys_imgpaths_path)

if os.path.isfile(gta5_cls_distr_path)==False:
    gta5_list = os.listdir(gta5_dir)
    gta5_cls_distrs = []
    gta5_img_paths=[]
    rnd_num=np.random.randint(0,2,500)
    add_account=0
    gta5_list_num=len(gta5_list)
    for i in range(0, gta5_list_num):
        if i%1000==0:
            print('%d of %d' % (i,gta5_list_num))
        png_path = gta5_dir + gta5_list[i]
        png = np.array(Image.open(png_path))
        png_flat = png.flatten()
        for k, v in id_to_trainid.items():
            png_flat[png_flat == k] = v
        gta5_cls_distr = class_statistics(png_flat, 20)  # including bg
        gta5_cls_distr = gta5_cls_distr[np.newaxis, :]  # add a dim
        gta5_cls_distrs.extend(gta5_cls_distr)
        gta5_img_paths.extend([gta5_list[i]])
        if (16 in png_flat) & (add_account < 500) & (rnd_num[add_account] == 1):
            add_account= add_account + 1
            sopen.write(gta5_list[i])

    gta5_cls_distrs_np = np.array(gta5_cls_distrs, dtype=int)
    gta5_img_paths_np = np.array(gta5_img_paths)
    np.save(gta5_cls_distr_path, gta5_cls_distrs_np)
    np.save(gta5_imgpaths_path, gta5_img_paths_np)
    print('GTA5 color distribution calculation is done..')
else:
    gta5_cls_distrs_np = np.load(gta5_cls_distr_path)
    gta5_img_paths_np = np.load(gta5_imgpaths_path)


img_pairs=[]
citys_pixels=2048*1024
gta5_pixels=1914*1052
citys_cls_distrs_np=np.true_divide(citys_cls_distrs_np,citys_pixels)
gta5_cls_distrs_np = np.true_divide(gta5_cls_distrs_np, gta5_pixels)
for i in range(citys_cls_distrs_np.shape[0]):
    cur_citys_cls_distrs_np=citys_cls_distrs_np[i,:]
    #---------find the same labels---
    # citys_cls_nozeros=cur_citys_cls_distrs_np# only a vector
    # gta5_cls_nozeros = gta5_cls_distrs_np#it is a mat
    # citys_cls_nozeros[citys_cls_nozeros!=0]=1
    # gta5_cls_nozeros[gta5_cls_nozeros != 0] = 1
    # compare_mat=citys_cls_nozeros*gta5_cls_nozeros
    # compare_mat_smus=compare_mat.sum(axis=1)
    # citys_cls_nozeros_sum=citys_cls_nozeros.sum()
    # same_pic_loc=np.where(compare_mat_smus==citys_cls_nozeros_sum)
    # gta5_cls_distrs_np[same_pic_loc,:]=np.ones(20)
    # ---------find the same labels---end
    citys_diff=abs(cur_citys_cls_distrs_np - gta5_cls_distrs_np)
    citys_sum = citys_cls_distrs_np[i, :] + gta5_cls_distrs_np #+0.00000001
    citys_iou=np.true_divide(citys_diff,citys_sum)
    citys_iou[np.isnan(citys_iou)] = 0

    #citys_iou_means=citys_iou.sum(axis=1)/20.
    # citys_kls=np.zeros([gta5_cls_distrs_np.shape[0], 1])
    # for j in range(gta5_cls_distrs_np.shape[0]):
    #     citys_kl=scipy.stats.entropy(citys_cls_distrs_np[i,:]+0.01,gta5_cls_distrs_np[j,:]+0.01)
    #     citys_kls[j]=citys_kl
    e_dis=np.sqrt(np.square(citys_iou).sum(axis=1))
    closest_img = np.where(e_dis == np.min(e_dis))
    #------------discard--------
    # citys_iou[citys_sum < 0.005] = 0
    # citys_iou_means=np.zeros([gta5_cls_distrs_np.shape[0], 1])
    # if i % 100 ==0:
    #     print '%d processed of %d' % (i, citys_cls_distrs_np.shape[0])
    # citys_iou[np.isnan(citys_iou)] = 0
    # for j in range(gta5_cls_distrs_np.shape[0]):
    #     cur_cls_num=np.nonzero(citys_iou[j])
    #     cur_cls_num=cur_cls_num[0].size
    #     citys_iou_mean = np.sum(citys_iou[j]) / cur_cls_num
    #     citys_iou_means[j] = citys_iou_mean
    #---------discard--end------
    # closest_img=np.where(citys_iou_means == np.min(citys_iou_means))
    closest_img_num=closest_img[0][0]
    img_pair=np.array([citys_img_paths_np[i], gta5_img_paths_np[closest_img_num]])
    img_pairs.extend([img_pair])
img_pairs_np = np.array(img_pairs)
# np.save('../result/img_pairs.npy', img_pairs_np)
# file = open('../result/img_pairs.txt', 'w')
# file.write(str(img_pairs))
# file.close()


tlines=topen.readlines()
for line in tlines:
    line_list=line.split('/')
    t_img_name=line_list[1].split('\n')[0]#source image name
    img_ord=np.where(img_pairs_np==t_img_name)[0][0]
    s_img_name=img_pairs_np[img_ord,1]+'\n'
    sopen.write(s_img_name)
topen.close()
sopen.close()

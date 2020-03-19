import os
import shutil
from PIL import Image
import numpy as np


id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

adapt_iter=6
gta5_lbl_dir = '/home/guest/data/data/GTA5/labels/'
gta5_img_dir ='/home/guest/data/data/GTA5/images/'
citys_lbl_dir = './result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_pt/cityscapes_train/'
citys_img_dir = '/home/guest/data/data/Cityscapes/data/leftImg8bit/train/'

gta5_train_closest_list_path='./result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_pt/gta5_train_closest.txt'
# gta5_train_closest_list_path='../result/adapt_slective_pt/null.txt'
citys_retrain_list_path='./result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_pt/confidence_values/retrain.txt'

# output path
mixed_imgs_dir='./result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_mixdata/images/'
mixed_lbls_dir='./result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_mixdata/labels/'
train_mixed_sv_path='./result/adapt_slective_pt/adapt_iter'+str(adapt_iter)+'_mixdata/train_mixed.txt'

city_train_class_name={'aachen','bochum','bremen','cologne','darmstadt','dusseldorf','erfurt','hamburg','hanover','jena','krefeld',
            'monchengladbach','strasbourg','stuttgart','tubingen','ulm','weimar','zurich'}


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

if not os.path.exists(mixed_imgs_dir):
    os.makedirs(mixed_imgs_dir)
if not os.path.exists(mixed_lbls_dir):
    os.makedirs(mixed_lbls_dir)
citys_file=open(citys_retrain_list_path,'r')
gta5_file=open(gta5_train_closest_list_path,'r')
mixed_file=open(train_mixed_sv_path,'w')

for line in citys_file.readlines():
    s_img_file_path = citys_img_dir+line[0:len(line)-1]
    t_img_file_path = mixed_imgs_dir+line[0:len(line)-1].split('/')[1]
    s_lbl_file_path = citys_lbl_dir+line[0:len(line)-1].split('/')[1]
    t_lbl_file_path = mixed_lbls_dir+line[0:len(line)-1].split('/')[1]
    shutil.copyfile(s_img_file_path,t_img_file_path)
    shutil.copyfile(s_lbl_file_path,t_lbl_file_path)
    mixed_file.write(line.split('/')[1])
    # #===begin===
    #
    # img_embed = np.array(Image.open(s_img_file_path))
    # lbl_embed = np.array(Image.open(s_lbl_file_path))
    # m=img_embed.shape[0]
    # n=img_embed.shape[1]
    # m_1 = m / 4
    # m_2 = m / 4 + m * 1 / 2
    # n_1 = n / 4
    # n_2 = n / 4 + n * 1 / 2
    # height_embed = m * 1 / 2
    # width_embed = n * 1 / 2
    # image_embed = img_embed[m_1:m_2, n_1:n_2,:]
    # label_embed = lbl_embed[m_1:m_2, n_1:n_2]
    # image_embed=Image.fromarray(image_embed)
    # label_embed=Image.fromarray(label_embed.astype(np.uint8)).convert('P')
    # img_emb_sv_path = t_img_file_path[:-4]+'_emb0.5.png'
    # lbl_emb_sv_path = t_lbl_file_path[:-4]+'_emb0.5.png'
    # image_embed.save(img_emb_sv_path)
    # label_embed.save(lbl_emb_sv_path)
    # mixed_file.write(line.split('.')[0].split('/')[1]+'_emb0.5.png'+'\n')
    #===end====
for line in gta5_file.readlines():
    # s_img_file_path = gta5_img_dir + line[0:len(line)-1]  #byzq
    # t_img_file_path = mixed_imgs_dir + line[0:len(line)-1]
    # shutil.copyfile(s_img_file_path, t_img_file_path)
    # t_lbl_file_path = mixed_lbls_dir + line[0:len(line)-1].strip()
    # lbl_path = gta5_lbl_dir + line[0:len(line)-1]
    # lbl = np.array(Image.open(lbl_path))

    s_img_file_path = gta5_img_dir + line.strip()
    t_img_file_path = mixed_imgs_dir + line.strip()
    shutil.copyfile(s_img_file_path, t_img_file_path)
    t_lbl_file_path = mixed_lbls_dir + line.strip()
    lbl_path = gta5_lbl_dir + line.strip()
    lbl = np.array(Image.open(lbl_path))

    label_copy = 255 * np.ones(lbl.shape, dtype=np.float32)
    for k, v in id_to_trainid.items():
        label_copy[lbl == k] = v
    lbl=label_copy
    # for k, v in id_to_trainid.items():
    #     lbl[lbl == k] = v
    lbl_col = colorize_mask(lbl)
    lbl_col.save(t_lbl_file_path)
    mixed_file.write(line)
citys_file.close()
gta5_file.close()
mixed_file.close()
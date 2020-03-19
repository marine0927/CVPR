from PIL import Image
import numpy as np
train_mixed_sv_path='../result/adapt_iter1/train_mixed.txt'
train_dir='../data/Mixeddata1/labels/'
train_list=open(train_mixed_sv_path,'r')
all_label_uni=[]
lbl_num=1
for line in train_list.readlines():
    png_path=train_dir+line[:-1]
    png=Image.open(png_path)
    label=np.array(png)
    label_uni=np.unique(label)
    all_label_uni.extend(list(label_uni))
all_label_uni=np.array(all_label_uni)
for lbl_num in range(19):
    label_loc= all_label_uni==lbl_num
    static=all_label_uni[label_loc]
    lbl_amount=static.size
    print('%d : %d' % (lbl_num,lbl_amount))
# all_uni=np.unique(np.array(all_label_uni))
zq=1
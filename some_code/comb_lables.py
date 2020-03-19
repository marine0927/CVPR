citys_file=open('../result/confidence_values/retrain.txt','r')
gta5_file=open('../dataset/gta5_list/train_closest.txt','r')
mixed_file=open('../dataset/gta5_list/train_mixed.txt','w')
for line in citys_file.readlines():
    mixed_file.write('../ext_images/'+line)
for line in gta5_file.readlines():
    mixed_file.write(line)
citys_file.close()
gta5_file.close()
mixed_file.close()
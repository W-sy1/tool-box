import numpy as np
from PIL import Image
import os
import torch
from os.path import join

labelp_dir = r'F:\fengjie_edge\data_fb\train_fb\labelp'
labell_dir = r'F:\fengjie_edge\data_fb\train_fb\labell'
save_dir = r'F:\fengjie_edge\data_fb\train_fb\labelp-l'
namelist= r'F:\fengjie_edge\data_fb\train_fb\train.lst'

def nameget(list):
    with open(list, 'r') as f:
        filelist = f.readlines()
        for i in range(len(filelist)):
            filelist[i] = filelist[i].split('\n')[0]
    return (filelist)

# fig= plt.figure(num="abc",figsize=(16,12),dpi=100,facecolor='red',edgecolor='green',frameon=False)

def sub(labelp_dir, labell_dir, save_dir, namel):
    labelp = sorted(os.listdir(labelp_dir))
    labell = sorted(os.listdir(labell_dir))
    mask_list = []
    for item in range(len(labelp)):
        p = Image.open(labelp_dir + '/' + labelp[item])
        l = Image.open(labell_dir + '/' + labell[item])
        # 转换成array形式，便于以下标形式获取值
        p = np.array(p, dtype=np.float32)
        l = np.array(l, dtype=np.float32)
        mask = p - l
        mask = Image.fromarray(mask.astype(np.uint8))
        mask.save(join(save_dir, "%s" % namel[item]))

torch.cuda.set_device(0)
if os.path.exists(save_dir) != True:
    os.makedirs(save_dir)

name = nameget(namelist)
sub(labelp_dir, labell_dir, save_dir, name)

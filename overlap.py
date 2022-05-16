# 将分割图和原图合在一起
import os
from PIL import Image
import numpy as np
import torch
from os.path import join
import matplotlib.pyplot as plt

# image1 原图
# image2 分割图
img_dir = r'D:\MyDate\MASD\256\vaild\image'
labpre_dir = r'D:\PyCharmProject\first\glanet\GLANetResultMS'
lab_dir = r'D:\MyDate\MASD\256\vaild\label'
save_dir = r'D:\PyCharmProject\first\glanet\overlapResultMS'
namelist=r'D:\MyDate\MASD\256\vaild\test.lst'

def nameget(list):
    with open(list, 'r') as f:
        filelist = f.readlines()
        for i in range(len(filelist)):
            filelist[i] = filelist[i].split('\n')[0]
    return (filelist)

def overlap(img_dir, labpre_dir, lab_dir, save_dir, namelist):
    image_dir = sorted(os.listdir(img_dir))
    labelpre_dir = sorted(os.listdir(labpre_dir))
    label_dir = sorted(os.listdir(lab_dir))
    for item in range(len(image_dir)):
        image = Image.open(img_dir + '/' + image_dir[item])
        label_pre = Image.open(labpre_dir + '/' + labelpre_dir[item])
        label = Image.open(lab_dir + '/' + label_dir[item])
        image = image.convert('RGBA')
        label_pre = label_pre.convert('RGBA')
        label = label.convert('RGBA')

        x, y = label_pre.size
        for i in range(x):
            for j in range(y):
                color = label.getpixel((i, j))
                Mean = np.mean(list(color[:-1]))
                if Mean < 255:  # 我的标签区域为白色，非标签区域为黑色
                    color = color[:-1] + (0,)  # 若非标签区域则设置为透明
                else:
                    color = (255, 0, 0, 255)  # 标签区域设置为红色，前3位为RGB值，最后一位为透明度情况，255为完全不透明，0为完全透明
                label.putpixel((i, j), color)
        # image = Image.blend(image, label, 0.5)
        image.paste(label, (0, 0), label)  # 贴图操作
        for i in range(x):
            for j in range(y):
                color = label_pre.getpixel((i, j))
                Mean = np.mean(list(color[:-1]))
                if Mean < 255:  # 我的标签区域为白色，非标签区域为黑色
                    color = color[:-1] + (0,)  # 若非标签区域则设置为透明
                else:
                    color = (0, 255, 0, 255)  # 标签区域设置为红色，前3位为RGB值，最后一位为透明度情况，255为完全不透明，0为完全透明
                label_pre.putpixel((i, j), color)
        # image = Image.blend(image, label_pre, 0.5)  #叠加
        image.paste(label_pre, (0, 0), label_pre)     #贴图
        image.save(join(save_dir, "%s" %namelist[item]))

if __name__ == '__main__':
    torch.cuda.set_device(0)
    namelist = nameget(namelist)
    if os.path.exists(save_dir) != True:
        os.makedirs(save_dir)
    overlap(img_dir, labpre_dir, lab_dir, save_dir, namelist)

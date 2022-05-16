import fnmatch
import os.path
import pandas as pd
import numpy as np
import sys
import shutil
import os, sys
import random

########## 多个文件夹下的文件移动到一个文件夹下并修改名字    ###########
def mmoveo():
    InputStra = r'E:\plough_data\CS70\img'
    InputStrb = '*.tif'
    save_path = r'E:\plough_data\CS70\dataset_cs\image_clip'
    if os.path.exists(save_path) != True:
        os.makedirs(save_path)
    a_list = fnmatch.filter(os.listdir(InputStra), InputStrb)
    for i in range(len(a_list)):
        i = str(i)
        image = os.listdir(os.path.join(InputStra + '/' + i + '/' + "split"))
        img_path = os.path.join(InputStra + '/' + i + '/' + "split")
        print(image)
        for n in range(4):
            img = os.path.join(img_path + '/' + image[n])
            name = save_path + '/' + i + '_' + str(n) + '.tif'
            shutil.copy(img,name)
            print(save_path + '/' + i + '_' + str(n) + '.tif')
            print(img)
        # print(img)
# mmoveo()
########## 多个文件夹下的文件移动到一个文件夹下并修改名字    ###########


##########  随机抽取训练、测试、验证集  ###########
def randmove():
    img = r'E:\plough_data\CS70\dataset_cs\image_clip'
    edge = r'E:\plough_data\CS70\dataset_cs\line_clip'
    # seg = r'G:\Sample\dataset2_x\seglabel_x'
    tarimg = r'E:\plough_data\CS70\dataset_cs\test\image'
    taredge = r'E:\plough_data\CS70\dataset_cs\test\edgelabel'
    # tarseg = r'G:\Sample\dataset2_x\val\seglabel'
    pathDir = os.listdir(img)
    sample = random.sample(pathDir, 50)
    print(sample)
    for name in sample:
        shutil.move(img + '/' + name, tarimg + '/' + name)
        shutil.move(edge + '/' + name, taredge + '/' + name)
        # shutil.move(seg + '/' + name, tarseg + '/' + name)
randmove()
##########  随机抽取训练、测试、验证集  ###########

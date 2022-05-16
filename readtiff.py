import fnmatch
import os.path
import pandas as pd
import numpy as np
import sys
import shutil
import os, sys
import random
import gdal
import osr
from osgeo import gdal


gdal.AllRegister() #先载入数据驱动，也就是初始化一个对象，让它“知道”某种数据结构，但是只能读，不能写
ds = gdal.Open(r"G:\Sample\Image\0000000001.tif") # 打开文件
bands = ds.RasterCount# 获取波段数
img_width,img_height = ds.RasterXSize,ds.RasterYSize # 获取影像的宽高
geotrans = ds.GetGeoTransform() # 获取影像的投影信息
im_proj = ds.GetProjection()  # 地图投影信息
im_data = ds.ReadAsArray(0, 0, img_width, img_height)  # 此处读取整张图像

print(img_height,img_width)
print(geotrans)
print(bands)
print(im_proj)
print(im_data.shape)


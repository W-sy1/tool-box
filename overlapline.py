import cv2
import numpy as np
# from PIL import Image
import matplotlib.pyplot as plt
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# Image.MAX_IMAGE_PIXELS = None
# imgfile = r'F:\Download\bigmap\Rectangle_#2_卫图\pre_large\merge.tif'	 #原图路径
# pngfile = r'F:\Download\bigmap\Rectangle_#2_卫图\hqyp.tif' 	#mask路径

imgfile = r'F:\PycharmProject\plough\data\valid\img\100_0_0.tif'	 #原图路径
pngfile = r'F:\Download\Wechat\WeChat Files\wxid_f5qe5wr46xnb22\FileStorage\File\2022-05\ForB\resultBin\100_0_0.tif' 	#mask路径
img = cv2.imread(imgfile, 1)
mask = cv2.imread(pngfile, 0)

# img = Image.open(imgfile)
# mask = Image.open(pngfile)
# img = np.array(img)
# mask = np.array(mask)
# img = img.convert('RGBA')
# mask = mask.convert('RGBA')

# print(img.shape, mask.shape)
# print(mask)
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)	  #findContours函数用于找出边界点
cv2.drawContours(img, contours, -1, (255, 0, 0), 1)  	##drawContours函数用于根据边界点画出图形
img = img[:, :, ::-1]
img[..., 2] = np.where(mask == 1, 255, img[..., 2])
cv2.imshow('img', img)
cv2.waitKey(0)
save_path = r'F:\Download\Wechat\WeChat Files\wxid_f5qe5wr46xnb22\FileStorage\File\2022-05\ForB\resultBin\over\100_0_0.tif'
cv2.imwrite(save_path, img)

IMREAD_UNCHANGED = -1#不进行转化，比如保存为了16位的图片，读取出来仍然为16位。
IMREAD_GRAYSCALE = 0#进行转化为灰度图，比如保存为了16位的图片，读取出来为8位，类型为CV_8UC1。
IMREAD_COLOR = 1#进行转化为RGB三通道图像，图像深度转为8位
IMREAD_ANYDEPTH = 2#保持图像深度不变，进行转化为灰度图。
IMREAD_ANYCOLOR = 4#若图像通道数小于等于3，则保持原通道数不变；若通道数大于3则只取取前三个通道。图像深度转为8位

# from PIL import Image
# import cv2
# import numpy as np
# imgfile = r'F:\Download\bigmap\suzhou\suzhou_Level_18_crop.tif'	 #原图路径
# # pngfile = r'F:\PycharmProject\plough\result\result\dexined_post\dexined_hf1\75_0_0gj.tif' 	#mask路径
# pngfile = r'F:\Download\bigmap\suzhou\suzhou_Level_18_crop\szpre_merge.tif'
# img = cv2.imread(imgfile, 1)
# mask = cv2.imread(pngfile, 0)
# # mask = cv2.resize(mask, (9472, 5888))
# # mask = Image.open(pngfile)
# print(img.shape, mask.shape)
#
# line =  np.zeros((mask.shape[0], mask.shape[1]), np.uint8) # 生成一个空灰度图像
# contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)	  #findContours函数用于找出边界点
# cv2.drawContours(img, contours, -1, (255, 0, 0), 1)  	##drawContours函数用于根据边界点画出图形
# img = img[:, :, ::-1]
# print(img[..., 2].shape)
# img[..., 2] = np.where(mask == 1, 255, img[..., 2])
# cv2.imshow('img', img)
# cv2.waitKey(0)
# save_path = r'F:\Download\bigmap\苏州\苏州_Level_18_crop\szpre_merge1_overlapline.tif'
# cv2.imwrite(save_path, img)

# cv2.drawContours(line, contours, -1, (255, 0, 0), 1)  	##drawContours函数用于根据边界点画出图形
# line= line[:, :]
# print(line[..., 2].shape)
# line = np.w   here(mask == 1, 255, line)
# cv2.imshow('img', line)
# cv2.waitKey(0)
# save_path = r'F:\PycharmProject\plough\result\result\dexined_post\dexined_overlapline1\75_0_0.tif'
# cv2.imwrite(save_path, line)


#################################################
###预测图转为线叠加在原图上
################################################
# # 将分割图和原图合在一起
# import os
# from PIL import Image
# import cv2
# import numpy as np
# import torch
# from os.path import join
# import matplotlib.pyplot as plt
#
# # image1 原图
# # image2 分割图
# img_dir = r'D:\MyDate\MASD\256\test\image'
# labpre_dir = r'D:\PyCharmProject\first\glanet\GLANetResult_jointloss_MS32_strip pool'
# # lab_dir = r'D:\MyDate\MASD\256\vaild\label'
# save_dir = r'D:\PyCharmProject\first\glanet\overlaplineResultMS'
# namelist=r'D:\MyDate\MASD\256\test\test.lst'
#
# def nameget(list):
#     with open(list, 'r') as f:
#         filelist = f.readlines()
#         for i in range(len(filelist)):
#             filelist[i] = filelist[i].split('\n')[0]
#     return (filelist)
#
# def overlap(img_dir, labpre_dir, save_dir, namelist):
#     image_dir = sorted(os.listdir(img_dir))
#     labelpre_dir = sorted(os.listdir(labpre_dir))
#     for item in range(len(image_dir)):
#         image = cv2.imread(img_dir + '/' + image_dir[item], 1)
#         label_pre = cv2.imread(labpre_dir + '/' + labelpre_dir[item], 0)
#         contours, _ = cv2.findContours(label_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)	  #findContours函数用于找出边界点
#         cv2.drawContours(image, contours, -1, (255, 0, 0), 1)  	##drawContours函数用于根据边界点画出图形
#         image = image[:, :, ::-1]
#         image[..., 2] = np.where(label_pre == 1, 255, image[..., 2])
#         # image = np.array(image)
#         # cv2.imshow('image', image)
#         # cv2.waitKey(0)
#         cv2.imwrite(join(save_dir, namelist[item]), image)
#         print(image)
#         # image = np.array(image)
#         # image.save(join(save_dir, "%s" %namelist[item]), quality=95)
#
# if __name__ == '__main__':
#     torch.cuda.set_device(0)
#     namelist = nameget(namelist)
#     if os.path.exists(save_dir) != True:
#         os.makedirs(save_dir)
#     overlap(img_dir, labpre_dir, save_dir, namelist)


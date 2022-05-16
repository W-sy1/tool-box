# import os
# import gdal
# import numpy as np
#
#
# #  读取tif数据集
# def readTif(fileName):
#     dataset = gdal.Open(fileName)
#     if dataset == None:
#         print(fileName + "文件无法打开")
#     return dataset
#
#
# #  保存tif文件函数
# def writeTiff(im_data, im_geotrans, im_proj, path):
#     if 'int8' in im_data.dtype.name:
#         datatype = gdal.GDT_Byte
#     elif 'int16' in im_data.dtype.name:
#         datatype = gdal.GDT_UInt16
#     else:
#         datatype = gdal.GDT_Float32
#     if len(im_data.shape) == 3:
#         im_bands, im_height, im_width = im_data.shape
#     elif len(im_data.shape) == 2:
#         im_data = np.array([im_data])
#         im_bands, im_height, im_width = im_data.shape
#     # 创建文件
#     driver = gdal.GetDriverByName("GTiff")
#     dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
#     if (dataset != None):
#         dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
#         dataset.SetProjection(im_proj)  # 写入投影
#     for i in range(im_bands):
#         dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
#     del dataset
#
#
# '''
# 滑动窗口裁剪函数
# TifPath 影像路径
# SavePath 裁剪后保存目录
# CropSize 裁剪尺寸
# RepetitionRate 重复率
# '''
#
#
# def TifCrop(TifPath, SavePath, CropSize, RepetitionRate):
#     dataset_img = readTif(TifPath)
#     width = dataset_img.RasterXSize
#     height = dataset_img.RasterYSize
#     proj = dataset_img.GetProjection()
#     geotrans = dataset_img.GetGeoTransform()
#     img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据
#
#     #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
#     new_name = len(os.listdir(SavePath)) + 1
#     #  裁剪图片,重复率为RepetitionRate
#
#     for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
#         for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
#             #  如果图像是单波段
#             if (len(img.shape) == 2):
#                 cropped = img[
#                           int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
#                           int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
#             #  如果图像是多波段
#             else:
#                 cropped = img[:,
#                           int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
#                           int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
#             #  写图像
#             writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
#             #  文件名 + 1
#             new_name = new_name + 1
#     #  向前裁剪最后一列
#     for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
#         if (len(img.shape) == 2):
#             cropped = img[int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
#                       (width - CropSize): width]
#         else:
#             cropped = img[:,
#                       int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
#                       (width - CropSize): width]
#         #  写图像
#         writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
#         new_name = new_name + 1
#     #  向前裁剪最后一行
#     for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
#         if (len(img.shape) == 2):
#             cropped = img[(height - CropSize): height,
#                       int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
#         else:
#             cropped = img[:,
#                       (height - CropSize): height,
#                       int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
#         writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
#         #  文件名 + 1
#         new_name = new_name + 1
#     #  裁剪右下角
#     if (len(img.shape) == 2):
#         cropped = img[(height - CropSize): height,
#                   (width - CropSize): width]
#     else:
#         cropped = img[:,
#                   (height - CropSize): height,
#                   (width - CropSize): width]
#     writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
#     new_name = new_name + 1
#
#
# #  将影像1裁剪为重复率为0.1的256×256的数据集
# TifCrop(r"Data\data2\tif\data2.tif",
#         r"Data\train\image1", 256, 0.1)
# TifCrop(r"Data\data2\label\label.tif",
#         r"data\train\label1", 256, 0.1)


import numpy as np
from PIL import Image
import os
import torch
img_dir=r'F:\奉节影像和样本\data - 副本\image'
imgcrop_dir=r'F:\奉节影像和样本\data - 副本\image_clip'
lab_dir=r'F:\奉节影像和样本\data - 副本\labelpolygon'
labcrop_dir=r'F:\奉节影像和样本\data - 副本\labelpolygon_clip'
labl_dir=r'F:\奉节影像和样本\data - 副本\labelline'
labcropl_dir=r'F:\奉节影像和样本\data - 副本\labelline_clip'

def crop(img_dir, lab_dir, labl_dir, imgcrop_dir, labcrop_dir, labcropl_dir):
    image_dir = sorted(os.listdir(img_dir))
    label_dir = sorted(os.listdir(lab_dir))
    labell_dir = sorted(os.listdir(labl_dir))
    image_list = []
    label_list = []
    labell_list = []
    for item in range(len(image_dir)):
        image = Image.open(img_dir + '/' + image_dir[item])
        label = Image.open(lab_dir + '/' + label_dir[item])
        labell = Image.open(labl_dir + '/' + labell_dir[item])
        #转换成array形式，便于以下标形式获取值
        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        labell = np.array(labell, dtype=np.float32)
        h_step = image.shape[0] // 256
        w_step = image.shape[1] // 256
        # h_rest = -(image.shape[0] - 256 * h_step)
        # w_rest = -(image.shape[1] - 256 * w_step)
        for h in range(h_step):
            for w in range(w_step):
                # 划窗采样
                image_sample = image[(h * 256 * 1):(h * 256 * 1 + 256), (w * 256 * 1):(w * 256 * 1 + 256), :]
                label_sample = label[(h * 256 * 1):(h * 256 * 1 + 256), (w * 256 * 1):(w * 256 * 1 + 256)]
                labell_sample = labell[(h * 256 * 1):(h * 256 * 1 + 256), (w * 256 * 1):(w * 256 * 1 + 256)]
                image_list.append(image_sample)
                label_list.append(label_sample)
                labell_list.append(labell_sample)
            image_list.append(image[(h * 256):(h * 256 + 256), -256:, :])
            label_list.append(label[(h * 256):(h * 256 + 256), -256:])
            labell_list.append(labell[(h * 256):(h * 256 + 256), -256:])
        for w in range(w_step):
            image_list.append(image[-256:, (w * 256):(w * 256 + 256), :])
            label_list.append(label[-256:, (w * 256):(w * 256 + 256)])
            labell_list.append(labell[-256:, (w * 256):(w * 256 + 256)])
        image_list.append(image[-256:, -256:, :])
        label_list.append(label[-256:, -256:])
        labell_list.append(labell[-256:, -256:])
    print(len(image_list),len(label_list),len(labell_list))

    for i in range(len(image_list)):
        image = image_list[i]
        label = label_list[i]
        labell = labell_list[i]
        #转换成图片形式，便于保存
        image = Image.fromarray(np.uint8(image))
        label = Image.fromarray(np.uint8(label))
        labell = Image.fromarray(np.uint8(labell))
        image.save(os.path.join(imgcrop_dir, str(i) + '.tif'))
        label.save(os.path.join(labcrop_dir, str(i) + '.tif'))
        labell.save(os.path.join(labcropl_dir, str(i) + '.tif'))


torch.cuda.set_device(0)
if os.path.exists(imgcrop_dir) | os.path.exists(labcrop_dir) | os.path.exists(labcropl_dir) != True:
    os.makedirs(imgcrop_dir)
    os.makedirs(labcrop_dir)
    os.makedirs(labcropl_dir)
crop(img_dir, lab_dir, labl_dir, imgcrop_dir, labcrop_dir, labcropl_dir)


# def total_predict(ori_image):
#     h_step = ori_image.size[0] // 256
#     w_step = ori_image.size[1] // 256
#
#     h_rest = -(ori_image.size[0] - 256 * h_step)
#     w_rest = -(ori_image.size[1] - 256 * w_step)
#
#     image_list = []
#     predict_list = []
#     # 循环切图
#     for h in range(h_step):
#         for w in range(w_step):
#             # 划窗采样
#             image_sample = ori_image[(h * 256):(h * 256 + 256),(w * 256):(w * 256 + 256), :]
#             image_list.append(image_sample)
#         image_list.append(ori_image[(h * 256):(h * 256 + 256), -256:, :])
#     for w in range(w_step - 1):
#         image_list.append(ori_image[-256:, (w * 256):(w * 256 + 256), :])
#     image_list.append(ori_image[-256:, -256:, :])
#
# total_predict(image)

    # # 对每个图像块预测
    # # predict
    # for image in image_list:
    #     x_batch = image / 255.0
    #     x_batch = np.expand_dims(x_batch, axis=0)
    #     feed_dict = {img: x_batch}
    #     pred1 = sess.run(pred, feed_dict=feed_dict)
    #
    #     predict = np.argmax(pred1, axis=3)
    #     predict = np.squeeze(predict).astype(np.uint8)
    #     # 保存覆盖小图片
    #     predict_list.append(predict)

    # 将预测后的图像块再拼接起来
    # count_temp = 0
    # tmp = np.ones([ori_image.shape[0], ori_image.shape[1]])
    # #print('tmp shape: ', tmp.shape)
    # for h in range(h_step):
    #     for w in range(w_step):
    #         tmp[
    #         h * 256:(h + 1) * 256,
    #         w * 256:(w + 1) * 256
    #         ] = predict_list[count_temp]
    #         count_temp += 1
    #     tmp[h * 256:(h + 1) * 256, w_rest:] = predict_list[count_temp][:, w_rest:]
    #     count_temp += 1
    # for w in range(w_step - 1):
    #     tmp[h_rest:, (w * 256):(w * 256 + 256)] = predict_list[count_temp][h_rest:, :]
    #     count_temp += 1
    # # tmp[h_rest:, w_rest:] = predict_list[count_temp][h_rest:, w_rest:]
    # tmp[-257:-1, -257:-1] = predict_list[count_temp][:, :]
    # return tmp

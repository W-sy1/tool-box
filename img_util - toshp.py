# -*- coding: utf-8 -*-
"""
Image operations
Author: Zhou Ya'nan
"""
import os
import time
import datetime
import argparse
import gc
import numpy as np
# import matplotlib.pyplot as plt
from skimage import io, transform
from osgeo import gdal, osr
import shapefile # 使用pyshp
os.environ['PROJ_LIB'] = r'D:\Anaconda3\envs\python36\Library\share\proj'

###########################################################
def resize_image(src_img, target_size):
    """Stretch image to target size.

    Parameters
    ----------
    src_img : tensor,
        The input image for features (np.array).
    target_size :
        The size for output images.
    """
    src_shape = src_img.shape

    dst_img = np.ones((target_size[1], target_size[0], src_shape[2]), dtype=None, order='C')

    for channel in range(0, src_shape[2]):
        channel_array = src_img[:, :, channel]

        # transform.resize()要求数值的取值范围是[0~1]
        channel_max = channel_array.max()
        channel_min = channel_array.min()
        channel_array = (channel_array-channel_min)/(channel_max-channel_min)
        resize_array = transform.resize(channel_array, (target_size[1], target_size[0]))
        resize_array = resize_array*(channel_max-channel_min)+channel_min

        resize_array = resize_array.reshape([target_size[1], target_size[0], -1])
        dst_img[:, :, channel:channel + 1] = resize_array

    # io.imsave('G:/experiments-dataset/hunan-lixian-crop2/000-sar/ls-char/1.tiff', dst_img, plugin='tifffile')

    return dst_img

def load_gdalimage(image_path):
    """Load image on disk, with gdal driver.

    Parameters
    ----------
    image_path : string
        The path of image.

    :return : np.array
    """
    data_type = None

    image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if not image_ds:
        print("Fail to open image {}".format(image_path))
        return None
    else:
        print("Driver: {}/{}".format(image_ds.GetDriver().ShortName, image_ds.GetDriver().LongName))
        print("Size is {} x {} x {}".format(image_ds.RasterXSize, image_ds.RasterYSize, image_ds.RasterCount))  #x（width），y（）height方向上的像素数，波段数

        print("Projection is {}".format(image_ds.GetProjection()))
        geotransform = image_ds.GetGeoTransform()  #六个返回值：0左上角x坐标，1水平分辨率，2旋转参数，3左上角y坐标，4旋转参数，5竖直分辨率
        if geotransform:
            print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))   #起始左上角x，y坐标
            print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))  #水平、垂直分辨率

        raster_band0 = image_ds.GetRasterBand(1)
        if raster_band0:
            data_type = raster_band0.DataType
            print("Band Type = {}".format(gdal.GetDataTypeName(data_type)))

            min = raster_band0.GetMinimum()
            max = raster_band0.GetMaximum()
            if not min or not max:
                (min, max) = raster_band0.ComputeRasterMinMax(True)
            print("Min = {:.3f}, Max = {:.3f}".format(min, max))

            if raster_band0.GetOverviewCount() > 0:
                print("Band has {} overviews".format(raster_band0.GetOverviewCount()))
            if raster_band0.GetRasterColorTable():
                print("Band has a color table with {} entries".format(raster_band0.GetRasterColorTable().GetCount()))

    image_shape = (image_ds.RasterYSize, image_ds.RasterXSize, image_ds.RasterCount)
    # image_array = np.zeros(image_shape, dtype=None)
    image_array = image_ds.ReadAsArray(xoff=0, yoff=0, xsize=image_shape[1], ysize=image_shape[0],
                                       buf_xsize=image_shape[1], buf_ysize=image_shape[0],
                                       buf_type=data_type)
    # image_array = image_array[np.newaxis,:,:]
    image_array = image_array.transpose(1, 2, 0)


    return image_array

def saveas_gdalimage(img_array, save_path, format='GTiff'):
    """Save image on disk, with tiff format.

    Parameters
    ----------
    img_array : np.array,
        The image for saving (np.array).
    save_path :
        The path for output images.
        :param format:
    """
    print("### Writing image {}".format(save_path))
    img_shape = img_array.shape

    file_format = format
    file_driver = gdal.GetDriverByName(file_format)

    metadata = file_driver.GetMetadata()
    if metadata.get(gdal.DCAP_CREATE) == "YES":
        print("Driver {} supports Create() method.".format(file_format))
    if metadata.get(gdal.DCAP_CREATE) == "YES":
        print("Driver {} supports CreateCopy() method.".format(file_format))

    dst_ds = file_driver.Create(save_path, xsize=img_shape[1], ysize=img_shape[0], bands=img_shape[2],
                                eType=gdal.GDT_Float32)
    if not dst_ds:
        print("Fail to create image {}".format(save_path))
        return False

    # dst_ds.SetGeoTransform([444720, 30, 0, 3751320, 0, -30])
    # srs = osr.SpatialReference()
    # srs.SetUTM(11, 1)
    # srs.SetWellKnownGeogCS("NAD27")
    # dst_ds.SetProjection(srs.ExportToWkt())

    # raster = np.zeros((img_shape[1], img_shape[0]), dtype=np.float32)
    for channel in range(0, img_shape[2]):
        print("### Writing band {}".format(channel))
        channel_array = img_array[:, :, channel:channel + 1]
        channel_array = channel_array.reshape(img_shape[0], img_shape[1])
        channel_array = channel_array.astype(np.float32)

        # dst_ds.GetRasterBand(channel + 1).WriteArray(channel_array)
        dst_ds.GetRasterBand(channel + 1).WriteRaster(0, 0, img_shape[1], img_shape[0], channel_array.tostring())
        # if (channel+1) % 16 == 0: gc.collect()

    # print("### Building overviews")
    # dst_ds.BuildOverviews("NEAREST")
    dst_ds.FlushCache()
    return True

def split_image(img_path, sub_size, overlay_size):
    """Split a big image into multi sub images.
    Parameters
    ----------
    img_path : string,
        The input image.
    sub_size : tuple,
        The size for output images.
    overlay_size : tuple,
        The size for overlay area.
    """
    src_img = load_gdalimage(img_path)
    src_img_shape = src_img.shape

    if (src_img_shape[0] < sub_size[0] or src_img_shape[1] < sub_size[1]):
        print("Sub size {} is too big.".format(sub_size))
        return False
    if (sub_size[0] < overlay_size[0] or sub_size[1] < overlay_size[1]):
        print("Overlay size {} is too big.".format(sub_size))
        return False

    sub_image_dir = os.path.join(os.path.splitext(img_path)[0], 'split')
    if not os.path.exists(sub_image_dir):
        os.makedirs(sub_image_dir)
    if not os.path.exists(sub_image_dir):
        print("Target directory {} not exist.".format(sub_image_dir))
        return False
    image_ext = os.path.splitext(img_path)[1]

    row_num = 1 + int(src_img_shape[0] / (sub_size[0] - overlay_size[0]))
    col_num = 1 + int(src_img_shape[1] / (sub_size[1] - overlay_size[1]))
    for row in range(0, row_num):
        row_stt = row * (sub_size[0] - overlay_size[0])
        row_end = row_stt + sub_size[0]
        if row_end > src_img_shape[0] : row_end = src_img_shape[0]

        for col in range(0, col_num):
            col_stt = col * (sub_size[1] - overlay_size[1])
            col_end = col_stt + sub_size[1]
            if col_end > src_img_shape[1] : col_end = src_img_shape[1]

            sub_img_path = os.path.join(sub_image_dir, "{}_{}{}".format(row, col, image_ext))
            sub_array = src_img[row_stt:row_end, col_stt:col_end, :]
            io.imsave(sub_img_path, sub_array)
            print("Creating image on {}".format(sub_img_path))

            if col_end >= src_img_shape[1]:
                break
        if row_end >= src_img_shape[0]:
            break

    print("### Split image done.")
    return True

def merge_image(sub_image_dir, sub_size, overlay_size, target_path, target_shape):
    """Merge multi sub images into a big one.

    Parameters
    ----------
    sub_image_dir : string,
        The input image.
    sub_size : tuple,
        The size for output images.
    overlay_size : tuple,
        The size for overlay area.
    """
    start_time = time.time()
    target_img = np.zeros(target_shape, dtype=None)

    sub_image_files = list(filter(lambda filename: os.path.splitext(filename)[1] == '.tiff', os.listdir(sub_image_dir)))
    for sub_file in sub_image_files:
        print("Merge image {}".format(sub_file))

        file_name = os.path.basename(sub_file)
        split_list = (os.path.splitext(file_name)[0]).split('_')
        row = int(split_list[0])
        col = int(split_list[1])

        row_stt = row * (sub_size[0] - overlay_size[0])
        row_end = row_stt + sub_size[0]
        if row > 0: row_stt = row_stt + overlay_size[0] / 2
        if row_end > target_shape[0]: row_end = target_shape[0]

        col_stt = col * (sub_size[1] - overlay_size[1])
        col_end = col_stt + sub_size[1]
        if col > 0: col_stt = col_stt + overlay_size[1] / 2
        if col_end > target_shape[1]: col_end = target_shape[1]

        from_row_stt = 0
        if row > 0: from_row_stt = from_row_stt + overlay_size[0] / 2
        from_col_stt = 0
        if col > 0: from_col_stt = from_col_stt + overlay_size[1] / 2

        # sub_array = io.imread(os.path.join(sub_image_dir, sub_file), as_gary=False)
        sub_image_path = os.path.join(sub_image_dir, sub_file)
        if not os.path.exists(sub_image_path):
            print("Fail to find the sub image file {}".format(sub_image_path))
        sub_array = load_gdalimage(sub_image_path)

        target_img[int(row_stt):int(row_end), int(col_stt):int(col_end), :] = \
            sub_array[int(from_row_stt):, int(from_col_stt):, :]

    # io.imsave(target_path, target_img)
    saveas_gdalimage(target_img, target_path)

    end_time = time.time()
    print("### START at {} ###".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("### OVER at {} ###".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))))
    print("### Running time {} m ###".format((end_time - start_time)/60))

    return True

def copy_spatialref(img_path4, img_path2):
    image_ds4 = gdal.Open(img_path4, gdal.GA_ReadOnly)
    if not image_ds4:
        print("Fail to open image {}".format(img_path4))
        return False

    image_ds2 = gdal.Open(img_path2, gdal.GA_Update)
    if not image_ds2:
        print("Fail to open image {}".format(img_path2))
        return False

    proj = image_ds4.GetProjection()
    if proj:
        print("Projection is {}".format(proj))
    geotransform = image_ds4.GetGeoTransform()
    if geotransform:
        print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
        print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

    image_ds2.SetProjection(proj)
    image_ds2.SetGeoTransform(geotransform)

    print("### Copy spatial reference over")
    return True

# 创建shpline
def to_shp(img_path, shp_path):
    image_dir = sorted(os.listdir(img_path))
    for item in range(len(image_dir)):
        img = gdal.Open(img_path + '/' + image_dir[item], gdal.GA_ReadOnly)
        geotransform = img.GetGeoTransform()

        print(item)

        shp = os.path.join(shp_path + '/' + str(os.path.splitext(image_dir[item])[0]) + '.shp')   # 新建数据存放位置
        lineshp = shapefile.Writer(shp)

        # 创建字段
        lineshp.field('class')  # 'SECOND_FLD'为字段名称，C代表数据类型为字符串， 长度为40
        lineshp.line([[[geotransform[0], geotransform[3]], [geotransform[0] + 1000 * geotransform[1], geotransform[3]],
                    [geotransform[0] + 1000 * geotransform[1], geotransform[3] + 1000 * geotransform[5]],
                    [geotransform[0], geotransform[3] + 1000 * geotransform[5]], [geotransform[0], geotransform[3]]]])
        lineshp.record(0)
        # 写入数据
        lineshp.close()
        # 定义投影1
        proj = img.GetProjection()  # 同源投影
        print("Projection is {}".format(proj))
        # lineshp.SetProjection(proj)  # 写入投影  TIFF的写入投影方式
        

        # 定义投影2
        # proj = osr.SpatialReference()
        # proj.ImportFromEPSG(4326) # 4326-GCS_WGS_1984; 4490-GCS_China_Geodetic_Coordinate_System_2000
        # wkt = proj.ExportToWkt()
        f = open(shp.replace(".shp", ".prj"), 'w')
        f.write(proj)
        f.close()
    return True

def main():
    print("### img_util.main() ###########################################")
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Manual to this script')
    parser.add_argument('--datetime', type=str, default=None)
    args = parser.parse_args()
    date_time = args.datetime
    ####################################################################
    ### Split multi- images
    ####################################################################
    sub_size = (1000, 1000)
    overlay_size = (0, 0)
    # split_image(image_path, sub_size, overlay_size)
    image_dir = r'F:\Download\bigmap\南通市双甸镇'
    image_files = list(filter(lambda filename: os.path.splitext(filename)[1] == '.tif', os.listdir(image_dir)))
    for imgfile in image_files:
        image_path = os.path.join(image_dir, imgfile)
        split_image(image_path, sub_size, overlay_size)
        print("Spliting image {}.".format(image_path))
    print("### OVER ###")
    ####################################################################
    ### Merge sub images
    ###################################################################
    # sub_size = (400, 400)
    # overlay_size = (125, 125)
    # target_shape = (1500, 1500,1)
    # sub_image_dir = r'D:\小论文2\SENet建筑数据集应用\预测结果\23429020_15\splitSEGNet3'
    # target_path = r"D:\小论文2\SENet建筑数据集应用\预测结果\23429020_15\mergeSEGNet3\23429020_15.tif"
    # merge_image(sub_image_dir, sub_size, overlay_size, target_path, target_shape)
    # print('###' * 20)

    # ####################################################################
    # img_path4 = 'G:/experiments-dataset/hunan-lixian-crop2/000-sar/ls-char/S1A_IW_GRDH_1SDV_20170827.tif'
    # img_path2 = 'G:/experiments-dataset/hunan-lixian-crop2/000-sar/ls-char/S1A_IW_GRDH_1SDV_20170827/block2_conv2.tif'
    # copy_spatialref(img_path4, img_path2)

    ####################################################################
    end_time = time.time()
    print("### START at {} ###".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("### OVER at {} ###".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))))
    print("### Running time {} m ###".format((end_time - start_time)/60))
    print('###' * 20)


img_path = r'E:\plough_data\nantong\img'
shp_path =  r"E:\plough_data\nantong\lshp" # 新建数据存放位置
# file = shapefile.Writer(data_address)
# # 创建字段
# file.field('ID') # 'SECOND_FLD'为字段名称，C代表数据类型为字符串， 长度为40
# file.line([[[geotransform[0], geotransform[3]], [geotransform[0]+1000*geotransform[1], geotransform[3]], [geotransform[0]+1000*geotransform[1], geotransform[3]+1000*geotransform[5]], [geotransform[0], geotransform[3]+1000*geotransform[5]], [geotransform[0], geotransform[3]]]])
# file.record(0)
# # 写入数据
# file.close()
# # 定义投影
# proj = img.GetProjection()  # 同源投影
# print("Projection is {}".format(proj))
# # proj = osr.SpatialReference()
# # proj.ImportFromEPSG(4326) # 4326-GCS_WGS_1984; 4490- GCS_China_Geodetic_Coordinate_System_2000
# # wkt = proj.ExportToWkt()
# # print(proj)
# # data_address.SetProjection(proj)
# # data_address.SetGeoTransform(geotransform)
# # 写入投影
# f = open(data_address.replace(".shp", ".prj"), 'w')
# f.write(proj)
# f.close()

to_shp(img_path, shp_path)

####################################################################
# if __name__ == "__main__":
    # main()

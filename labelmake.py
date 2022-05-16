import gdal
import gdalconst
import ogr
import os

def tifdatamaker(path1,path2,path3):
    data = gdal.Open(path1, gdalconst.GA_ReadOnly)
    geo_transform = data.GetGeoTransform()
    x_min = geo_transform[0]
    y_min = geo_transform[3]
    x_res = data.RasterXSize
    y_res = data.RasterYSize
    mb_v = ogr.Open(path2)
    mb_l = mb_v.GetLayer()
    pixel_width = geo_transform[1]
    target_ds = gdal.GetDriverByName('GTiff').Create(path3, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, -1*pixel_width))
    band = target_ds.GetRasterBand(1)
    NoData_value = -999
    band.SetNoDataValue(NoData_value)#黑边处理
    band.FlushCache()#数据写入磁盘
    gdal.RasterizeLayer(target_ds, [1], mb_l, options= ["ATTRIBUTE=ID"])
    target_ds = None

imgpath=r'E:\plough_data\CS70\img'
shppath=r'E:\plough_data\CS70\lshp'
savepath=r'E:\plough_data\CS70\linetiff'
namelist= r'E:\plough_data\CS70\CS70.lst'

if os.path.exists(savepath) != True:
    os.makedirs(savepath)

def nameget(list):
    with open(list, 'r') as f:
        filelist = f.readlines()
        for i in range(len(filelist)):
            filelist[i] = filelist[i].split('\n')[0]
    return (filelist)

name = nameget(namelist)
shpf=[]
images = sorted(os.listdir(imgpath))
shpfiles = sorted(os.listdir(shppath))

for i in range(len(shpfiles)):
    if shpfiles[i].split('.')[-1]=='shp':
        shpf.append(shpfiles[i])
for j in range(len(shpf)):
    imagepathfile = os.path.join(imgpath, images[j])
    shppathfile=os.path.join(shppath,shpf[j])
    savepathfile=os.path.join(savepath,'{}'.format(name[j]))
    tifdatamaker(imagepathfile,shppathfile,savepathfile)

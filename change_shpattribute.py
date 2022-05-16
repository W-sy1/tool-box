# # -*- coding: utf-8 -*-
# import arcpy
# #文件夹路径
# arcpy.env.workspace = r'E:\plough_data\NTdata\lshp'
# shpfiles = arcpy.ListFeatureClasses()
#
# for shp in shpfiles:
#     try:
#         inFeatures = shp
#         #print shp
#         #新增字段
#         fieldName = "calss1"
#         arcpy.AddField_management(inFeatures, fieldName,'text')
#         arcpy.CalculateField_management(inFeatures,fieldName, '"'+shp[0:-4]+'"',"PYTHON_9.3")
#     except arcpy.ExecuteError:
#         print(arcpy.GetMessages())


import ogr

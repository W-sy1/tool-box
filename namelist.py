import fnmatch
import os
import pandas as pd
import numpy as np
import sys

InputStra =r'E:\plough_data\CS70\dataset_cs\test\image'
InputStrb = '*.tif'

def ReadSaveAddr(Stra,Strb):
    a_list = fnmatch.filter(os.listdir(Stra),Strb)
    for i in range(len(a_list)):
        a_list[i] = a_list[i]
    print("Find = ",len(a_list))
    df = pd.DataFrame(np.arange(len(a_list)).reshape((len(a_list),1)),columns=['Addr'])
    df.Addr = a_list
    #print(df.head())
    df.to_csv(r'E:\plough_data\CS70\dataset_cs\test\\test_CS70.lst',columns=['Addr'],index=False,header=False)
    print("Write To Get.lst !")



ReadSaveAddr(InputStra,InputStrb)
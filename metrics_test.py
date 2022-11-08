# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import joblib
from skimage.io import imread



from input_dataset import mydata,Dataset2

from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses
from utils import str2bool, count_params
import pandas as pd
from vnet import VNet
import csv
import xlwt
import xlrd



book = xlwt.Workbook()
sheet = book.add_sheet('tumor_precision',cell_overwrite_ok=True)
col = ('file_name','dice')
for i in range(0,2):
        sheet.write(0,i,col[i])




workbook = xlrd.open_workbook('/home/pubsys/jinqianlong/Program/PyCharm/Chen/kidney/kits19/code_lhx/tumor_precision_train.xls')
sheet1_name = workbook.sheet_names()[0]
print(sheet1_name)

sheet1 = workbook.sheet_by_name(sheet1_name)
print(sheet1.name,sheet1.nrows,sheet1.ncols)

# 获得一行
rows = sheet1.row_values(0)
#print(rows)
#print(len(rows)) # 539
# 获得一列
cols = sheet1.col_values(0)
dice_score=sheet1.col_values(1)
#print(cols,dice_score)
#print(len(cols))# 3066

file_name=cols[1:]
#file_name=np.array(file_name)
#sub_name=file_name.split('_')[1]
#print(file_name)



sub=[]
for name in (file_name):
    sub_name=name.split('_')[1]
    #print(sub_name)
    sub.append(sub_name)
#print(sub)
sub=list(set(sub))


list=0
for i in (sub):

    dice_sum=0
    sheet.write(list+1,0,i)
    num = 0
    for j in (file_name):
        position = file_name.index(j)
        dice = dice_score[position + 1]
        dice_sub_name=j.split('_')[1]

        if i==dice_sub_name:
            #print('case_name:',i,dice_sub_name,'true')
            num=num+1
            dice_sum=(dice_sum+dice)
            #print(num)
        #dice_avg=dice_sum/num
        sheet.write(list+1,1,dice_sum)
        sheet.write(list+1,2,num)
    list=list+1
savepath = '/home/pubsys/jinqianlong/Program/PyCharm/Chen/kidney/kits19/code_lhx/tumor_sum_train.xls'
book.save(savepath)

















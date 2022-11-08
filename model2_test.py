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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from model2_input import Dataset_model2,Dataset_model2_test
from input_dataset import mydata,Dataset

from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses
from utils import str2bool, count_params
import pandas as pd
from vnet import VNet
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import xlwt
book = xlwt.Workbook()
sheet = book.add_sheet('tumor_precision',cell_overwrite_ok=True)
col = ('file_name','dice')
for i in range(0,2):
        sheet.write(0,i,col[i])

def dice(output,target):
    smooth=1e-5
    if torch.is_tensor(output):
        output=torch.sigmoid(output).view(-1).data.cpu().numpy()
    if torch.is_tensor(target):
        target=target.view(-1).data.cpu().numpy()
    #output_ = torch.sigmoid(output).view(-1).data.cpu().numpy()
    #target_ = target.view(-1).data.cpu().numpy()
    output_=output>0.5
    target_=target>0.5
    intersection=(output_*target_).sum()
    return (2*intersection+smooth)/(output_.sum()+target_.sum()+smooth)

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='kindey_DeepResUNet256-128_100_model2_woDS',
                        help='model name')
    parser.add_argument('--mode', default=None,
                        help='')

    args = parser.parse_args()

    return args




if __name__ == '__main__':
    val_args = parse_args()

    args = joblib.load('/home/pubsys/jinqianlong/Program/PyCharm/Chen/kidney/kits19/code_lhx/models'
                       '/%s/args.pkl' %val_args.name)
    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # create model
    print("=> creating model %s" %args.arch)
    model = VNet(1)
    ################
    model=model.cuda()


    img_paths = glob('/home/pubsys/jinqianlong/Program/PyCharm/Chen/kidney/kits19/code_lhx/'
                     'process2_tumor_img_train/*')
    mask_paths = glob('/home/pubsys/jinqianlong/Program/PyCharm/Chen/kidney/kits19/code_lhx/'
                      'process2_tumor_label_train/*')

    #print('lll',img_paths)
    val_img_paths = img_paths
    val_mask_paths = mask_paths


    #train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
    #   train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('/home/pubsys/jinqianlong/Program/PyCharm/Chen/kidney/kits19/code_lhx/'
                                     'models/%s/model.pth' %args.name))
    model.eval()

    val_dataset = Dataset_model2_test(args, val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    slice = 0
    metrics = 0
    for mynum, (input, target, file) in tqdm(enumerate(val_loader), total=len(val_loader)):

        # print('ooo',mynum)
        input = input.float()
        input = input.cuda()
        output = model(input)
        input = torch.squeeze(input)
        file_path = file
        # print('mmm',file_path)

        output = torch.sigmoid(output)
        output = torch.squeeze(output)
        output = output.data.cpu().numpy()

        target = torch.squeeze(target).data.cpu().numpy()

        print('zzz', output.shape, target.shape)

        for i in range(len(file)):
            file_path = file[i]
            file_name = file_path.split('/')[-1]
            out = output[i, :, :, :]
            mask = target[i, :, :, :]

            dice_score = dice(out, mask)
            metrics = metrics + dice_score
            slice = slice + 1

            sheet.write(slice, 0, file_name)
            sheet.write(slice, 1, dice_score)

            #print('nnn',i,len(output),file_name,mask.shape)
            '''
            #for j in range(50):
            plt.subplot(131)
            plt.imshow(input[i, 8, :, :].cpu().numpy(),cmap='gray')
            plt.subplot(132)
            plt.imshow(target[i,8,:,:])
            plt.subplot(133)
            plt.imshow(mask[8,:,:])
            plt.show()
            '''


            pre_save_file = './prediction_tumor_train'
            #print('aaa',mask.shape)
            if not os.path.isdir(pre_save_file):
                os.makedirs(pre_save_file)
            pre_kidney_save = sitk.GetImageFromArray(mask)
            sitk.WriteImage(pre_kidney_save, pre_save_file + '/' + file_name + '.nii.gz')



        savepath = '/home/pubsys/jinqianlong/Program/PyCharm/Chen/kidney/kits19/code_lhx/tumor_precision_train.xls'
        book.save(savepath)
    metrics=metrics/len(img_paths)
    print('metrics:',metrics)







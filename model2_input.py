import torch.utils.data
import numpy as np
import os, random, glob
from skimage import transform
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk
import skimage.io as io

##########################################
######## 该程序导入用于训练、测试等的输入图像、标签
##########################################

class Dataset_model2(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug

    def __len__(self):
        #print(len(self.img_paths))
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        npimage = nib.load(img_path)
        npmask = nib.load(mask_path)


        npimage = npimage.get_fdata()
        npmask = npmask.get_fdata()



        img_split = torch.tensor(npimage)
        label_split = torch.tensor(npmask)
        #print(label_split.shape)

        ### 当通道为1时，增加一个通道轴
        #img_split = img_split.permute(1, 0, 2)
        #label_split = label_split.permute(1, 0, 2)
        img_split = torch.unsqueeze(img_split, axis=0)
        label_split = torch.unsqueeze(label_split, axis=0)
        #print('lll',img_split.shape, label_split.shape)


        return img_split,label_split#,img_path






class Dataset_model2_test(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug

    def __len__(self):
        #print(len(self.img_paths))
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        npimage = nib.load(img_path)
        npmask = nib.load(mask_path)


        npimage = npimage.get_fdata()
        npmask = npmask.get_fdata()



        img_split = torch.tensor(npimage)
        label_split = torch.tensor(npmask)
        #print(label_split.shape)

        ### 当通道为1时，增加一个通道轴
        #img_split = img_split.permute(1, 0, 2)
        #label_split = label_split.permute(1, 0, 2)
        img_split = torch.unsqueeze(img_split, axis=0)
        label_split = torch.unsqueeze(label_split, axis=0)
        #print('lll',img_split.shape, label_split.shape)


        return img_split,label_split,img_path

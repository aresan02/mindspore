import torch.utils.data
import numpy as np
import os, random, glob
from skimage import transform
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk
import skimage.io as io





###############################################
####### 该程序用于转换初始图像为25张一组，并按文件名保存
###############################################

def mydata(img_path,mask_path):
    img_file = os.listdir(img_path)####文件子目录
    for i in img_file:
        image_id = i

        image_path = os.path.join(img_path, i)
        label_path = os.path.join(mask_path, i)
        print('ooo',label_path)

        #### image
        image = nib.load(image_path)
        image = image.get_fdata()

        #### label
        label = nib.load(label_path)
        label = label.get_fdata()

        ### 把标签中有肿瘤的序列挑出来
        gray_pixel = np.unique(label)
        if len(gray_pixel) > 1:
            img_save_file='./tumor_train_img_val'
            if not os.path.isdir(img_save_file):
                os.makedirs(img_save_file)
            image=image.transpose(2,1,0)
            img_save=sitk.GetImageFromArray(image)
            sitk.WriteImage(img_save, img_save_file+'/'+image_id)

            '''
            label_save_file='./tumor_train_label'
            if not os.path.isdir(label_save_file):
                os.makedirs(label_save_file)
            label_save=sitk.GetImageFromArray(label)
            sitk.WriteImage(label_save, label_save_file+'/'+image_id)
            '''

    return 0


if __name__ == "__main__":
    img_path = '/home/pubsys/jinqianlong/Program/PyCharm/Chen/kidney/kits19/code_lhx/process_img_val_512'
    label_path='/home/pubsys/jinqianlong/Program/PyCharm/Chen/kidney/kits19/code_lhx/process_tumor_val_512'
    dataset = mydata(img_path,label_path)
import torch.utils.data
import numpy as np
import os, random, glob
from skimage import transform
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk
import skimage.io as io




w_width = 400
w_center = 40
#data_adjusted1 = adjustMethod1(data_resampled,w_width,w_center)

def adjustMethod1(data_resampled,w_width,w_center):
    val_min = w_center - (w_width / 2)
    val_max = w_center + (w_width / 2)

    data_adjusted = data_resampled.copy()
    data_adjusted[data_resampled < val_min] = val_min
    data_adjusted[data_resampled > val_max] = val_max

    return data_adjusted
###############################################
####### 该程序用于转换初始图像为25张一组，并按文件名保存
###############################################

def mydata(file_path):
    img_file = os.listdir(file_path)
    img_split = []
    label_split = []
    tumor_split=[]
    split_image_id = []
    #a=0

    for i in img_file:
        image_id = i
        print('name:',image_id)
        image_path = os.path.join(file_path, i + '/imaging.nii.gz')
        label_path = os.path.join(file_path, i + '/segmentation.nii.gz')

        image = nib.load(image_path)
        '''
        pixdim = image.header['pixdim']
        print(f'z轴分辨率： {pixdim[3]}')
        print(f'in plane 分辨率： {pixdim[1]} * {pixdim[2]}')
        z_range = pixdim[3] * image.shape[0]
        x_range = pixdim[1] * image.shape[1]
        y_range = pixdim[2] * image.shape[2]
        print(i,image.shape,x_range, y_range, z_range)
        '''

        image = image.get_fdata()
        image=adjustMethod1(image,w_width, w_center)
        #### 图像归一化
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        #### label无需归一化
        label = nib.load(label_path)
        label = label.get_fdata()

        #### 缩小图像
        image = transform.resize(image, (image.shape[0], 512, 512))
        label = transform.resize(label, (label.shape[0], 512, 512))


        #### 查看原图灰度值
        '''
        plt.subplot(121)
        plt.imshow(image[99,:,:],cmap='gray')
        plt.subplot(122)
        plt.imshow(label[99,:,:])
        plt.show()
        '''

        ### 把标签中肿瘤和肾脏合二为一
        gray_pixel = np.unique(label)
        tumor=np.zeros_like(label)
        tumor[label==2]=1
        if len(gray_pixel) > 2:
            label[label > 0] = 1

        ### 找到标签中肾脏起始、终止的切片位置
        z = np.any(label, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]
        valid_slice = end_slice - start_slice + 1



        # 为数据25张一组作准备，补全矩阵
        if valid_slice % 25 == 0:
            start_slice = start_slice
            end_slice = end_slice+1
        else:

            insert_all = valid_slice - (valid_slice // 25) * 25
            if insert_all >= 10:

                insert = 25 - insert_all + 1

                if insert % 2 == 0:

                    start_slice = int(start_slice - insert / 2)
                    end_slice = int(end_slice + insert / 2)

                else:

                    start_slice = int(start_slice - (insert // 2) * 2)
                    end_slice = int(end_slice + insert - ((insert // 2) * 2))

                if start_slice < 0:

                    start_slice = 0
                    end_slice = (valid_slice // 25 + 1) * 25
                elif end_slice > label.shape[0]:

                    start_slice = label.shape[0] - (valid_slice // 25 + 1) * 25
                    end_slice = label.shape[0]
                else:

                    start_slice = start_slice
                    end_slice = end_slice
            else:

                start_slice = start_slice + (insert_all // 2) * 2 - 1
                end_slice = end_slice - (insert_all - ((insert_all // 2) * 2))

        img_new = image[start_slice:end_slice, :, :]
        label_new = label[start_slice:end_slice, :, :]
        tumor_new=tumor[start_slice:end_slice, :, :]
        #print(start_slice,end_slice)
        '''
        sq=(label_new.shape[0]) // 25
        a=sq+a
        print('ddd',i,a)
        '''



        ##### channel=25 开始聚合成新的数据集
        
        for i in range((label_new.shape[0]) // 25):
            img_copy = img_new[i * 25:(i * 25 + 25), :, :]
            img_split.append(img_copy)
            label_copy = label_new[i * 25:(i * 25 + 25), :, :]
            label_split.append(label_copy)
            tumor_copy = tumor_new[i * 25:(i * 25 + 25), :, :]
            tumor_split.append(tumor_copy)

            #### 每个样本分块数据标号，eg: case_00146_5
            image_name = image_id + '_' + str(i)
            split_image_id.append(image_name)


            ###### 保存分块的图像
            img_save_file='./process_img_train_512'
            if not os.path.isdir(img_save_file):
                os.makedirs(img_save_file)
            img_save=sitk.GetImageFromArray(img_copy)
            sitk.WriteImage(img_save, img_save_file+'/'+image_name+'.nii.gz')

            label_save_file='./process_label_train_512'
            if not os.path.isdir(label_save_file):
                os.makedirs(label_save_file)
            label_save=sitk.GetImageFromArray(label_copy)
            sitk.WriteImage(label_save, label_save_file+'/'+image_name+'.nii.gz')

            tumor_save_file = './process_tumor_train_512'
            if not os.path.isdir(tumor_save_file):
                os.makedirs(tumor_save_file)
            tumor_save = sitk.GetImageFromArray(tumor_copy)
            sitk.WriteImage(tumor_save, tumor_save_file+'/'+image_name+'.nii.gz')


    img_split = np.array(img_split)
    label_split = np.array(label_split)
    tumor_split = np.array(tumor_split)


    ### numpy to tensor
    img_split=torch.tensor(img_split)
    label_split=torch.tensor(label_split)
    tumor_split = torch.tensor(tumor_split)


    ### 当通道为1时，增加一个通道轴
    img_split=torch.unsqueeze(img_split, axis=-1)
    label_split=torch.unsqueeze(label_split, axis=-1)
    tumor_split = torch.unsqueeze(tumor_split, axis=-1)
    print(img_split.shape)


    return img_split, label_split, tumor_split






if __name__ == "__main__":

    #file_path = '/Users/lihaixing/Documents/lhx/肾脏肿瘤比赛/kits19-master/kits19/data'
    file_path='../kits19/train'
    dataset = mydata(file_path)
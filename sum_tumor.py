import numpy as np
from glob import glob
from skimage import transform
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import os
from skimage import transform
import SimpleITK as sitk
import xlrd

# import skimage.io as io
book=xlrd.open_workbook('/Users/lihaixing/Documents/lhx/肾脏肿瘤比赛/kits19-master/insert_case.xls')
table=book.sheets()[0]
kidney_case=table.col_values(0)[1:]
#print(case_num.index('case_00179'))
insert_slice=table.col_values(1)[1:]
#print(case_num)


###############################################
####### 该程序用于转换初始图像为25张一组，并按文件名保存
###############################################

def mydata(img_path):
    sd = 0
    img_file = os.listdir(img_path)  ####文件子目录
    a = []
    ###### 找到case序号
    for i in img_file:
        name = i.split('.')[0]
        num = name.split('_')[1]
        a.append(num)
    sum_name = list(set(a))

    ###### 聚合序列
    for case_num in sum_name:
        sd = sd + 1
        img= []
        n = 0
        for i in img_file:
            #### 序列号
            name = i.split('.')[0]
            num = name.split('_')[1]

            if num == case_num:
                n = n + 1
        ##### 聚合
        img=np.zeros([512,512,0])
        for i in range(n):
            path = os.path.join(img_path, 'case_' + case_num + '_' + str(i) + '.nii.gz')
            ###### 读图
            image = nib.load(path)
            image = image.get_fdata()
            image = np.array(image)
            img=np.concatenate([img,image],axis=-1)
            #img.append(image)

        case = np.array(img)
        case_name='case_'+case_num
        if case_name in kidney_case:
            cite=kidney_case.index(case_name)
            insert_case_slice=int(insert_slice[cite])
            case=case[:,:,0:case.shape[2]-insert_case_slice]
            case=transform.resize(case,(case.shape[0],512,512))####
        img_save_file = './prediction_whole_tumor'
        if not os.path.isdir(img_save_file):
            os.makedirs(img_save_file)
        img_save = sitk.GetImageFromArray(case)
        sitk.WriteImage(img_save, img_save_file + '/' + case_name+'.nii.gz')
        print('sss', n, case_num, case.shape)
    return 0


if __name__ == "__main__":
    prediction_path = '/Users/lihaixing/Downloads/van_s/mindspore/models/model2_image_test_512'
    #'/home/ma-user/work/code_lhx_data/code_lhx/prediction_kidney'
    dataset = mydata(prediction_path)
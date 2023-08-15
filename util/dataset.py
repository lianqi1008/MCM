import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CreateDatasetFromImages(Dataset):
    def __init__(self, is_train, dataset_path):
        if is_train:
            self.file_path = os.path.join(dataset_path, 'images/train')
            csv_t = os.path.join(dataset_path, 'scores/train/texture.csv')
            csv_s = os.path.join(dataset_path, 'scores/train/structure.csv')
        else:
            # self.file_path = os.path.join(file_path, 'images/test')
            self.file_path = os.path.join(dataset_path, 'images/test')
            csv_t = os.path.join(dataset_path, 'scores/test/texture.csv')
            csv_s = os.path.join(dataset_path, 'scores/test/structure.csv')

        # self.to_tensor = transforms.ToTensor() #将数据转换成tensor形
        # 读取 csv 文件
        #利用pandas读取csv文件
        self.data_info_t = pd.read_csv(csv_t, sep=',', encoding="utf-8")  #header=None是去掉表头部分
        self.data_info_s = pd.read_csv(csv_s, sep=',', encoding="utf-8")  #header=None是去掉表头部分
        # 文件第一列包含图像文件的名称
        self.image_arr = self.data_info_t.iloc[:, 0].values.tolist()  #self.data_info.iloc[1:,0表示读取第一列，从第二行开始一直读取到最后一行
        self.label_t = self.data_info_t.iloc[:, 1:].values.tolist()
        self.label_s = self.data_info_s.iloc[:, 1:].values.tolist()

        # 计算 length
        self.data_len = len(self.image_arr)

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(os.path.join(self.file_path, single_image_name))

        #如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        # if img_as_img.mode != 'L':
        #     img_as_img = img_as_img.convert('L')

        #设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        img_as_img = transform(img_as_img)

        # 得到图像的 label
        label_t = torch.Tensor(self.label_t[index])
        label_s = torch.Tensor(self.label_s[index])

        return (img_as_img, label_t, label_s)  #返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.data_len
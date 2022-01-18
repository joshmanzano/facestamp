
import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from deepface import DeepFace
import sys, inspect, functools
import io
import pickle
import random
from PIL import ImageDraw
import utils

class Data(Dataset):
    def __init__(self, partition, secret_size=14, size=(400, 400), dataset_size=None):
        self.data_path = './train_data'
        self.secret_size = secret_size
        self.size = size
        self.data_list = pickle.load(open(f'{self.data_path}/partition.pkl','rb'))[partition]
        # if(partition == 'test'):
            # self.data_list = self.data_list[:int(len(self.data_list) * 0.2)]
            # self.data_list = self.data_list[:3]
        self.analysis_data = pickle.load(open(f'{self.data_path}/analysis_data.pkl', 'rb'))
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        idx = self.data_list[idx]
        img = Image.open(f'{self.data_path}/images/{idx}.png').resize(self.size)
        img = self.to_tensor(img)
        mask = Image.open(f'{self.data_path}/masks/{idx}.png').resize(self.size)
        mask = self.to_tensor(mask)
        # face_mask = Image.open(f'{self.data_path}/face_masks/{idx}.png').resize(self.size)
        # face_mask = self.to_tensor(face_mask)
        analysis = self.analysis_data[idx]
        # img_cover = np.array(img_cover, dtype=np.float32) / 255.
        secret = utils.convert_secret(analysis[0])
        secret = torch.Tensor(secret)
        region = analysis[1] 
        # secret = torch.round(torch.rand(self.secret_size))
        # print(len(secret))
        return img, mask, secret, region, str(f'{self.data_path}/images/{idx}.png')

    def __len__(self):
        return len(self.data_list)
        
if __name__ == '__main__':
    # dataset = Data('train', secret_size=14, size=(400, 400))
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    # print(len(dataset))
    # dataset = Data('eval', secret_size=14, size=(400, 400))
    # print(len(dataset))
    # dataset = Data('test', secret_size=14, size=(400, 400))
    # print(len(dataset))
    data_list = pickle.load(open('./train_data/partition.pkl','rb'))['test']
    analysis_data = pickle.load(open('./train_data/analysis_data.pkl', 'rb'))
    random.shuffle(data_list)
    analysis = analysis_data[data_list[0]]
    filename = str(f'./train_data/images/{data_list[0]}.png')
    print(utils.convert_secret(analysis[0]))
    print(filename)
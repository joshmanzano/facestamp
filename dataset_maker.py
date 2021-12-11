
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
import lpips
import zlib
import json
from tqdm import tqdm
import random
import utils
import pathlib
import pickle

# def create_dataset(partition, code):
#     data_path='./data/celebamaskhq/images'
#     eval_list_path='./data/celeba/list_eval_partition.txt'

#     attr_list = open('./data/celeba/list_attr_celeba.txt', 'r').read().splitlines()

#     headers = attr_list[1]
#     index_list = {}
#     for row in attr_list[2:]:
#         index = row.split('.jpg')[0]
#         index_list[index] = row

#     new_attr_list = [None, headers]

#     eval_list_lines = open(eval_list_path, 'r').read().splitlines()
#     eval_list = {}
#     for line in eval_list_lines:
#         line_data = line.split(' ')
#         eval_list[line_data[0]] = line_data[1]
    
#     files_list = []
#     for file in glob(os.path.join(data_path, '**/*.jpg'), recursive=True):
#         split_file = file.split('/')
#         name = split_file[len(split_file) - 1]
#         if(eval_list[name] == code):
#             files_list.append(file)

#     random.shuffle(files_list)

#     count = 0

#     analysisList = [] 

#     output_dir = partition
#     amount = float('inf')

#     for file in tqdm(files_list):
#         try:
#             img_cover_path = file
#             img_cover = Image.open(img_cover_path).convert('RGB').resize((128,128))
#             split_file = file.split('/')
#             name = split_file[len(split_file) - 1]
#             new_img_name = f'./data/{output_dir}/images/{name}'
#             img_cover.save(new_img_name)
#             secret, region = utils.analyzeFace(new_img_name)
#             analysisList.append((file, secret, region))
#             index = name.split('.jpg')[0]
#             new_attr_list.append(index_list[index])
#         except Exception as e:
#             os.remove(new_img_name)
#             continue
#         if(count == amount):
#             break
#         count += 1
    
#     new_attr_list[0] = str(count)
    
#     json.dump(analysisList, open(f'./data/{output_dir}/analysis.json','w'))
#     with open(f'./data/{output_dir}/list_attr_celeba.txt', 'w') as f:
#         for line in new_attr_list:
#             f.write(line.strip() + '\n')

def process_images(data):
    output = './data/dataset'
    files = glob('./data/celebamaskhq/CelebA-HQ-img/*')
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((400))
    to_pil = transforms.ToPILImage()
    for file in files:
        path = pathlib.Path(file)
        idx = int(path.stem)
        mask_path = data[idx][0]
        image = Image.open(file)
        image = to_tensor(image)
        image = resize(image)
        raw_mask = Image.open(mask_path)
        raw_mask = to_tensor(raw_mask)
        raw_mask = resize(raw_mask)
        mask = torch.ones(raw_mask.shape)
        mask -= raw_mask
        (to_pil(image)).save(f'{output}/images/{idx}.png')
        (to_pil(mask)).save(f'{output}/masks/{idx}.png')
    return data

def process_masks():
    files = glob('./data/celebamaskhq/CelebAMask-HQ-mask-anno/**/*')
    dataset = {}
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    for file in files:
        path = pathlib.Path(file)
        stem = path.stem
        data = stem.split('_')
        idx = int(data[0])
        if(data[1] == 'skin'):
            dataset[idx] = [file]
    return dataset

def process_data():
    images = glob('./data/dataset/images/*')
    random.shuffle(images)
    mask_path = './data/dataset/masks'
    face_mask_path = './data/dataset/face_masks'
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    analysis_data = {}
    for image in tqdm(images):
        image_input = to_tensor(Image.open(image)).cuda()[None]
        idx = int(pathlib.Path(image).stem)
        try:
            analyzed, region = utils.analyzeFace(image)
        except:
            os.remove(image)
            os.remove(f'{mask_path}/{idx}.png')
            continue
        secret = utils.convert_secret(analyzed)
        mask = utils.create_mask_input(image_input, region, region_transform=False)
        (to_pil(mask.squeeze())).save(f'{face_mask_path}/{idx}.png')
        analysis_data[idx] = (analyzed, region)
    with open('./data/dataset/analysis_data.pkl','wb') as f:
        pickle.dump(analysis_data, f)

def partition():
    partitions = {
        'train': [],
        'eval': [],
        'test': [],
    }
    images = glob('./data/dataset/images/*')
    random.shuffle(images)
    count = 0
    for image in tqdm(images):
        idx = int(pathlib.Path(image).stem)
        if(count < len(images) * 0.7):
            partitions['train'].append(idx)
        elif(count < len(images) * 0.9):
            partitions['eval'].append(idx)
        else:
            partitions['test'].append(idx)
        count += 1
    pickle.dump(partitions, open('./data/dataset/partition.pkl','wb'))

if __name__ == '__main__':
    # dataset = process_masks()
    # process_images(dataset)
    # process_data()
    partition()

from re import match
import subprocess
from glob import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from dataset import Data
from torch.utils.data import DataLoader
import utils
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
import time
import pickle
import json
import pathlib

base_path = './test_data/vidtimit/process'

df_models = json.load(open('df_models.json','r')) 

for df_model in df_models:
    input_path = base_path + df_model['source']
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    resized_path = base_path + df_model['source'] + '-resized'
    if not os.path.exists(resized_path):
        os.makedirs(resized_path)

    output_path = base_path + df_model['source'] + '-enc'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    residual_path = base_path + df_model['source'] + '-residual'
    if not os.path.exists(residual_path):
        os.makedirs(residual_path)

    swap_path = base_path + df_model['source'] + '-enc-swap'
    if not os.path.exists(swap_path):
        os.makedirs(swap_path)

    df_model['input_path'] = input_path
    df_model['resized_path'] = resized_path
    df_model['residual_path'] = residual_path
    df_model['output_path'] = output_path
    df_model['swap_path'] = swap_path
    df_model['example'] = None

size = 400

def delete_extra_dir():
    dirs = glob('./test_data/vidtimit/process/*')
    for dir in tqdm(dirs):
        images = glob(dir + '/*')
        for image in images:
            path = pathlib.Path(image)
            stem = path.stem
            if(stem != 'alignments' and stem != 'analysis' and image[-4:] != '.png' and image[-4:] != '.jpg'):
                for i in glob(image + '/*'):
                    os.remove(i)
                os.rmdir(image)

def prepare_resized(input_path, resized_path):
    images = glob(f'{input_path}/*.jpg')
    for step, image in enumerate(tqdm(images)):
        path = Path(image)
        full_stem = path.stem + '.png'

        image_input = transforms.ToTensor()(Image.open(image))
        image_input = transforms.CenterCrop(min(image_input.shape[1],image_input.shape[2]))(image_input)
        image_input = transforms.Resize((size,size))(image_input)
        # img_name = utils.save_image(transforms.ToPILImage()(image_input))
        img_path = f'{resized_path}/{full_stem}'
        (transforms.ToPILImage()(image_input)).save(img_path)

if __name__ == '__main__':
    identities = glob('./test_data/vidtimit/process/*-resized')
    print(identities)
    # for identity in identities:
    #     path = pathlib.Path(identity)
    #     full_stem = path.stem
    #     resized_path = './test_data/vidtimit/process/' + full_stem + '-resized'
    #     print(full_stem, resized_path)
    #     if not os.path.exists(resized_path):
    #         os.makedirs(resized_path)
    #         prepare_resized(identity, resized_path)
    delete_extra_dir()

    for identity in identities:
        path = pathlib.Path(identity)
        resized_path = identity
        full_stem = path.stem
        print(path)
        print(full_stem, resized_path)
        if not os.path.exists(resized_path + '/analysis.bin'):
            analysis = {}
            images = glob(f'{resized_path}/*.png')
            for step, image in enumerate(tqdm(images)):
                path = Path(image)
                full_stem = path.stem + '.png'
                try:
                    secret_input, region_input = utils.get_secret_string(image)
                except:
                    continue
                analysis[image] = (secret_input, region_input)
            print(image, (secret_input, region_input))
            pickle.dump(analysis,open(f'{resized_path}/analysis.bin','wb'))
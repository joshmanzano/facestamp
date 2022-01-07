import subprocess 
from glob import glob
from PIL import Image, ImageOps
import numpy as np
import os
import pathlib
import torch
from torch import nn
import ast
from tqdm import tqdm
import pickle
import random
import sys
import json
from steganogan.models import SteganoGAN
import random
from numpy import dot
from numpy.linalg import norm
import argparse


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

def clear_previous():
    for df_model in df_models:
        output_path = df_model['output_path'] 
        swap_path = df_model['swap_path']
        residual_path = df_model['residual_path']
        files = glob(f'{output_path}/*') + glob(f'{swap_path}/*') + glob(f'{residual_path}/*')
        for file in files:
            os.remove(file)

def compile_examples(example):
    output = Image.new('RGB', (size*3, size*len(df_models)), 'white')
    for idx, df_model in enumerate(df_models):
        img_path = pathlib.Path(glob(df_model['output_path'] + '/' + '*.png')[0])
        full_stem = img_path.stem + img_path.suffix
        clean_path = df_model['resized_path'] + '/' + full_stem 
        encoded_path = df_model['output_path'] + '/' + full_stem
        swapped_path = df_model['swap_path'] + '/' + full_stem
        # residual_path = df_model['residual_path'] + '/' + full_stem

        clean = Image.open(clean_path)
        encoded = Image.open(encoded_path)
        swapped = Image.open(swapped_path)
        # residual = Image.open(residual_path)

        output.paste(clean, (0, idx*size))
        # output.paste(residual, (size, idx*size))
        output.paste(encoded, (size, idx*size))
        output.paste(swapped, (size*2, idx*size))

    output.save(f'./results/stegastamp-{example}.png')

def bitstring_to_bytes(s):
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')

def secret_to_string(secret):
    secret = ''.join(secret)
    byte_string = bitstring_to_bytes(secret)
    print(byte_string)
    print(byte_string.decode('utf-8'))

def get_steganogan(architecture):

    steganogan_kwargs = {
        'cuda': True,
        'verbose': False,
        # 'depth': 4,
    }

    steganogan_kwargs['path'] = f'./steganogan/pretrained/{architecture}.steg'
    # steganogan_kwargs['architecture'] = 'dense'

    return SteganoGAN.load(**steganogan_kwargs)

def channel_code(mode, secret):
    secret_string = ''.join([str(s) for s in secret])
    encoded = subprocess.check_output(f'env/bin/python3 channel_coder.py {mode} {secret_string}',shell=True)
    encoded = encoded.splitlines()[-1].decode('utf8')
    encoded_array = []
    for i in encoded:
        encoded_array.append(int(i))
    return encoded_array
    # secret = ast.literal_eval(secret)

def encode(preswap, input_dir, output_dir, architecture):
    """Given loads a pretrained pickle, encodes the image with it."""
    similarities = []
    steganogan = get_steganogan(architecture)
    images = glob(f'{input_dir}/*.png')
    random.shuffle(images)
    images = images[:25]
    analysis = pickle.load(open(f'{input_dir}/analysis.bin','rb'))
    pbar = tqdm(images)
    for image in pbar:
        path = pathlib.Path(image)
        full_stem = path.stem + '.png'
        try:
            gt = analysis[image][0]
        except KeyError as e:
            continue
        encoded = channel_code('encode',gt)
        steganogan.encode(image, f'{output_dir}/{full_stem}', encoded)
        undecoded = steganogan.decode(f'{output_dir}/{full_stem}')
        undecoded = [int(i) for i in undecoded]
        decoded = channel_code('decode',undecoded)

        try:
            analyzed = subprocess.check_output(f'env/bin/python3 analyze_face.py {output_dir}/{full_stem}',shell=True)
            analyzed = analyzed.splitlines()[-1].decode('utf8')
            analyzed = ast.literal_eval(analyzed)
        except Exception as e:
            print(e)
            continue

        try:
            score = dot(analyzed, decoded)/(norm(analyzed)*norm(decoded))
        except:
            continue
        # if np.isnan(cos_sim):
        #     print(gt, dec)
        #     print(cos_sim)
        #     breakpoint()
            # gt_secrets = torch.T
        preswap.append(score)
        pbar.set_description(f'score: {np.mean(preswap)}')

    return preswap 

def decode(swapped, input_dir, architecture):
    """Given loads a pretrained pickle, encodes the image with it."""
    similarities = []
    steganogan = get_steganogan(architecture)
    images = glob(f'{input_dir}/*.png')
    pbar = tqdm(images)
    for image in pbar:
        path = pathlib.Path(image)
        full_stem = path.stem + '.png'
        undecoded = steganogan.decode(f'{image}')
        undecoded = [int(i) for i in undecoded]
        decoded = channel_code('decode',undecoded)

        try:
            analyzed = subprocess.check_output(f'env/bin/python3 analyze_face.py {image}',shell=True)
            analyzed = analyzed.splitlines()[-1].decode('utf8')
            analyzed = ast.literal_eval(analyzed)
        except Exception as e:
            print(e)
            continue

        try:
            score = dot(analyzed, decoded)/(norm(analyzed)*norm(decoded))
        except:
            continue
        # if np.isnan(cos_sim):
        #     print(gt, dec)
        #     print(cos_sim)
        #     breakpoint()
            # gt_secrets = torch.T

        swapped.append(score)
        pbar.set_description(f'score: {np.mean(swapped)}')
    
    return swapped



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify GPU and run')
    parser.add_argument('filename')
    parser.add_argument('mode')
    parser.add_argument('--secret')
    parser.add_argument('--analysis')

    args = parser.parse_args()

    if(args.mode == 'encode'):
        encode(args.filename, args.secret)
    elif(args.mode == 'decode'):
        decode(args.filename, args.analysis)
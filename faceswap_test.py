import subprocess
from glob import glob
import torch
import cProfile
import pstats
from pstats import SortKey
import time
import random
import numpy as np
from tqdm import tqdm
import model
from dataset import Data
from torch.utils.data import DataLoader
import utils
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os

input_path = './faceswap/data/test/fadg0'
# input_path = './data/celebatest/images'
output_path = './faceswap/data/test/fadg0-enc'
swap_path = './faceswap/data/test/fadg0-enc-swap'
size = 256

cos = torch.nn.CosineSimilarity(dim=1)

def clear_previous():
    files = glob(f'{output_path}/*') + glob(f'{swap_path}/*')
    for file in files:
        os.remove(file)

def run_faceswap():
    extract_cmd = '/home/luna/anaconda3/envs/facestamp/bin/python3 ./faceswap.py extract -i /home/luna/GitHub/machine-learning/facestamp-pytorch/faceswap/data/test/fadg0-enc -o /home/luna/GitHub/machine-learning/facestamp-pytorch/faceswap/data/test/fadg0-enc-extract -D s3fd -A fan -nm none -rf 0 -min 0 -l 0.4 -sz 512 -een 1 -si 0 -L INFO'
    convert_cmd = '/home/luna/anaconda3/envs/facestamp/bin/python3 ./faceswap.py convert -i ./data/test/fadg0-enc -o ./data/test/fadg0-enc-swap -al ./data/test/fadg0-enc/alignments.fsa -m ./data/f1_model -c avg-color -M extended -w opencv -osc 100 -l 0.4 -j 0 -L INFO'

    subprocess.run(extract_cmd, shell=True, cwd='./faceswap')
    subprocess.run(convert_cmd, shell=True, cwd='./faceswap')

def create_encoded(sample_size, encoder, decoder, channel_encoder, channel_decoder, cache_secrets, channel_coding=True, cuda=True):
    images = glob(f'{input_path}/*')
    random.shuffle(images)
    images = images[:sample_size]
    score = []
    for image in tqdm(images):
        path = Path(image)
        full_stem = path.stem + '.png'
        image_input = transforms.ToTensor()(Image.open(image))
        image_input = transforms.CenterCrop(min(image_input.shape[1],image_input.shape[2]))(image_input)
        image_input = transforms.Resize((size,size))(image_input)
        (transforms.ToPILImage()(image_input)).save('/encoded/image.png')

        image_input = transforms.ToTensor()(Image.open('/encoded/image.png'))
        try:
            secret_input, region_input = utils.get_secret_string('/encoded/image.png')
        except:
            continue
        image_input = image_input[None]
        secret_input = torch.Tensor(secret_input)[None]
        mask_input = utils.create_mask_input(image_input, region_input, region_transform=False)

        if(cuda):
            secret_input = secret_input.cuda()
            image_input = image_input.cuda()
            mask_input = mask_input.cuda()
        cache_secrets[full_stem] = secret_input
        orig_secret_input = secret_input.clone().detach()
        if(channel_coding):
            secret_input = channel_encoder(secret_input)
        
        residual = encoder((secret_input, image_input, mask_input))
        encoded_image = residual + image_input

        digital_copy = transforms.ToPILImage()(encoded_image.squeeze())
        digital_copy.save(f'{output_path}/{full_stem}')

        analyzed, region = utils.get_secret_string(f'{output_path}/{full_stem}')
        analyzed = torch.Tensor(analyzed).cuda()[None]
        encoded_input = transforms.ToTensor()(Image.open(f'{output_path}/{full_stem}'))
        encoded_input = encoded_input.cuda()[None]
        decoded = decoder(encoded_input)
        decoded = channel_decoder(decoded)
        decoded = torch.round(torch.clip(decoded, 0, 1))
        similarity = cos(analyzed, decoded)
        score.append(similarity.item())

        # (transforms.ToPILImage()(residual.squeeze())).show()
        # (transforms.ToPILImage()((image_input * mask_input).squeeze())).show()
    
    return score, cache_secrets

def decode_swapped(decoder, channel_decoder, cache_secrets):
    images = glob(f'{swap_path}/*')
    match_score = []
    for image in tqdm(images):
        path = Path(image)
        full_stem = path.stem + path.suffix
        image_input = transforms.ToTensor()(Image.open(image))
        image_input = image_input.cuda()
        try:
            analyzed, region = utils.get_secret_string(image)
        except:
            continue
        analyzed = torch.Tensor(analyzed).cuda()[None]
        decoded = decoder(image_input[None])
        decoded = channel_decoder(decoded)
        decoded = torch.round(torch.clip(decoded, 0, 1))
        match_similarity = cos(analyzed, decoded)
        match_score.append(match_similarity.item())
    return match_score

def faceswap_test(sample_size, encoder, decoder, channel_encoder, channel_decoder, results):
    cache_secrets = {}
    clear_previous()
    pre_score, cache_secrets = create_encoded(sample_size, encoder, decoder, channel_encoder, channel_decoder, cache_secrets)
    results['preswap_match'] = np.mean(pre_score)
    run_faceswap()
    swap_score = decode_swapped(decoder, channel_decoder, cache_secrets)
    results['swap_match'] = np.mean(swap_score)
    return results
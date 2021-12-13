import pickle
import os
import glob
import numpy as np
from PIL import Image,ImageOps
from torchvision import transforms

import torch
import model
from dataset import Data
from torch.utils.data import DataLoader, RandomSampler
import random
import string
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import utils
from tqdm import tqdm
import time
import kornia as K
from torch import nn
import json
from glob import glob
import subprocess

def read_results(decoder, channel_decoder, device, channel_encoding):
    files = glob('./data/celebaencoded/results/*.jpg')
    random.shuffle(files)
    name_results = json.load(open('./data/celebaencoded/results/names.json','r'))
    analysis_data = json.load(open('./data/celebatest/analysis.json','r'))
    cos = torch.nn.CosineSimilarity(dim=0)
    analysis = {}
    for a in analysis_data:
        fname = a[0]
        split = fname.split('/')
        fname = split[len(split)-1]
        analysis[fname] = a[1:]

    results = {}

    cur_sample = 0

    gt_analyzed = []
    gt_decoded = []
    analyzed_decoded = []

    for file in files:
        fsplit = file.split('/')
        fname = fsplit[len(fsplit)-1]
        index = fname.split('-')[0]
        results[file] = {}
        img = transforms.ToTensor()(Image.open(file))
        w_split = int(img.shape[2] / 128)
        h_split = int(img.shape[1] / 128)
        for h in range(h_split):
            results[file][h] = {}
            h_start = h * 128 
            h_end = h_start + 128
            for w in range(w_split):
                start_time = time.time()
                cur_sample += 1
                w_start = w * 128 
                w_end = w_start + 128
                if(w_start == 0):
                    cur_img = img[:,h_start: h_end, w_start: w_end]
                    cur_img = transforms.ToPILImage()(cur_img)
                    img_name = utils.save_image(cur_img)

                    cur_img = Image.open(img_name)
                    cur_img = transforms.ToTensor()(cur_img)
                    cur_img = cur_img.cuda()

                    decoded = decoder(cur_img)
                    if(channel_encoding):
                        decoded = channel_decoder(decoded)
                    decoded = torch.clip(decoded, 0, 1)
                    decoded = torch.round(decoded)
                    gt = utils.convert_secret(analysis[name_results[index][h]][0])
                    results[file][h]['gt'] = gt
                    try:
                        analyzed, region = utils.get_secret_string(img_name)
                    except Exception as e:
                        continue
                    gt = torch.Tensor(gt).cuda()
                    analyzed = torch.Tensor(analyzed).cuda()
                    decoded = decoded.squeeze()
                    gt_analyzed.append(cos(gt,analyzed).item()) 
                    gt_decoded.append(cos(gt,decoded).item())
                    analyzed_decoded.append(cos(analyzed,decoded).item())

    return np.mean(gt_analyzed), np.mean(gt_decoded), np.mean(analyzed_decoded) 

def create_encoded(dataloader, encoder, decoder, channel_encoder, channel_decoder, channel_encoding):
    timestamp = int(time.time())
    # // checkpoints/encoder_99000

    attr_list = open('./data/celebatest/list_attr_celeba.txt', 'r').read().splitlines()
    headers = attr_list[1]
    index_list = {}
    for row in attr_list[2:]:
        index = row.split('.jpg')[0]
        index_list[index] = row
    new_attr_list = [None, headers]

    count = 1
    analysis = {}
    for step, data in enumerate(tqdm(dataloader)):
        image_input, gt_secret, gt_region, name = data
        orig_gt_secret = gt_secret.clone().detach()
        image_input = image_input.cuda()
        if(channel_encoding):
            gt_secret = gt_secret.cuda()
            gt_secret = channel_encoder(gt_secret)
        gt_secret = gt_secret.cuda()
        gt_secret = torch.clip(gt_secret, 0, 1)
        name = name[0].split('/')[-1]
        mask_input = utils.create_mask_input(image_input, gt_region)
        inputs = (gt_secret, image_input, mask_input)

        with torch.no_grad():
            residual = encoder(inputs)

        encoded_image = image_input + residual 
        encoded_image = torch.clip(encoded_image, min=0, max=1)

        with torch.no_grad():
            decoded = decoder(encoded_image)
            if(channel_encoding):
                decoded = channel_decoder(decoded)
            decoded = torch.clip(decoded, 0, 1)
            decoded = torch.round(decoded)
    
        img = transforms.ToPILImage()(encoded_image.squeeze())
        index = name.split('.jpg')[0]
        img.save(f'./data/celebaencoded/images/{name}')
        new_attr_list.append(index_list[index])
        analysis[name] = (orig_gt_secret.tolist(), torch.round(decoded).tolist())
        count += 1
    
    new_attr_list[0] = str(count)
    
    json.dump(analysis, open('./data/celebaencoded/analysis.json','w'))
    with open('./data/celebaencoded/list_attr_celeba.txt', 'w') as f:
        for line in new_attr_list:
            f.write(line.strip() + '\n')

def init(channel_encoding, enc, dec, ch_enc, ch_dec):
    dataset = Data('./data/celebatest', 14, size=(128, 128), dataset_size=32)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    # enc_model_path = './evaluate/encoder_current_rw_distortion_no_channel'
    # dec_model_path = './evaluate/decoder_current_rw_distortion_no_channel'
    enc_model_path = enc
    dec_model_path = dec
    if(channel_encoding):
        scale_factor = 4
    else:
        scale_factor = 1
    ch_enc_model_path = ch_enc
    ch_dec_model_path = ch_dec

    if(channel_encoding):
        channel_encoder = model.ChannelEncoder(14)
        channel_decoder = model.ChannelDecoder(14)
        channel_encoder.load_state_dict(torch.load(ch_enc_model_path))
        channel_decoder.load_state_dict(torch.load(ch_dec_model_path))
        channel_encoder.cuda()
        channel_decoder.cuda()
        channel_encoder.eval()
        channel_decoder.eval()
    else:
        channel_encoder = None
        channel_decoder = None

    encoder = model.EncoderNet(14*scale_factor, 128, 128)
    decoder = model.DecoderNet(14*scale_factor, 128, 128)
    encoder.load_state_dict(torch.load(enc_model_path))
    decoder.load_state_dict(torch.load(dec_model_path))
    encoder.cuda()
    decoder.cuda()
    encoder.eval()
    decoder.eval()

    return dataloader, encoder, decoder, channel_encoder, channel_decoder

def clear_previous():
    files = glob('./data/celebaencoded/images/*')
    for file in files:
        os.remove(file)
    files = glob('./data/celebaencoded/results/*')
    for file in files:
        os.remove(file)
    try:
        os.remove('./data/celebaencoded/analysis.json')
    except:
        pass
    try:
        os.remove('./data/celebaencoded/list_attr_celeba.txt')
    except Exception as e:
        pass

def run_stargan():
    subprocess.run('bash test.sh', shell=True, cwd='./StarGAN')

def run_stargan_eval(enc, dec, ch_enc, ch_dec, channel_coding):
    clear_previous()
    dataloader, encoder, decoder, channel_encoder, channel_decoder = init(channel_coding, enc,
    dec, ch_enc, ch_dec)
    create_encoded(dataloader, encoder, decoder, channel_encoder, channel_decoder, channel_coding)
    run_stargan()
    results = read_results(decoder, channel_decoder, 'cuda', channel_coding)
    return results

if __name__ == '__main__':
    source = 'checkpoints'
    enc = f'{source}/encoder_current_main'
    dec = f'{source}/decoder_current_main'
    ch_enc = f'{source}/channel_encoder_current_0.2'
    ch_dec = f'{source}/channel_decoder_current_0.2'
    channel_coding = True
    gt_analyzed, gt_decoded, analyzed_decoded = run_stargan_eval(enc, dec, ch_enc, ch_dec, channel_coding)
    print('gt_analyzed %.2f' % gt_analyzed)
    print('gt_decoded %.2f' % gt_decoded)
    print('analyzed_decoded %.2f' % analyzed_decoded)
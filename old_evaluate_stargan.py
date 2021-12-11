import pickle
import os
import glob
import numpy as np
from PIL import Image,ImageOps
from torchvision import transforms

import torch
import model
import random
import string
import pandas as pd
from dataset import Data
from torch.utils.data import DataLoader, RandomSampler
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
    def getScore(class_score, age_error, total, row, row2):
        print(row, row2)
        if(row[0] == row2[0]):
            class_score['emotion'] += 1
        if(row[1] == row2[1]):
            class_score['race'] += 1
        if(row[2] == row2[2]):
            class_score['gender'] += 1
        row_age = int(row[3])
        row2_age = int(row2[3])

        age_error.append(abs(row_age - row2_age))
        total += 1
        
        return class_score, age_error, total
    
    def compile(class_score, age_error, total):
        emotion_score = class_score['emotion'] / total
        race_score = class_score['race'] / total
        gender_score = class_score['gender'] / total
        age_error_mean = np.array(age_error).mean()
        age_error_std = np.array(age_error).std()

        return (emotion_score, race_score, gender_score, age_error_mean, age_error_std)


    files = glob('./data/celebaencoded/results/*.jpg')
    name_results = json.load(open('./data/celebaencoded/results/names.json','r'))
    analysis_data = json.load(open('./data/celebatest/analysis.json','r'))
    analysis = {}
    for a in analysis_data:
        fname = a[0]
        split = fname.split('/')
        fname = split[len(split)-1]
        analysis[fname] = a[1:]
    results = {}

    files = files

    total_samples = len(files) * 16 * 6
    cur_sample = 0
    time_avg = None

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
                print('%.2f' % ((cur_sample/total_samples)*100))
                w_start = w * 128 
                w_end = w_start + 128
                cur_img = img[:,h_start: h_end, w_start: w_end]
                cur_img = cur_img.cuda()
                decoded = decoder(cur_img)
                if(channel_encoding):
                    decoded = channel_decoder(decoded)
                decoded = torch.clip(decoded, 0, 1)
                decoded = torch.round(decoded)
                decoded = utils.to_secret_string(decoded.squeeze())
                result = utils.parse_string(decoded)
                cur_img = transforms.ToPILImage()(cur_img)
                cur_img.save('./encoded/temp.png')
                gt = utils.convert_secret(analysis[name_results[index][h]][0])
                gt = ''.join([str(int(i)) for i in gt])
                gt = utils.parse_string(gt)
                results[file][h]['gt'] = gt
                try:
                    analyzed = utils.get_secret_string('./encoded/temp.png')[0]
                    analyzed = ''.join([str(int(i)) for i in analyzed])
                    analyzed = utils.parse_string(analyzed)
                    print(gt, analyzed, result)
                except Exception as e:
                    end_time = time.time()
                    run_time = end_time - start_time
                    if(time_avg == None):
                        time_avg = run_time
                    else:
                        time_avg += run_time
                        time_avg /= 2
                    time_left = time_avg * (total_samples - cur_sample)
                    minutes_left = time_left / 60
                    print('%.2f minutes left' % (minutes_left))
                    # continue
                    analyzed = [None, None, None, 0]
                if(w_start == 0):
                    print(gt, analyzed, result)
                    results[file][h]['analyzed'] = analyzed
                    results[file][h]['decoded'] = result
                # else:
                #     analyzed = analyzed
                #     fake_decoded = results[file][h].get('fake_decoded', [])
                #     fake_decoded.append(result)
                #     fake_analyzed = results[file][h].get('fake_analyzed', [])
                #     fake_analyzed.append(analyzed)
                #     results[file][h]['fake_decoded'] = fake_decoded 
                #     results[file][h]['fake_analyzed'] = fake_analyzed

                end_time = time.time()
                run_time = end_time - start_time
                if(time_avg == None):
                    time_avg = run_time
                else:
                    time_avg += run_time
                    time_avg /= 2
                time_left = time_avg * (total_samples - cur_sample)
                minutes_left = time_left / 60
                print('%.2f minutes left' % (minutes_left))
    
    age_error = []
    class_score = {}
    class_score['emotion'] = 0
    class_score['race'] = 0
    class_score['gender'] = 0
    total = 0

    final_results = {}

    for file in results:
        file_data = results[file]
        for row in file_data:
            try:
                class_score, age_error, total = getScore(class_score, age_error, total, file_data[row]['gt'], file_data[row]['decoded'])
            except Exception as e:
                print(e)
                continue

    final_results['untransformed'] = compile(class_score, age_error, total)
    
    # age_error = []
    # class_score = {}
    # class_score['emotion'] = 0
    # class_score['race'] = 0
    # class_score['gender'] = 0
    # total = 0

    # for file in results:
    #     file_data = results[file]
    #     for row in file_data:
    #         try:
    #             for idx, r in enumerate(file_data[row]['fake_decoded']):
    #                 class_score, age_error, total = getScore(class_score, age_error, total, file_data[row]['gt'], file_data[row]['fake_decoded'][idx])
    #         except Exception as e:
    #             print(e)
    #             continue
    
    # final_results['transformed'] = compile(class_score, age_error, total)

    # age_error = []
    # class_score = {}
    # class_score['emotion'] = 0
    # class_score['race'] = 0
    # class_score['gender'] = 0
    # total = 0

    # for file in results:
    #     file_data = results[file]
    #     for row in file_data:
    #         try:
    #             class_score, age_error, total = getScore(class_score, age_error, total, file_data[row]['decoded'], file_data[row]['analyzed'])
    #         except:
    #             continue
    
    # final_results['test_untransformed'] = compile(class_score, age_error, total)

    # age_error = []
    # class_score = {}
    # class_score['emotion'] = 0
    # class_score['race'] = 0
    # class_score['gender'] = 0
    # total = 0

    # for file in results:
    #     file_data = results[file]
    #     for row in file_data:
    #         try:
    #             for idx, r in enumerate(file_data[row]['fake_decoded']):
    #                     class_score, age_error, total = getScore(class_score, age_error, total, file_data[row]['fake_analyzed'][idx], r)
    #         except:
    #             continue
    
    # final_results['test_transformed'] = compile(class_score, age_error, total)

    return final_results

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
    dataset = Data('./data/celebatest', 14, size=(128, 128), dataset_size=16)
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
    # source = 'evaluate'
    # enc = 'encoder_current_no_adv_training'
    # dec = 'decoder_current_no_adv_training'
    # ch_enc = 'channel_encoder_current_0.2'
    # ch_dec = 'channel_decoder_current_0.2'
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
    results = run_stargan_eval(enc, dec, ch_enc, ch_dec, channel_coding)
    print(results)
from deepface import DeepFace
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
from faceswap_test import faceswap_test
import argparse
import os
from easydict import EasyDict
import yaml
import pickle

def tensor_similarity(inputs):
    image_input, encoded_image, secret_input, orig_secret_input, cuda, channel_coding, cos, mask_input, encoder, decoder, channel_decoder = inputs

    decoded_secret = decoder(encoded_image)

    if(channel_coding):
        decoded_secret = channel_decoder(decoded_secret)

    decoded_secret = torch.clip(decoded_secret, 0, 1)
    decoded_secret = torch.round(decoded_secret)

    tensor_similarity = cos(orig_secret_input, decoded_secret)

    return tensor_similarity.item()

def test_similarity(inputs):
    image_input, encoded_image, secret_input, orig_secret_input, cuda, channel_coding, cos, mask_input, encoder, decoder, channel_decoder = inputs
    digital_image = transforms.ToPILImage()(encoded_image.squeeze())
    img_name = utils.save_image(digital_image)

    new_digital_image = Image.open(img_name)
    new_digital_image = transforms.ToTensor()(new_digital_image)
    if(cuda):
        new_digital_image = new_digital_image.cuda()
    
    decoded_secret = decoder(new_digital_image[None])

    if(channel_coding):
        decoded_secret = channel_decoder(decoded_secret)

    decoded_secret = torch.clip(decoded_secret, 0, 1)
    decoded_secret = torch.round(decoded_secret)

    analyzed_secret, region = utils.get_secret_string(img_name)
    analyzed_secret = torch.Tensor(analyzed_secret).cuda()[None]

    test_similarity = cos(analyzed_secret, decoded_secret) 
    return test_similarity.item()

def rw_distort_similarity(inputs, args, gt_secret):
    ## RW distortions
    ## saturation 5.0
    ## hue (0.2)
    ## s&p (0.05)
    ## gaussian (0.06) 
    image_input, encoded_image, secret_input, orig_secret_input, cuda, channel_coding, cos, mask_input, encoder, decoder, channel_decoder = inputs
    encoded_image = model.distort(args, encoded_image, disortion='rw_distortion')
    digital_image = transforms.ToPILImage()(encoded_image.squeeze())
    digital_image.show()
    breakpoint()
    img_name = utils.save_image(digital_image)

    new_digital_image = Image.open(img_name)
    new_digital_image = transforms.ToTensor()(new_digital_image)
    if(cuda):
        new_digital_image = new_digital_image.cuda()
    
    decoded_secret = decoder(new_digital_image[None])

    if(channel_coding):
        decoded_secret = channel_decoder(decoded_secret)

    decoded_secret = torch.clip(decoded_secret, 0, 1)
    decoded_secret = torch.round(decoded_secret)

    print(gt_secret.shape, decoded_secret.shape)
    breakpoint()

    test_similarity = cos(gt_secret, decoded_secret) 
    return test_similarity.item()

def start_testrun(args, run_results):
    run = args.exp_name
    secret_size = args.secret_size
    channel_coding = args.channel_coding
    im_size = args.im_height
    ch_enc = './checkpoints/channel_encoder_s22'
    ch_dec = './checkpoints/channel_decoder_s22'
    enc = f'./checkpoints/encoder_current_{run}'
    dec = f'./checkpoints/decoder_current_{run}'
    cuda = True
    mask_residual = args.mask_residual
    dataset = Data('test',args.secret_size,size=(im_size,im_size))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    cos = torch.nn.CosineSimilarity(dim=1)

    if(channel_coding):
        channel_encoder = model.ChannelEncoder(args.small_secret_size, 'small')
        channel_decoder = model.ChannelDecoder(args.small_secret_size, 'small')
        channel_encoder.load_state_dict(torch.load(ch_enc))
        channel_decoder.load_state_dict(torch.load(ch_dec))
        channel_encoder.eval()
        channel_decoder.eval()
        if(cuda):
            channel_encoder = channel_encoder.cuda()
            channel_decoder = channel_decoder.cuda()
        scale_factor = 4
    else:
        scale_factor = 1
        channel_encoder = None
        channel_decoder = None

    encoder = model.EncoderNet(secret_size,im_size,im_size,mask_residual=mask_residual)
    decoder = model.DecoderNet(secret_size,im_size,im_size)
    encoder.load_state_dict(torch.load(enc))
    decoder.load_state_dict(torch.load(dec))
    encoder.cuda()
    decoder.cuda()
    encoder.eval()
    decoder.eval()

    if(cuda):
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    tensor_score = []
    test_score = []
    rw_score = []
    secrets = []
    results = {} 

    for data in tqdm(dataloader):
        image_input, mask_input, secret_input, region_input, name_input = data

        if(cuda):
            image_input = image_input.cuda()
            secret_input = secret_input.cuda()
            mask_input = mask_input.cuda()

        orig_secret_input = secret_input.clone().detach()
        if(channel_coding):
            secret_input = channel_encoder(secret_input)

        # encoded_image = image_input + encoder((secret_input, image_input, mask_input))
        residual = encoder((secret_input, image_input, mask_input))
        encoded_image = torch.clip(image_input + residual, 0, 1)

        inputs = image_input, encoded_image, secret_input, orig_secret_input, cuda, channel_coding, cos, mask_input, encoder, decoder, channel_decoder 

        tensor_score.append(tensor_similarity(inputs))
        try:
            test_score.append(test_similarity(inputs))
        except Exception as e:
            print(e)
            continue
        rw_score.append(rw_distort_similarity(inputs, args, orig_secret_input))

    results['tensor_score'] = np.mean(tensor_score)
    results['test_score'] = np.mean(test_score)
    results['rw_score'] = np.mean(rw_score)
    # results = faceswap_test(args.exp_name, encoder, decoder, channel_encoder, channel_decoder, args, results)

    run_results[run] = results

    return run_results

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify GPU and run')
    parser.add_argument('--gpu')

    args = parser.parse_args()

    try:
        gpu = str(args.gpu)
    except:
        gpu = str(0)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    def replace_args(base_args, new_args):
        for arg in new_args:
            base_args[arg] = new_args[arg]
        return base_args

    run_results = {}

    for run in glob('cfg/testing/*.yaml'):
        with open('cfg/base.yaml','r') as base_yaml:
            base_args = EasyDict(yaml.load(base_yaml, Loader=yaml.SafeLoader))
            with open(run, 'r') as run_yaml:
                run_args = EasyDict(yaml.load(run_yaml, Loader=yaml.SafeLoader))
                args = replace_args(base_args, run_args)

                utils.set_run(args.exp_name)
                run_results = start_testrun(args, run_results)
    
    timestamp = str(int(time.time()))
    pickle.dump(run_results,open(f'testing_results_{timestamp}.bin','wb'))
    print(run_results)








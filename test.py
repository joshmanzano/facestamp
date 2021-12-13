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

channel_coding = True
secret_size = 14

runs = ['main', 'rw_distortion']
run_results = {}

for run in runs:
    ch_enc = './checkpoints/channel_encoder_current_0.2'
    ch_dec = './checkpoints/channel_decoder_current_0.2'
    enc = f'./checkpoints/encoder_current_{run}'
    dec = f'./checkpoints/decoder_current_{run}'
    cuda = True
    mask_residual = True
    sample_size = 100

    dataset = Data('test',14,size=(128,128), dataset_size=sample_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    cos = torch.nn.CosineSimilarity(dim=1)

    if(channel_coding):
        channel_encoder = model.ChannelEncoder(secret_size)
        channel_decoder = model.ChannelDecoder(secret_size)
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
        
    encoder = model.EncoderNet(secret_size*scale_factor,128,128,mask_residual=mask_residual)
    decoder = model.DecoderNet(secret_size*scale_factor,128,128)
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
    secrets = []
    results = {} 

    # for data in tqdm(dataloader):
    #     image_input, mask_input, secret_input, region_input, name_input = data

    #     if(cuda):
    #         image_input = image_input.cuda()
    #         secret_input = secret_input.cuda()
    #         mask_input = mask_input.cuda()

    #     orig_secret_input = secret_input.clone().detach()
    #     if(channel_coding):
    #         secret_input = channel_encoder(secret_input)

    #     # encoded_image = image_input + encoder((secret_input, image_input, mask_input))
    #     residual = encoder((secret_input, image_input, mask_input))
    #     encoded_image = torch.clip(image_input + residual, 0, 1)

    #     inputs = image_input, encoded_image, secret_input, orig_secret_input, cuda, channel_coding, cos, mask_input, encoder, decoder, channel_decoder 

    #     tensor_score.append(tensor_similarity(inputs))
    #     try:
    #         test_score.append(test_similarity(inputs))
    #     except Exception as e:
    #         print(e)
    #         continue

    # results['tensor_score'] = np.mean(tensor_score)
    # results['test_score'] = np.mean(test_score)
    results = faceswap_test(sample_size, encoder, decoder, channel_encoder, channel_decoder, results)

    run_results[run] = results

for run in run_results:
    print(run)
    print(run_results[run])






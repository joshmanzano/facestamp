import argparse
import os
import yaml
import random
import model
import numpy as np
from glob import glob
from easydict import EasyDict
from PIL import Image, ImageOps
from torch import optim
import torch

import utils
from dataset import Data 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import time

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def checkpoint(encoder, decoder, args, prob):
    torch.save(encoder.state_dict(), f'{args.checkpoints_path}channel_encoder_current_{prob}')
    torch.save(decoder.state_dict(), f'{args.checkpoints_path}channel_decoder_current_{prob}')

def clip_round(tensor):
    tensor = torch.clip(tensor, 0, 1)
    tensor = torch.round(tensor)
    return tensor

def main(writer, args, strength):

    encoder = model.ChannelEncoder(args.secret_size)
    decoder = model.ChannelDecoder(args.secret_size)
    dataset = Data(args.train_path, args.secret_size, size=(args.im_height, args.im_width), dataset_size=args.dataset_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_dataset = Data(args.test_path, args.secret_size, size=(args.im_height, args.im_width), dataset_size=args.dataset_size)
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    encoder.apply(init_weights)
    decoder.apply(init_weights)

    mse = torch.nn.MSELoss()
    bce = torch.nn.BCELoss()

    if args.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        cross_entropy = bce.cuda()

    encoder.train()
    decoder.train()

    g_vars = [{'params': encoder.parameters()},
              {'params': decoder.parameters()}]

    optimize_loss = optim.Adam(g_vars, lr=args.lr)

    global_step = 0

    epoch = 1

    noise_avg_acc = None
    clean_avg_acc = None
    last_noise_acc = 0
    last_clean_acc = 0
    no_improvement = 0  

    while epoch < args.total_epoch:
        start_time = time.time()
        pbar = tqdm(dataloader)
        for step, data in enumerate(pbar):
            # train 
            img_cover, secret, region, name = data
            secret_input = secret

            if args.cuda:
                secret_input = secret_input.cuda()

            global_step += 1

            encoded = encoder(secret_input)


            noise_input = utils.channel_noise(encoded, strength, False)

            ## check for sigmoid vs threshold

            decoded_noise = decoder(noise_input)

            loss = cross_entropy(secret_input, decoded_noise)

            optimize_loss.zero_grad()
            loss.backward()
            optimize_loss.step()
            writer.add_scalar('loss/current',loss, global_step)

            # eval
            encoder.eval()
            decoder.eval()

            with torch.no_grad():
                noise_input = utils.channel_noise(encoded, 0.0, args.rounding_clipping)
                decoded_no_noise = decoder(noise_input)
                noise_input = utils.channel_noise(encoded, 0.1, args.rounding_clipping)
                decoded_low_noise = decoder(noise_input)
                noise_input = utils.channel_noise(encoded, 0.2, args.rounding_clipping)
                decoded_medium_noise = decoder(noise_input)
                noise_input = utils.channel_noise(encoded, 0.3, args.rounding_clipping)
                decoded_high_noise = decoder(noise_input)
                noise_input = utils.channel_noise(encoded, 0.4, args.rounding_clipping)
                decoded_extreme_noise = decoder(noise_input)
            
            cos = torch.nn.CosineSimilarity(dim=1)

            low_noise_acc = cos(secret_input, decoded_low_noise).mean()
            medium_noise_acc = cos(secret_input, decoded_medium_noise).mean()
            high_noise_acc = cos(secret_input, decoded_high_noise).mean()
            extreme_noise_acc = cos(secret_input, decoded_extreme_noise).mean() 
            clean_acc = cos(secret_input, decoded_no_noise).mean()
            
            writer.add_scalar('eval_acc/low_noise',low_noise_acc, global_step)
            writer.add_scalar('eval_acc/medium_noise',medium_noise_acc, global_step)
            writer.add_scalar('eval_acc/high_noise',high_noise_acc, global_step)
            writer.add_scalar('eval_acc/extreme_noise',extreme_noise_acc, global_step)
            writer.add_scalar('eval_acc/clean',clean_acc, global_step)
            
            encoder.train()
            decoder.train()

            if(step % 5 == 0):
                mean = torch.mean(encoded).item()
                std = torch.std(encoded).item()
                pbar.set_description('mean: %.2f, std: %.2f, clean: %.2f, low: %.2f, med: %.2f, high: %.2f' % (mean, std, clean_acc, low_noise_acc, medium_noise_acc, high_noise_acc))
            if(step % 11 == 0):
                pbar.set_description('%.2f -> %.2f' % (secret_input[0][0].item(), decoded_noise[0][0].item()))

        checkpoint(encoder, decoder, args, strength)
        print(f'Epoch {epoch}: {clean_acc}')
        end_time = time.time()
        time_taken = (end_time - start_time)
        print(f'Total time taken: {int(time_taken)} seconds')
        writer.add_scalar('misc/time_taken', time_taken, epoch)
        epoch += 1

    checkpoint(encoder, decoder, args, strength)
    print(f'Run with {strength} probability done.')
    print(f'{last_noise_acc} {last_clean_acc}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('channel_strength')
    cmd_args = parser.parse_args()
    with open('cfg/channel.yaml','r') as channel_yaml:
        args = EasyDict(yaml.load(channel_yaml, Loader=yaml.SafeLoader))
        # for prob in [float(i/10) for i in range(0, 10, 3)]:
        writer = SummaryWriter(log_dir=f'./channel_logs/channel_training_{cmd_args.channel_strength}')
        main(writer, args, float(cmd_args.channel_strength))
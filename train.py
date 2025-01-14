
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
import lpips

import utils
from dataset import Data 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import time
from multiprocessing import Process

import gc

import cProfile
from evaluate_stargan import run_stargan_eval
import argparse

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def periodic_checkpoint(encoder, decoder, args, epoch):
    torch.save(encoder.state_dict(), f'{args.periodic_checkpoints_path}encoder_epoch_{epoch}_{args.exp_name}')
    torch.save(decoder.state_dict(), f'{args.periodic_checkpoints_path}decoder_epoch_{epoch}_{args.exp_name}')

def checkpoint(encoder, decoder, args):
    torch.save(encoder.state_dict(), f'{args.checkpoints_path}encoder_current_{args.exp_name}')
    torch.save(decoder.state_dict(), f'{args.checkpoints_path}decoder_current_{args.exp_name}')

def main(writer, args, gpu):

    dataset = Data('train', args.small_secret_size, size=(args.im_height, args.im_width))
    eval_dataset = Data('eval', args.small_secret_size, size=(args.im_height, args.im_width))
    test_dataset = Data('test', args.small_secret_size, size=(args.im_height, args.im_width))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    channel_encoder = None
    channel_decoder = None

    if(args.channel_coding):
        channel_encoder = model.ChannelEncoder(args.small_secret_size, 'small')
        channel_decoder = model.ChannelDecoder(args.small_secret_size, 'small')
        channel_encoder.load_state_dict(torch.load(args.channel_encoder))
        channel_decoder.load_state_dict(torch.load(args.channel_decoder))
        channel_encoder.eval()
        channel_decoder.eval()
        if(args.cuda):
            channel_encoder = channel_encoder.cuda()
            channel_decoder = channel_decoder.cuda()
    
    if(args.distortion_method == 'network' or args.distortion_method == 'all'):
        attacker = model.AttackNet(args.im_height,args.im_width)
        attacker.apply(init_weights)
        if(args.cuda):
            attacker = attacker.cuda()
        attacker.train()
    else:
        attacker = None

    encoder = model.EncoderNet(args.secret_size,args.im_height,args.im_width,mask_residual=args.mask_residual)
    decoder = model.DecoderNet(args.secret_size,args.im_height,args.im_width)

    encoder.apply(init_weights)
    decoder.apply(init_weights)

    # real_distortions = ['grayscale','saturation','hue','motion_blur','color_manipulation','gaussian','crop','noise', 'rw_distortion']
    real_distortions = ['grayscale','crop','rw_distortion']
    fake_distortions = ['black','white','random']

    ## losses
    mse = torch.nn.MSELoss()
    cos = torch.nn.CosineSimilarity(dim=1)

    if args.load:
        # torch.load(open('checkpoints/encoder_current'))
        pass

    if args.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        mse = mse.cuda()
        cos = cos.cuda()

    encoder.train()
    decoder.train()

    g_vars = [{'params': encoder.parameters()},
              {'params': decoder.parameters()}]

    if(args.distortion_method == 'network' or args.distortion_method == 'all'):
        g_vars.append({'params': attacker.parameters()})

    optimize_loss = optim.Adam(g_vars, lr=args.lr)

    total_steps = len(dataset) // args.batch_size + 1
    global_step = 0

    epoch = 0
    all_losses = False

    while(epoch < args.max_epochs):
        pbar = tqdm(dataloader)
        pbar.set_description('epoch %.2f, eval_acc: %.2f, train_acc: %.2f, gpu_temp: %.2f' % (0,0,0,0))
        train_acc_arr = []
        for data in pbar:
            # train 
            image_input, mask_input, secret_input, region_input, name_input = data

            if args.cuda:
                image_input = image_input.cuda()
                secret_input = secret_input.cuda()
                mask_input = mask_input.cuda()
            
            if args.channel_coding:
                orig_secret_input = secret_input.clone().detach()
                secret_input = channel_encoder(secret_input)
            else:
                orig_secret_input = None

            no_adv = False
            
            global_step += 1

            loss, train_acc = model.build_model(encoder, decoder, channel_decoder, attacker, cos, mse, orig_secret_input, secret_input, image_input, mask_input, 
                    args, global_step, writer, region_input, epoch, all_losses)
            
            train_acc_arr.append(train_acc.cpu().item())

            optimize_loss.zero_grad()
            loss.backward()
            optimize_loss.step()

            # eval 
            image_input, mask_input, secret_input, region_input, name_input = next(iter(eval_dataloader))

            if args.cuda:
                image_input = image_input.cuda()
                secret_input = secret_input.cuda()
                mask_input = mask_input.cuda()

            if args.channel_coding:
                orig_secret_input = secret_input.clone().detach()
                secret_input = channel_encoder(secret_input)
            else:
                orig_secret_input = None

            if(global_step % 10 == 0):
                gpu_temp = int(utils.get_temperature(gpu))
                eval_acc = model.single_eval(encoder, decoder, channel_decoder, cos, orig_secret_input, secret_input, image_input, mask_input,
                        args, region_input)
                if(not all_losses and eval_acc > 0.99):
                    print('All losses will now be applied.')
                    all_losses = True
                if(gpu_temp > 87):
                    time.sleep(1)
                writer.add_scalar('summary/train_acc', train_acc, global_step)
                writer.add_scalar('summary/eval_acc', eval_acc, global_step)
                writer.add_scalar('misc/gpu_temp', gpu_temp, global_step)
                pbar.set_description('epoch: %.0f, eval_acc: %.2f, train_acc: %.2f, gpu_temp: %.2f' % (epoch, eval_acc.item(), train_acc.item(), gpu_temp))

            # clear memory
            # gc.collect()
            # torch.cuda.empty_cache()

        checkpoint(encoder, decoder, args)
        periodic_checkpoint(encoder, decoder, args, epoch)

        performance = []
        test_performance = []
        test_losses = []
        if(args.eval_tuning):
            for idx, data in enumerate(tqdm(eval_dataloader)):
                image_input, mask_input, secret_input, region_input, name_input = data
                if args.cuda:
                    image_input = image_input.cuda()
                    secret_input = secret_input.cuda()
                    mask_input = mask_input.cuda()
                if args.channel_coding:
                    orig_secret_input = secret_input.clone().detach()
                    secret_input = channel_encoder(secret_input)
                else:
                    orig_secret_input = None

                if(not args.eval_tuning):
                    encoder.eval()
                    decoder.eval()

                test_acc, loss = model.eval_model(encoder, decoder, mse, channel_decoder, cos, image_input, mask_input, 
                        secret_input, args, region_input)

                if(args.eval_tuning and all_losses):
                    test_losses.append(loss.cpu().item())
                    optimize_loss.zero_grad()
                    loss.backward()
                    optimize_loss.step()
                
                encoder.train()
                decoder.train()
                
                test_performance.append(test_acc.cpu().item())

            test_mean_data, test_std_data = utils.process_test_data(test_performance)
            writer.add_scalar('summary/test_acc', np.mean(test_mean_data), global_step)
            test_loss_mean = np.mean(test_losses)
            writer.add_scalar('train_loss/test_loss', test_loss_mean, global_step)

        encoder.eval()
        decoder.eval()

        performance = []
        test_performance = []
        for idx, data in enumerate(tqdm(test_dataloader)):
            image_input, mask_input, secret_input, region_input, name_input = data
            if args.cuda:
                image_input = image_input.cuda()
                secret_input = secret_input.cuda()
                mask_input = mask_input.cuda()
            if args.channel_coding:
                orig_secret_input = secret_input.clone().detach()
                secret_input = channel_encoder(secret_input)
            else:
                orig_secret_input = None
            performance_row = model.test_model(encoder, decoder, channel_decoder, cos, orig_secret_input, secret_input, image_input, mask_input,
                    args, global_step, writer, region_input, real_distortions, 'real_distortions', idx == 0)
            performance.append(performance_row)

        real_mean_data, real_std_data = utils.process_test_data(performance)
        utils.create_barh(writer, 'eval_graph/real_distortions','Real Distortions', 'Accuracy', real_distortions, real_mean_data, epoch)
        writer.add_scalar('summary/real_dist_acc', np.mean(real_mean_data), global_step)

        performance = []
        for idx, data in enumerate(tqdm(test_dataloader)):
            image_input, mask_input, secret_input, region_input, name_input = data
            if args.cuda:
                image_input = image_input.cuda()
                secret_input = secret_input.cuda()
                mask_input = mask_input.cuda()
            if args.channel_coding:
                orig_secret_input = secret_input.clone().detach()
                secret_input = channel_encoder(secret_input)
            else:
                orig_secret_input = None
            performance_row = model.test_model(encoder, decoder, channel_decoder, cos, orig_secret_input, secret_input, image_input, mask_input,
                    args, global_step, writer, region_input, fake_distortions, 'fake_distortions', idx == 0)
            performance.append(performance_row)

        fake_mean_data, fake_std_data = utils.process_test_data(performance)
        utils.create_barh(writer, 'eval_graph/fake_distortions','Fake Distortions', 'Accuracy', fake_distortions, fake_mean_data, epoch)
        writer.add_scalar('summary/fake_dist_acc', np.mean(fake_mean_data), global_step)

        if(args.eval_tuning):
            test_mean = np.mean(test_mean_data) * 100
            test_std = np.mean(test_std_data) * 100
        train_mean = np.mean(train_acc_arr) * 100
        train_std = np.std(train_acc_arr) * 100
        real_mean = np.mean(real_mean_data) * 100
        real_std = np.std(real_std_data) * 100
        fake_mean = np.mean(fake_mean_data) * 100
        fake_std = np.std(fake_std_data) * 100

        encoder.train()
        decoder.train()

        if(not all_losses and train_mean > 90):
            print('All losses will now be applied.')
            all_losses = True
            
        epoch += 1
        print(f'epoch = {epoch}\nstep = {global_step}')
        print('avg_train_acc = %.2f (%.2f)' % (train_mean, train_std))
        if(args.eval_tuning):
            print('avg_test_acc = %.2f (%.2f)' % (test_mean, test_std))
        print('avg_real_acc = %.2f (%.2f)' % (real_mean, real_std))
        print('avg_fake_acc = %.2f (%.2f)' % (fake_mean, fake_std))

        utils.clear_run_images()


    checkpoint(encoder, decoder, args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify GPU and run')
    parser.add_argument('--gpu')
    parser.add_argument('--runs')

    args = parser.parse_args()

    try:
        runs = args.runs
    except:
        runs = 'main'

    try:
        gpu = str(args.gpu)
    except:
        gpu = str(0)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    def replace_args(base_args, new_args):
        for arg in new_args:
            base_args[arg] = new_args[arg]
        return base_args
    
    runs = runs.split(',')

    for run in runs:

        run = run.strip()

        if run == '':
            continue

        with open('cfg/base.yaml','r') as base_yaml:
            base_args = EasyDict(yaml.load(base_yaml, Loader=yaml.SafeLoader))
            with open(f'cfg/runs/{run}.yaml', 'r') as run_yaml:
                run_args = EasyDict(yaml.load(run_yaml, Loader=yaml.SafeLoader))
                args = replace_args(base_args, run_args)

                if not os.path.exists(args.checkpoints_path):
                    os.makedirs(args.checkpoints_path)

                if not os.path.exists(args.logs_path):
                    os.makedirs(args.logs_path)

                if not os.path.exists(args.encoded_path):
                    os.makedirs(args.encoded_path)
                
                def random_weight():
                    rand_num = float(args.random_min) + (random.random() * float(args.random_max))
                    return rand_num
                
                if(args.randomize_weights):
                    eval = random_weight()
                    args.eval_loss_weight = eval
                    percep = random_weight()
                    args.lpips_loss_weight = percep
                    secret_loss = random_weight()
                    args.secret_loss_weight = secret_loss
                    adv_secret = random_weight()
                    args.adv_secret_loss_weight = adv_secret
                    residual = random_weight()
                    args.residual_loss_weight = residual
                    # a1
                    a1 = random_weight()
                    args.adv_similarity_weight = a1
                    # a2
                    a2 = random_weight()
                    args.adv_strength_weight = a2
                
                    args.exp_name = 'weights_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f' % (eval, percep, secret_loss, adv_secret, residual, a1, a2)
                    args.verbose_exp_name = 'Weights %.2f %.2f %.2f %.2f %.2f %.2f %.2f' % (eval, percep, secret_loss, adv_secret, residual, a1, a2)
                
                log_path = os.path.join(args.logs_path, f'{args.verbose_exp_name} ({int(time.time())})')
                utils.set_run(args.exp_name)
                writer = SummaryWriter(log_dir=log_path)
                main(writer, args, int(gpu))
                writer.close()

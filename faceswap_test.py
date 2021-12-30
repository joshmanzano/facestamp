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

base_path = './faceswap/data/test/'

df_models = [
    {
        'name': 'ff1',
        'source': 'fadg0'
    },
    {
        'name': 'mm3',
        'source': 'mwbt0'
    },
    {
        'name': 'ff2',
        'source': 'fcft0'
    },
    {
        'name': 'mm4',
        'source': 'mccs0'
    },
    {
        'name': 'mf6',
        'source': 'mcem0'
    },
    {
        'name': 'fm7',
        'source': 'fgjd0',
    }
]

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

threshold = 0.9
thresholds = np.linspace(0.7, 0.95, 5)

cos = torch.nn.CosineSimilarity(dim=1)

def clear_previous():
    for df_model in df_models:
        output_path = df_model['output_path'] 
        swap_path = df_model['swap_path']
        residual_path = df_model['residual_path']
        files = glob(f'{output_path}/*') + glob(f'{swap_path}/*') + glob(f'{residual_path}/*')
        for file in files:
            os.remove(file)

def run_faceswap():
    for df_model in df_models:
        name = df_model['name']
        source = df_model['source']
        extract_cmd = f'/home/luna/anaconda3/envs/facestamp/bin/python3 ./faceswap.py extract -i ./data/test/{source}-enc -o ./data/test/{source}-enc-extract -D s3fd -A fan -nm none -rf 0 -min 0 -l 0.4 -sz 512 -een 1 -si 0 -L INFO'
        convert_cmd = f'/home/luna/anaconda3/envs/facestamp/bin/python3 ./faceswap.py convert -i ./data/test/{source}-enc -o ./data/test/{source}-enc-swap -al ./data/test/{source}-enc/alignments.fsa -m ./data/{name}_model -c avg-color -M extended -w opencv -osc 100 -l 0.4 -j 0 -L INFO'

        subprocess.run(extract_cmd, shell=True, cwd='./faceswap')
        subprocess.run(convert_cmd, shell=True, cwd='./faceswap')

def create_encoded(encoder, decoder, channel_encoder, channel_decoder, args, cache_secrets, acc):
    score = []
    for df_model in df_models:
        print(df_model['name'], df_model['source'])
        output_path = df_model['output_path']
        resized_path = df_model['resized_path']
        residual_path = df_model['residual_path']
        images = glob(f'{resized_path}/*')
        random.shuffle(images)
        size = int(len(images) * 0.02)
        images = images[:size]
        for step, image in enumerate(tqdm(images)):
            path = Path(image)
            full_stem = path.stem + '.png'
            image_input = transforms.ToTensor()(Image.open(image))
            try:
                secret_input, region_input = utils.get_secret_string(image)
            except:
                continue
            df_model['example'] = full_stem
            image_input = image_input[None]
            secret_input = torch.Tensor(secret_input)[None]
            mask_input = utils.create_mask_input(image_input, region_input, region_transform=False)

            secret_input = secret_input.cuda()
            image_input = image_input.cuda()
            mask_input = mask_input.cuda()
            cache_secrets[full_stem] = secret_input
            orig_secret_input = secret_input.clone().detach()
            if(args.channel_coding):
                secret_input = channel_encoder(secret_input)
            
            residual = encoder((secret_input, image_input, mask_input))
            (transforms.ToPILImage()(residual.squeeze())).save(f'{residual_path}/{full_stem}')
            encoded_image = residual + image_input

            digital_copy = transforms.ToPILImage()(encoded_image.squeeze())
            digital_copy.save(f'{output_path}/{full_stem}')

            try:
                analyzed, region = utils.get_secret_string(f'{output_path}/{full_stem}')
            except:
                continue
            analyzed = torch.Tensor(analyzed).cuda()[None]
            encoded_input = transforms.ToTensor()(Image.open(f'{output_path}/{full_stem}'))
            encoded_input = encoded_input.cuda()[None]
            decoded = decoder(encoded_input)
            if(args.channel_coding):
                decoded = channel_decoder(decoded)
            decoded = torch.round(torch.clip(decoded, 0, 1))
            similarity = cos(analyzed, decoded)
            similarity = similarity.item()
            acc['tn'] = {}
            acc['fp'] = {}
            for threshold in thresholds:
                if(similarity >= threshold):
                    ## true negative
                    # acc['tn'][threshold] += 1
                    acc['tn'][threshold] = acc['tn'].get(threshold, 0) + 1
                elif(similarity < threshold):
                    ## false positive
                    acc['fp'][threshold] = acc['fp'].get(threshold, 0) + 1
            score.append(similarity)

            # (transforms.ToPILImage()(residual.squeeze())).show()
            # (transforms.ToPILImage()((image_input * mask_input).squeeze())).show()
    
    return score, acc

def decode_swapped(decoder, channel_decoder, args, cache_secrets, acc):
    match_score = []
    unreadable = 0
    failed = 0
    for df_model in df_models:
        swap_path = df_model['swap_path']
        images = glob(f'{swap_path}/*')
        for image in tqdm(images):
            path = Path(image)
            full_stem = path.stem + path.suffix
            try:
                image_input = transforms.ToTensor()(Image.open(image))
            except:
                failed += 1
                continue
            image_input = image_input.cuda()
            try:
                analyzed, region = utils.get_secret_string(image)
            except:
                unreadable += 1
                continue
            analyzed = torch.Tensor(analyzed).cuda()[None]
            decoded = decoder(image_input[None])
            if(args.channel_coding):
                decoded = channel_decoder(decoded)
            decoded = torch.round(torch.clip(decoded, 0, 1))
            match_similarity = cos(analyzed, decoded)
            similarity = match_similarity.item()
            match_score.append(similarity)
            acc['fn'] = {}
            acc['tp'] = {}
            for threshold in thresholds:
                if(similarity >= threshold):
                    ## false negative
                    acc['fn'][threshold] = acc['fn'].get(threshold, 0) + 1
                elif(similarity < threshold):
                    ## true positive
                    acc['tp'][threshold] = acc['tp'].get(threshold, 0) + 1
    return match_score, acc, unreadable, failed

def compile_examples(run, residuals=False):
    output = Image.new('RGB', (size*3, size*len(df_models)), 'white')
    for idx, df_model in enumerate(df_models):
        # clean_path = df_model['resized_path'] + '/' + df_model[example]
        # encoded_path = df_model['output_path'] + '/' + df_model[example]
        # swapped_path = df_model['swap_path'] + '/' + df_model[example]
        # residual_path = df_model['residual_path'] + '/' + df_model[example]
        try:
            clean_path = glob(df_model['resized_path'] + '/*.png')[0]
            encoded_path = glob(df_model['output_path'] + '/*.png')[0]
            swapped_path = glob(df_model['swap_path'] + '/*.png')[0]
            if(residuals):
                residual_path = glob(df_model['residual_path'] + '/*.png')[0]
        except:
            continue

        try:
            clean = Image.open(clean_path)
            encoded = Image.open(encoded_path)
            swapped = Image.open(swapped_path)
            if(residuals):
                residual = Image.open(residual_path)
        except:
            continue

        output.paste(clean, (0, idx*size))
        if(residuals):
            output.paste(residual, (size, idx*size))
        if(not residuals):
            output.paste(encoded, (size, idx*size))
            output.paste(swapped, (size*2, idx*size))
        else:
            output.paste(encoded, (size*2, idx*size))
            output.paste(swapped, (size*3, idx*size))

    timestamp = str(int(time.time()))
    output.save(f'./results/{run}-{timestamp}.png')

def prepare_resized(input_path, resized_path):
    images = glob(f'{input_path}/*')
    for step, image in enumerate(tqdm(images)):
        path = Path(image)
        full_stem = path.stem + '.png'
        image_input = transforms.ToTensor()(Image.open(image))
        image_input = transforms.CenterCrop(min(image_input.shape[1],image_input.shape[2]))(image_input)
        image_input = transforms.Resize((size,size))(image_input)
        # img_name = utils.save_image(transforms.ToPILImage()(image_input))
        img_path = f'{resized_path}/{full_stem}'
        (transforms.ToPILImage()(image_input)).save(img_path)

def prepare_all_resized():
    for df_model in df_models:
        prepare_resized(df_model['input_path'], df_model['resized_path'])

def faceswap_test(run, encoder, decoder, channel_encoder, channel_decoder, args, results):
    cache_secrets = {}
    acc = {
        # 'tp': 0,
        # 'tn': 0,
        # 'fn': 0,
        # 'fp': 0
    }
    clear_previous()
    # prepare_all_resized()
    pre_score, acc = create_encoded(encoder, decoder, channel_encoder, channel_decoder, args, cache_secrets, acc)
    results['preswap_match'] = (np.mean(pre_score), np.std(pre_score))
    run_faceswap()
    swap_score, acc, unreadable, failed = decode_swapped(decoder, channel_decoder, args, cache_secrets, acc)
    results['swap_match'] = (np.mean(swap_score), np.std(swap_score))
    results['unreadable'] = unreadable
    results['failed'] = failed
    results['acc'] = acc
    compile_examples(run, residuals=True)
    return results

def test_models():
    clear_previous()
    for df_model in df_models:
        resized_path = df_model['resized_path']
        source = df_model['source']
        images = glob(f'{resized_path}/*.png')
        random.shuffle(images)
        size = int(len(images) * 0.02)
        images = images[:size]

        for image in images:
            path = Path(image)
            full_stem = path.stem + '.png'
            with open(image, 'rb') as f:
                with open(f'./faceswap/data/test/{source}-enc/{full_stem}', 'wb') as f2:
                    f2.write(f.read())

        name = df_model['name']
        source = df_model['source']
        extract_cmd = f'/home/luna/anaconda3/envs/facestamp/bin/python3 ./faceswap.py extract -i ./data/test/{source}-enc -o ./data/test/{source}-enc-extract -D s3fd -A fan -nm none -rf 0 -min 0 -l 0.4 -sz 512 -een 1 -si 0 -L INFO'
        convert_cmd = f'/home/luna/anaconda3/envs/facestamp/bin/python3 ./faceswap.py convert -i ./data/test/{source}-enc -o ./data/test/{source}-enc-swap -al ./data/test/{source}-enc/alignments.fsa -m ./data/{name}_model -c avg-color -M extended -w opencv -osc 100 -l 0.4 -j 0 -L INFO'

        subprocess.run(extract_cmd, shell=True, cwd='./faceswap')
        subprocess.run(convert_cmd, shell=True, cwd='./faceswap')

    compile_examples('test')


if __name__ == '__main__':
    thresholds = np.linspace(70, 95, 5)
    print(thresholds)
    # test_models()
    # clear_previous()
    # prepare_all_resized()

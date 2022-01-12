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
import pathlib
import json
from model import test_distort
import pickle

base_path = './test_data/vidtimit/process/'
df_models = json.load(open('df_models.json','r')) 
test_size = 200
specific = '*.png'

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

    swap_distort_path = base_path + df_model['source'] + '-enc-swap-distort'
    if not os.path.exists(swap_distort_path):
        os.makedirs(swap_distort_path)

    temp_path = base_path + df_model['source'] + '-temp'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)


    df_model['input_path'] = input_path
    df_model['resized_path'] = resized_path
    df_model['residual_path'] = residual_path
    df_model['output_path'] = output_path
    df_model['swap_path'] = swap_path
    df_model['swap_distort_path'] = swap_distort_path
    df_model['temp_path'] = temp_path 

size = 400

cos = torch.nn.CosineSimilarity(dim=1)

def clear_previous():
    for df_model in df_models:
        output_path = df_model['output_path'] 
        swap_path = df_model['swap_path']
        swap_distort_path= df_model['swap_distort_path']
        residual_path = df_model['residual_path']
        temp_path = df_model['temp_path']
        files = glob(f'{output_path}/*') + glob(f'{swap_path}/*') + glob(f'{residual_path}/*') + glob(f'{swap_distort_path}/*') + glob(f'{temp_path}/*')
        for file in files:
            os.remove(file)

def clear_temp():
    for df_model in df_models:
        temp_path = df_model['temp_path']
        files = glob(f'{temp_path}/*')
        for file in files:
            os.remove(file)

def clear_swapped():
    for df_model in df_models:
        swap_path = df_model['swap_path']
        swap_distort_path = df_model['swap_distort_path']
        files = glob(f'{swap_path}/*') + glob(f'{swap_distort_path}/*')
        for file in files:
            os.remove(file)

def run_faceswap():
    clear_swapped()
    for df_model in df_models:
        name = df_model['name']
        source = df_model['source']
        extract_cmd = f'/home/luna/anaconda3/envs/facestamp/bin/python3 ./faceswap.py extract -i ./test_data/vidtimit/process/{source}-enc -o ./test_data/vidtimit/process/{source}-enc-extract -D s3fd -A fan -nm none -rf 0 -min 0 -l 0.4 -sz 512 -een 1 -si 0 -L INFO'
        convert_cmd = f'/home/luna/anaconda3/envs/facestamp/bin/python3 ./faceswap.py convert -i ./test_data/vidtimit/process/{source}-enc -o ./test_data/vidtimit/process/{source}-enc-swap -al ./test_data/vidtimit/process/{source}-enc/alignments.fsa -m ./test_data/faceswap_models/{name}_model -c avg-color -M extended -w opencv -osc 100 -l 0.4 -j 0 -L INFO'

        subprocess.run(extract_cmd, shell=True, cwd='./faceswap')
        subprocess.run(convert_cmd, shell=True, cwd='./faceswap')

def run_simswap():
    clear_swapped()
    for df_model in df_models:
        name = df_model['name']
        source = df_model['source']
        source_images = glob(f'./test_data/vidtimit/process/{source}-enc/*.png')
        target_dirs = glob(f'./test_data/vidtimit/raw/*')
        random.shuffle(target_dirs)

        for step, src in enumerate(source_images):
            filename = pathlib.Path(src).stem + '.png'
            target_dir = pathlib.Path(random.choice(target_dirs)).stem
            target_filename = pathlib.Path(random.choice(glob(f'./test_data/vidtimit/raw/{target_dir}/*.jpg'))).stem + '.jpg'
            convert_cmd = f'/home/luna/anaconda3/envs/simswap/bin/python3 test_wholeimage_swapsingle.py --crop_size 224 --use_mask --name people --Arc_path ./arcface_model/arcface_checkpoint.tar --pic_b_path ./test_data/vidtimit/process/{source}-enc/{filename} --pic_a_path ./test_data/vidtimit/raw/{target_dir}/{target_filename} --output_path ./test_data/vidtimit/process/{source}-enc-swap --no_simswaplogo'
            rename_cmd = f'mv ./test_data/vidtimit/process/{source}-enc-swap/result_whole_swapsingle.jpg ./test_data/vidtimit/process/{source}-enc-swap/{filename}'

            subprocess.run(convert_cmd, shell=True, cwd='./SimSwap')
            subprocess.run(rename_cmd, shell=True, cwd='./')
            # print(convert_cmd)

def run_compression(suffix, quality):
    ## suffix = enc-swap
    for df_model in df_models:
        name = df_model['name']
        source = df_model['source']
        images = glob(f'./test_data/vidtimit/process/{source}-{suffix}/*.png')
        for image in images:
            stem = pathlib.Path(image).stem
            img = Image.open(image)
            img = test_distort(img, 400, 400, 'jpeg_compression', quality=quality)
            os.remove(image)
            img.save(f'./test_data/vidtimit/process/{source}-{suffix}/{stem}.png')

def run_blur(suffix):
    ## suffix = enc-swap
    for df_model in df_models:
        name = df_model['name']
        source = df_model['source']
        images = glob(f'./test_data/vidtimit/process/{source}-{suffix}/*.png')
        for image in images:
            stem = pathlib.Path(image).stem
            img = Image.open(image)
            img = test_distort(img, 400, 400, 'gaussian_blur')
            os.remove(image)
            img.save(f'./test_data/vidtimit/process/{source}-{suffix}/{stem}.png')

def get_test_data(resized_path):
    analysis = pickle.load(open(resized_path + '/analysis.bin','rb'))
    test_data = []

    for step, data in enumerate(analysis):
        image_path = f'{resized_path}/{data}.png'
        secret_input = analysis[data][0]
        region_input = analysis[data][1]
        test_data.append((image_path, secret_input, region_input))
    
    random.shuffle(test_data)
    test_data = test_data[:test_size]

    return test_data 

def create_encoded(encoder, channel_encoder, args, cache_secrets):
    for df_model in df_models:
        print(df_model['name'], df_model['source'])
        output_path = df_model['output_path']
        resized_path = df_model['resized_path']
        residual_path = df_model['residual_path']
        # size = int(len(images) * 0.2)
        # size = int(len(images))
        test_data = get_test_data(resized_path)
        for step, data in enumerate(tqdm(test_data)):
            image = data[0]
            path = Path(image)
            stem = path.stem
            full_stem = stem + '.png'
            try:
                image_input = transforms.ToTensor()(Image.open(image))
            except Exception as e:
                print(e)
                breakpoint()
            secret_input = data[1]
            region_input = data[2]
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
                secret_input = torch.round(torch.clip(secret_input, 0, 1))
            
            residual = encoder((secret_input, image_input, mask_input))
            (transforms.ToPILImage()(residual.squeeze())).save(f'{residual_path}/{full_stem}')
            encoded_image = residual + image_input

            digital_copy = transforms.ToPILImage()(encoded_image.squeeze())
            digital_copy.save(f'{output_path}/{full_stem}')

def create_encoded_stegastamp(channel_encoder, args, cache_secrets):
    for df_model in df_models:
        print(df_model['name'], df_model['source'])
        output_path = df_model['output_path']
        resized_path = df_model['resized_path']
        residual_path = df_model['residual_path']
        temp_path = df_model['temp_path']
        test_data = get_test_data(resized_path)
        secrets = {}
        for step, data in enumerate(tqdm(test_data)):
            image = data[0]
            path = Path(image)
            stem = path.stem
            full_stem = path.stem + '.png'
            secret_input = data[1]
            secret_input = torch.Tensor(secret_input)[None]
            secret_input = secret_input.cuda()
            if(args.channel_coding):
                secret_input = channel_encoder(secret_input)
                secret_input = torch.round(torch.clip(secret_input, 0, 1))

            secrets[stem] = secret_input.squeeze().tolist()
            subprocess.run([f'cp {image} {temp_path}/{full_stem}'],shell=True)
        pickle.dump(secrets, open('secrets.bin','wb'))

        run_cmd = f'/home/luna/anaconda3/envs/stegastamp/bin/python3 run_stegastamp.py {temp_path} encode secrets.bin --output_dir {output_path}'

        subprocess.run(run_cmd, shell=True, cwd='./')

def decode_stegastamp(base_path, channel_decoder, args, cache_secrets):
    match_score = []
    for df_model in df_models:
        swap_path = df_model[base_path]
        images = glob(f'{swap_path}/*.png')
        run_cmd = f'/home/luna/anaconda3/envs/stegastamp/bin/python3 run_stegastamp.py {swap_path} decode secrets.bin'
        subprocess.run(run_cmd, shell=True, cwd='./')
        secrets = pickle.load(open('secrets.bin', 'rb'))
        for image in tqdm(images):
            path = Path(image)
            stem = path.stem
            full_stem = stem + path.suffix
            try:
                analyzed, region = utils.get_secret_string(image)
            except:
                continue
            decoded = secrets[stem]
            analyzed = torch.Tensor(analyzed).cuda()[None]
            decoded = torch.Tensor(decoded).cuda()[None]
            if(args.channel_coding):
                decoded = channel_decoder(decoded)
            decoded = torch.round(torch.clip(decoded, 0, 1))
            match_similarity = cos(analyzed, decoded)
            similarity = match_similarity.item()
            match_score.append(similarity)
    return match_score
            
def create_encoded_steganogan(channel_encoder, args, cache_secrets):
    for df_model in df_models:
        print(df_model['name'], df_model['source'])
        output_path = df_model['output_path']
        resized_path = df_model['resized_path']
        residual_path = df_model['residual_path']
        temp_path = df_model['temp_path']
        test_data = get_test_data(resized_path)
        secrets = {}
        for step, data in enumerate(tqdm(test_data)):
            image = data[0]
            path = Path(image)
            stem = path.stem
            full_stem = path.stem + '.png'
            secret_input = data[1]
            secret_input = torch.Tensor(secret_input)[None]
            secret_input = secret_input.cuda()
            if(args.channel_coding):
                secret_input = channel_encoder(secret_input)
                secret_input = torch.round(torch.clip(secret_input, 0, 1))

            secrets[stem] = secret_input.squeeze().tolist()
            subprocess.run([f'cp {image} {temp_path}/{full_stem}'],shell=True)
            print(stem, full_stem)
        pickle.dump(secrets, open('secrets.bin','wb'))

        run_cmd = f'/home/luna/anaconda3/envs/steganogan/bin/python3 run_steganogan.py {temp_path} encode secrets.bin residual --output_dir {output_path}'

        subprocess.run(run_cmd, shell=True, cwd='./')
    
def decode_steganogan(base_path, channel_decoder, args, cache_secrets):
    match_score = []
    for df_model in df_models:
        swap_path = df_model[base_path]
        images = glob(f'{swap_path}/*.png')
        run_cmd = f'/home/luna/anaconda3/envs/steganogan/bin/python3 run_steganogan.py {swap_path} decode secrets.bin residual'
        subprocess.run(run_cmd, shell=True, cwd='./')
        secrets = pickle.load(open('secrets.bin', 'rb'))
        for image in tqdm(images):
            path = Path(image)
            stem = path.stem
            full_stem = stem + path.suffix
            try:
                analyzed, region = utils.get_secret_string(image)
            except:
                continue
            decoded = secrets[stem]
            analyzed = torch.Tensor(analyzed).cuda()[None]
            decoded = torch.Tensor(decoded).cuda()[None]
            if(args.channel_coding):
                decoded = channel_decoder(decoded)
            decoded = torch.round(torch.clip(decoded, 0, 1))
            match_similarity = cos(analyzed, decoded)
            similarity = match_similarity.item()
            match_score.append(similarity)
    return match_score

def decode(base_path, decoder, channel_decoder, args, cache_secrets):
    match_score = []
    unreadable = 0
    failed = 0
    for df_model in df_models:
        swap_path = df_model[base_path]
        images = glob(f'{swap_path}/*.png')
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
    return match_score

def compile_examples(run, model, residuals=False):
    if(residuals):
        output = Image.new('RGB', (size*4, size*len(df_models)), 'white')
    else:
        output = Image.new('RGB', (size*3, size*len(df_models)), 'white')
    for idx, df_model in enumerate(df_models):
        # clean_path = df_model['resized_path'] + '/' + df_model[example]
        # encoded_path = df_model['output_path'] + '/' + df_model[example]
        # swapped_path = df_model['swap_path'] + '/' + df_model[example]
        # residual_path = df_model['residual_path'] + '/' + df_model[example]
        try:
            clean_path = random.choice(glob(df_model['resized_path'] + '/' + specific))
            encoded_path = random.choice(glob(df_model['output_path'] + '/*.png'))
            swapped_path = random.choice(glob(df_model['swap_path'] + '/*.png'))
            if(residuals):
                residual_path = random.choice(glob(df_model['residual_path'] + '/*.png'))
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
            output.paste(encoded, (size*2, idx*size))
            output.paste(swapped, (size*3, idx*size))
        else:
            output.paste(encoded, (size, idx*size))
            output.paste(swapped, (size*2, idx*size))

    timestamp = str(int(time.time()))
    output.save(f'./results/{run}-{model}-{timestamp}.png')

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

def facestamp_swap_test(run, encoder, decoder, channel_encoder, channel_decoder, args, results):
    cache_secrets = {}
    clear_previous()
    create_encoded(encoder, channel_encoder, args, cache_secrets)
    pre_score = decode('output_path', decoder, channel_decoder, args, cache_secrets)
    results['preswap_match'] = pre_score

    run_faceswap()
    swap_score = decode('swap_path', decoder, channel_decoder, args, cache_secrets)
    results['faceswap_swap_match'] = swap_score
    compile_examples(run, 'faceswap', residuals=False)

    run_compression('enc',80)
    run_compression('enc-swap',80)

    pre_score = decode('output_path', decoder, channel_decoder, args, cache_secrets)
    swap_score = decode('swap_path', decoder, channel_decoder, args, cache_secrets)
    results['preswap_compressed_match'] = pre_score
    results['faceswap_swap_compressed_match'] = swap_score

    run_blur('enc')
    run_blur('enc-swap')

    pre_score = decode('output_path', decoder, channel_decoder, args, cache_secrets)
    swap_score = decode('swap_path', decoder, channel_decoder, args, cache_secrets)
    results['preswap_blurred_compressed_match'] = pre_score
    results['faceswap_swap_blurred_compressed_match'] = swap_score

    run_simswap()
    swap_score = decode('swap_path', decoder, channel_decoder, args, cache_secrets)
    results['simswap_swap_match'] = swap_score
    compile_examples(run, 'simswap', residuals=False)
    # run_compression('enc',40)
    # run_compression('enc-swap',40)

    # pre_score = decode('output_path', decoder, channel_decoder, args, cache_secrets)
    # swap_score = decode('swap_path', decoder, channel_decoder, args, cache_secrets)
    # results['preswap_extreme_compressed_match'] = pre_score
    # results['faceswap_extreme_swap_compressed_match'] = swap_score

    # run_simswap()
    # run_compression(80)
    # swap_score, acc, unreadable, failed = decode_swapped('swap_path', decoder, channel_decoder, args, cache_secrets, acc)
    # swap_distort_score, acc, unreadable, failed = decode_swapped('swap_distort_path', decoder, channel_decoder, args, cache_secrets, acc)
    # results['simswap_swap_match'] = swap_score
    # results['simswap_swap_distort_match'] = swap_distort_score
    # compile_examples(run, 'simswap', residuals=True)

    # results['unreadable'] = unreadable
    # results['failed'] = failed
    # results['acc'] = acc
    return results

def steganogan_swap_test(run, channel_encoder, channel_decoder, args, results):
    cache_secrets = {}
    clear_previous()
    create_encoded_steganogan(channel_encoder, args, cache_secrets)
    pre_score = decode_steganogan('output_path', channel_decoder, args, cache_secrets)
    results['preswap_match'] = pre_score

    run_faceswap()
    swap_score = decode_steganogan('swap_path', channel_decoder, args, cache_secrets)
    results['faceswap_swap_match'] = swap_score
    compile_examples(run, 'faceswap', residuals=False)

    run_compression('enc',80)
    run_compression('enc-swap',80)

    pre_score = decode_steganogan('output_path', channel_decoder, args, cache_secrets)
    swap_score = decode_steganogan('swap_path', channel_decoder, args, cache_secrets)
    results['preswap_compressed_match'] = pre_score
    results['faceswap_swap_compressed_match'] = swap_score

    run_blur('enc')
    run_blur('enc-swap')

    pre_score = decode_steganogan('output_path', channel_decoder, args, cache_secrets)
    swap_score = decode_steganogan('swap_path', channel_decoder, args, cache_secrets)
    results['preswap_blurred_compressed_match'] = pre_score
    results['faceswap_swap_blurred_compressed_match'] = swap_score

    run_simswap()
    swap_score = decode_steganogan('swap_path', channel_decoder, args, cache_secrets)
    results['simswap_swap_match'] = swap_score
    compile_examples(run, 'simswap', residuals=False)

    return results

def stegastamp_swap_test(run, channel_encoder, channel_decoder, args, results):
    cache_secrets = {}
    clear_previous()
    create_encoded_stegastamp(channel_encoder, args, cache_secrets)
    run_faceswap()
    pre_score = decode_stegastamp('output_path', channel_decoder, args, cache_secrets)
    swap_score = decode_stegastamp('swap_path', channel_decoder, args, cache_secrets)
    results['preswap_match'] = pre_score
    results['faceswap_swap_match'] = swap_score
    compile_examples(run, 'faceswap', residuals=False)
    run_compression('enc',80)
    run_compression('enc-swap',80)

    pre_score = decode_stegastamp('output_path', channel_decoder, args, cache_secrets)
    swap_score = decode_stegastamp('swap_path', channel_decoder, args, cache_secrets)
    results['preswap_compressed_match'] = pre_score
    results['faceswap_swap_compressed_match'] = swap_score

    run_blur('enc')
    run_blur('enc-swap')

    pre_score = decode_stegastamp('output_path', channel_decoder, args, cache_secrets)
    swap_score = decode_stegastamp('swap_path', channel_decoder, args, cache_secrets)
    results['preswap_blurred_compressed_match'] = pre_score
    results['faceswap_swap_blurred_compressed_match'] = swap_score

    run_simswap()
    swap_score = decode_stegastamp('swap_path', channel_decoder, args, cache_secrets)
    results['simswap_swap_match'] = swap_score
    compile_examples(run, 'simswap', residuals=False)

    # run_compression('enc',40)
    # run_compression('enc-swap',40)

    # pre_score = decode('output_path', channel_decoder, args, cache_secrets)
    # swap_score = decode('swap_path', channel_decoder, args, cache_secrets)
    # results['preswap_extreme_compressed_match'] = pre_score
    # results['faceswap_extreme_swap_compressed_match'] = swap_score

    return results

def test_models():
    clear_previous()
    for df_model in df_models:
        resized_path = df_model['resized_path']
        source = df_model['source']
        images = glob(f'{resized_path}/*.png')
        random.shuffle(images)
        # size = int(len(images) * 0.02)
        size = 3
        images = images[:size]

        for image in images:
            path = Path(image)
            full_stem = path.stem + '.png'
            with open(image, 'rb') as f:
                with open(f'./test_data/vidtimit/process/{source}-enc/{full_stem}', 'wb') as f2:
                    f2.write(f.read())

        name = df_model['name']
        source = df_model['source']
        extract_cmd = f'/home/luna/anaconda3/envs/facestamp/bin/python3 ./faceswap.py extract -i ./test_data/vidtimit/process/{source}-enc -o ./test_data/vidtimit/process/{source}-enc-extract -D s3fd -A fan -nm none -rf 0 -min 0 -l 0.4 -sz 512 -een 1 -si 0 -L INFO'
        convert_cmd = f'/home/luna/anaconda3/envs/facestamp/bin/python3 ./faceswap.py convert -i ./test_data/vidtimit/process/{source}-enc -o ./test_data/vidtimit/process/{source}-enc-swap -al ./test_data/vidtimit/process/{source}-enc/alignments.fsa -m ./test_data/faceswap_models/{name}_model -c avg-color -M extended -w opencv -osc 100 -l 0.4 -j 0 -L INFO'

        subprocess.run(extract_cmd, shell=True, cwd='./faceswap')
        subprocess.run(convert_cmd, shell=True, cwd='./faceswap')

    compile_examples('test')


if __name__ == '__main__':
    # run_faceswap()
    run_simswap()
    # test_models()
    # clear_previous()
    # prepare_all_resized()

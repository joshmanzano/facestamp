import subprocess 
from glob import glob
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import os
import pathlib
import torch
from torch import nn
import ast
from tqdm import tqdm
import pickle
import json

cos = torch.nn.CosineSimilarity(dim=1)

class ChannelDecoder(nn.Module):
    def __init__(self, secret_size, model_size):
        super().__init__()
        if(model_size == 'small'):
            layers = nn.Sequential(
                nn.Linear(100, secret_size * 6),
                nn.ReLU(),
                nn.Linear(secret_size * 6, secret_size * 3),
                nn.ReLU(),
                nn.Linear(secret_size * 3, secret_size),
                nn.ReLU(),
            )
        elif(model_size == 'big'):
            layers = nn.Sequential(
                nn.Linear(100, secret_size * 6),
                nn.ReLU(),
                nn.Linear(secret_size * 6, secret_size * 4),
                nn.ReLU(),
                nn.Linear(secret_size * 4, secret_size * 2),
                nn.ReLU(),
                nn.Linear(secret_size * 2, secret_size),
                nn.ReLU(),
            )
        self.main = layers

    def forward(self, secret):
        output = self.main(secret)
        return output

class ChannelEncoder(nn.Module):
    def __init__(self, secret_size, model_size):
        super().__init__()
        if(model_size == 'small'):
            layers = nn.Sequential(
                nn.Linear(secret_size, secret_size * 3),
                nn.ReLU(),
                nn.Linear(secret_size * 3, secret_size * 6),
                nn.ReLU(),
                nn.Linear(secret_size * 6, 100),
                nn.ReLU(),
            )
        elif(model_size == 'big'):
            layers = nn.Sequential(
                nn.Linear(secret_size, secret_size * 2),
                nn.ReLU(),
                nn.Linear(secret_size * 2, secret_size * 4),
                nn.ReLU(),
                nn.Linear(secret_size * 4, secret_size * 6),
                nn.ReLU(),
                nn.Linear(secret_size * 6, 100),
                nn.ReLU(),
            )
        self.main = layers

    def forward(self, secret):
        output = self.main(secret)
        return output



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

channel_encoder = ChannelEncoder(14, 'small')
channel_decoder = ChannelDecoder(14, 'small')
channel_encoder.load_state_dict(torch.load('checkpoints/channel_encoder_s22'))
channel_decoder.load_state_dict(torch.load('checkpoints/channel_decoder_s22'))
channel_encoder.eval()
channel_decoder.eval()
channel_encoder = channel_encoder.cuda()
channel_decoder = channel_decoder.cuda()

dataset_size = 1000

def stegastamp_encode(input_dir, output_dir):

    model = 'StegaStamp/saved_models/stegastamp_pretrained'


    files_list = glob(input_dir + '/*.png')
    size = int(len(files_list) * 0.02)
    files_list = files_list[:25]

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model)

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

    width = 400
    height = 400

    secrets = []

    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        size = (width, height)

        analysis = pickle.load(open(f'{input_dir}/analysis.bin','rb'))
        for filename in tqdm(files_list):
            path = pathlib.Path(filename)
            full_stem = path.stem + path.suffix 
            image = Image.open(filename).convert("RGB")
            im = ImageOps.fit(image,size)
            image = np.array(ImageOps.fit(image,size),dtype=np.float32)
            image /= 255.

            try:
                secret = analysis[filename][0]
            except Exception as e:
                print(e)
                continue

            secret = torch.Tensor(secret).cuda()
            secret = channel_encoder(secret)

            secret = torch.round(torch.clip(secret, 0, 1)).tolist()
            secrets.append(secret)

            feed_dict = {input_secret:[secret],
                        input_image:[image]}

            hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)

            rescaled = (hidden_img[0] * 255).astype(np.uint8)
            raw_img = (image * 255).astype(np.uint8)

            residual = residual[0]+.5

            # im = Image.fromarray(np.squeeze(np.array(residual.astype(np.uint8))))
            # im.save(args.save_dir + 'raw_residual/'+save_name+'_residual.png')

            residual = (residual * 255).astype(np.uint8)
            
            im = Image.fromarray(np.array(rescaled))
            new_filename = output_dir + '/' + full_stem
            im.save(new_filename)

            # im = Image.fromarray(np.squeeze(np.array(residual)))
            # im.save(args.save_dir + '/' + full_stem)
    return secrets

def stegastamp_decode(input_dir):
    model = 'StegaStamp/saved_models/stegastamp_pretrained'

    files_list = glob(input_dir + '/*.png')

    unreadable = 0

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    similarities = []

    for filename in tqdm(files_list):
        try:
            analyzed = subprocess.check_output(f'env/bin/python3 analyze_face.py {filename}',shell=True)
            analyzed = analyzed.splitlines()[-1].decode('utf8')
            analyzed = ast.literal_eval(analyzed)
        except Exception as e:
            print(e)
            unreadable += 1
            continue

        image = Image.open(filename).convert("RGB")
        image = np.array(ImageOps.fit(image,(400, 400)),dtype=np.float32)
        image /= 255.

        feed_dict = {input_image:[image]}

        secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]

        secret = torch.Tensor(secret).cuda()
        secret = channel_decoder(secret)
        secret = torch.clip(secret, 0, 1)
        secret = torch.round(secret).cpu()

        analyzed = torch.Tensor(analyzed)

        similarity = cos(secret[None], analyzed[None])
        similarities.append(similarity.item())
    
    return similarities, unreadable


# def run_stegastamp_encode(source):
#     # python3 encode_image.py ../test_data/vidtimit/processfadg0 ../test_data/vidtimit/processfadg-enc
#     stega_cmd = f'/home/luna/anaconda3/envs/stegastamp/bin/python3 encode_image.py ../test_data/vidtimit/process{source} ../test_data/vidtimit/process{source}-enc'

#     subprocess.run(stega_cmd, shell=True, cwd='./StegaStamp')

# def run_stegastamp_decode(source):
#     # python3 encode_image.py ../test_data/vidtimit/processfadg0 ../test_data/vidtimit/processfadg-enc
#     stega_cmd = f'/home/luna/anaconda3/envs/stegastamp/bin/python3 encode_image.py ../test_data/vidtimit/process{source} ../test_data/vidtimit/process{source}-enc'

#     subprocess.run(stega_cmd, shell=True, cwd='./StegaStamp')
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

if __name__ == '__main__':
    cos = torch.nn.CosineSimilarity(dim=1)
    clear_previous()
    similarities = []
    for df_model in df_models:
        source = df_model['source']
        stegastamp_encode(f'./test_data/vidtimit/process{source}-resized',f'./test_data/vidtimit/process{source}-enc')
    #     gt_secrets = torch.Tensor(gt_secrets)
    #     secrets = torch.Tensor(secrets)
    #     similarity = cos(gt_secrets,secrets).mean().item()
    #     similarities.append(similarity)
    # similarities = np.array(similarities)
    subprocess.run('env/bin/python3 faceswap_test.py',shell=True)
    preswap = []
    swapped = []
    ps_std = []
    s_std = []
    unreadable_preswap = 0
    unreadable_swapped = 0
    for df_model in df_models:
        source = df_model['source']
        preswap_temp, preswap_unreadable = stegastamp_decode(f'./test_data/vidtimit/process{source}-enc')
        swapped_temp, swapped_unreadable = stegastamp_decode(f'./test_data/vidtimit/process{source}-enc-swap')
        preswap += preswap_temp
        swapped += swapped_temp
        unreadable_preswap += preswap_unreadable
        unreadable_swapped += swapped_unreadable
    
    print(preswap, swapped)
    results = {
        'preswap': preswap,
        'swapped': swapped,
        'unreadable_preswap': unreadable_preswap,
        'unreadable_swapped':unreadable_swapped 
    }

    pickle.dump(results, open('stegastamp_testing_results.bin','wb'))

    # ps_std = np.array(ps_std)
    # s_std = np.array(s_std)
    # print(preswap.mean())
    # print(preswap.std())
    # print(ps_std.mean())
    # print(unreadable_preswap)

    # print(swapped.mean())
    # print(swapped.std())
    # print(s_std.mean())
    # print(unreadable_swapped)

from glob import glob
import pathlib
from tqdm import tqdm
import pickle
from steganogan.models import SteganoGAN
from numpy import dot
from numpy.linalg import norm
import argparse
import random

global steganogan
steganogan = None

def get_steganogan(architecture):

    global steganogan
    if(steganogan == None):

        steganogan_kwargs = {
            'cuda': True,
            'verbose': False,
            # 'depth': 4,
        }

        steganogan_kwargs['path'] = f'./steganogan/pretrained/{architecture}.steg'
        # steganogan_kwargs['architecture'] = 'dense'
        steganogan = SteganoGAN.load(**steganogan_kwargs)

    return steganogan


def encode(input_dir, output_dir, secret_file, architecture):
    """Given loads a pretrained pickle, encodes the image with it."""
    secrets = pickle.load(open(secret_file,'rb'))
    steganogan = get_steganogan(architecture)
    images = glob(input_dir + '/*.png')
    for image in tqdm(images):
        stem = pathlib.Path(image).stem
        output_image = stem + '.png'
        gt = secrets[stem]
        steganogan.encode(image, f'{output_dir}/{output_image}', gt)

def decode(input_dir, secret_file, architecture):
    """Given loads a pretrained pickle, encodes the image with it."""
    secrets = {}
    steganogan = get_steganogan(architecture)
    images = glob(input_dir + '/*.png')
    for image in tqdm(images):
        decoded = steganogan.decode(image)
        decoded = [int(i) for i in decoded]
        stem = pathlib.Path(image).stem
        secrets[stem] = decoded
    pickle.dump(secrets, open(secret_file, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify GPU and run')
    parser.add_argument('input_dir')
    parser.add_argument('mode')
    parser.add_argument('secret_file')
    parser.add_argument('architecture')
    parser.add_argument('--output_dir')

    args = parser.parse_args()

    if(args.mode == 'encode'):
        encode(args.input_dir, args.output_dir, args.secret_file, args.architecture)
    elif(args.mode == 'decode'):
        decode(args.input_dir, args.secret_file, args.architecture)
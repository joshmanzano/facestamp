from glob import glob
import pathlib
from tqdm import tqdm
import pickle
from steganogan.models import SteganoGAN
from numpy import dot
from numpy.linalg import norm
import argparse

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


def encode(input_dir, output_dir, amount, secret_file, architecture):
    """Given loads a pretrained pickle, encodes the image with it."""
    secrets = pickle.load(open(secret_file,'rb'))
    steganogan = get_steganogan(architecture)
    images = glob(input_dir + '/*.png')
    images = images[:amount]
    for step, image in enumerate(tqdm(images)):
        output_image = pathlib.Path(image).stem + '.png'
        gt = secrets[step]
        steganogan.encode(image, f'{output_dir}/{output_image}', gt)

def decode(input_image, secret_file, architecture):
    """Given loads a pretrained pickle, encodes the image with it."""
    output_secrets = []
    steganogan = get_steganogan(architecture)
    decoded = steganogan.decode(input_image)
    decoded = [int(i) for i in decoded]
    output_secrets.append(decoded)
    pickle.dump(output_secrets, open(secret_file, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify GPU and run')
    parser.add_argument('input_dir')
    parser.add_argument('mode')
    parser.add_argument('secret_file')
    parser.add_argument('architecture')
    parser.add_argument('--amount')
    parser.add_argument('--output_dir')

    args = parser.parse_args()

    if(args.mode == 'encode'):
        encode(args.input_dir, args.output_dir, args.amount, args.secret_file, args.architecture)
    elif(args.mode == 'decode'):
        decode(args.input_dir, args.secret_file, args.architecture)
import bchlib
import glob
import os
from PIL import Image,ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import pickle
import time
import tables
import random

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--secret', type=str, default='Stega!!')
    parser.add_argument('--iterations', type=str, default=1)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '**/*')[:2000]
    else:
        print('Missing input image')
        return

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

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

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    constant_secret = 'secret2'
    wordlist = []

    with open('eff_large_wordlist.txt', 'r') as f:
        for l in f:
            word = l.split()[1]
            index = l.split()[0]
            wordlist.append(word)

    def get_secret():
        secret = random.choice(wordlist)
        
        while len(secret) > 7:
            secret = random.choice(wordlist)
        
        word = secret

        data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')
        ecc = bch.encode(data)
        packet = data + ecc

        packet_binary = ''.join(format(x, '08b') for x in packet)
        secret = [int(x) for x in packet_binary]
        secret.extend([0,0,0,0])
        return secret, word

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        size = (width, height)

        try:
            red_heatmap = np.load(f'heatmaps/red_heatmap_multi_{constant_secret}.npy')
            green_heatmap = np.load(f'heatmaps/green_heatmap_multi_{constant_secret}.npy')
            blue_heatmap = np.load(f'heatmaps/blue_heatmap_multi_{constant_secret}.npy')
            total_heatmap = np.load(f'heatmaps/total_heatmap_multi_{constant_secret}.npy')
        except:
            red_heatmap = np.zeros((0,400,400))
            green_heatmap = np.zeros((0,400,400))
            blue_heatmap = np.zeros((0,400,400))
            total_heatmap = np.zeros((0,400,400))
        
        i = 0

        for filename in files_list:
            
            raw_save_name = filename.split('/')[-1].split('.')[0]
            save_name = filename.split('/')[-1].split('.')[0] + '_' + str(time.time())

            image = Image.open(filename).convert("RGB")

            # im = ImageOps.fit(image,size)
            # im.save(args.save_dir + 'input/'+save_name+'_input.png')

            image = np.array(ImageOps.fit(image,size),dtype=np.float32)
            image /= 255.

            secret, word = get_secret()

            print(f'Iteration: {i}, {word}')

            feed_dict = {input_secret:[secret],
                        input_image:[image]}

            hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)

            rescaled = (hidden_img[0] * 255).astype(np.uint8)
            raw_img = (image * 255).astype(np.uint8)

            residual = residual[0]+.5

            # im = Image.fromarray(np.squeeze(np.array(residual.astype(np.uint8))))
            # im.save(args.save_dir + 'raw_residual/'+save_name+'_residual.png')

            residual = (residual * 255).astype(np.uint8)
            
            # im = Image.fromarray(np.array(rescaled))
            # im.save(args.save_dir + 'hidden/'+save_name+'_hidden.png')

            # im = Image.fromarray(np.squeeze(np.array(residual)))
            # im.save(args.save_dir + 'residual/'+save_name+'_residual.png')

            data = residual
            red = data[:,:,0]
            green = data[:,:,1]
            blue = data[:,:,2]
            red_heatmap = np.concatenate((red_heatmap, red[None]))
            green_heatmap = np.concatenate((green_heatmap, green[None]))
            blue_heatmap = np.concatenate((blue_heatmap, blue[None]))
            total_heatmap = np.concatenate((total_heatmap, blue[None] + green[None] + red[None]))
            print(total_heatmap.shape)

            i += 1
            # red_heatmap += red
            # green_heatmap += green
            # blue_heatmap += blue
            # total_heatmap += red + green + blue

            # red_heatmap -= np.amin(red_heatmap)
            # green_heatmap -= np.amin(green_heatmap)
            # blue_heatmap -= np.amin(blue_heatmap)
            # total_heatmap -= np.amin(total_heatmap)

            if(i % 100 == 0):
                np.save(f'heatmaps/red_heatmap_multi_{constant_secret}.npy', red_heatmap)
                np.save(f'heatmaps/green_heatmap_multi_{constant_secret}.npy', green_heatmap)
                np.save(f'heatmaps/blue_heatmap_multi_{constant_secret}.npy', blue_heatmap)
                np.save(f'heatmaps/total_heatmap_multi_{constant_secret}.npy', total_heatmap)
                print(f'Checkpoint saved! Iteration {i}')

if __name__ == "__main__":
    main()

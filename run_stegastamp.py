from glob import glob
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import pathlib
from tqdm import tqdm
import pickle
import argparse

def encode(input_dir, output_dir, amount, secret_file):

    model = 'StegaStamp/saved_models/stegastamp_pretrained'

    secrets = pickle.load(open(secret_file,'rb'))

    files_list = glob(input_dir + '/*.png')
    files_list = files_list[:amount]

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

    size = (400, 400)

    for step, filename in enumerate(tqdm(files_list)):
        path = pathlib.Path(filename)
        full_stem = path.stem + path.suffix 
        image = Image.open(filename).convert("RGB")
        image = np.array(ImageOps.fit(image,size),dtype=np.float32)
        image /= 255.

        secret = secrets[step] 
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

def decode(input_dir, secret_file):
    model = 'StegaStamp/saved_models/stegastamp_pretrained'

    files_list = glob(input_dir + '/*.png')

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    secrets = []

    for filename in tqdm(files_list):
        image = Image.open(filename).convert("RGB")
        image = np.array(ImageOps.fit(image,(400, 400)),dtype=np.float32)
        image /= 255.

        feed_dict = {input_image:[image]}

        secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]

        secret = np.array(secret)
        secret = np.clip(secret, 0, 1)
        secret = np.round(secret)
        secret = secret.tolist()

        secrets.append(secret)
    
    pickle.dump(secrets, open(secret_file,'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify GPU and run')
    parser.add_argument('input_dir')
    parser.add_argument('mode')
    parser.add_argument('secret_file')
    parser.add_argument('--amount')
    parser.add_argument('--output_dir')

    args = parser.parse_args()

    if(args.mode == 'encode'):
        encode(args.input_dir, args.output_dir, args.amount, args.secret_file)
    elif(args.mode == 'decode'):
        decode(args.input_dir, args.secret_file)
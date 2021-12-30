import glob
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

def main(input_dir):
    model = './saved_models/stegastamp_pretrained'

    files_list = glob.glob(input_dir + '/*.jpg')

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    secrets = []

    for filename in files_list:
        image = Image.open(filename).convert("RGB")
        image = np.array(ImageOps.fit(image,(400, 400)),dtype=np.float32)
        image /= 255.

        feed_dict = {input_image:[image]}

        secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]

        secrets.append(secret)
    
    return secrets


if __name__ == "__main__":
    main()

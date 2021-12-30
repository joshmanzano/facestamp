import typing as T
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import itertools

import torch.nn as nn


config_pretrained_path = "pretrained"


sess = tf.InteractiveSession(graph=tf.Graph())
model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], config_pretrained_path)

variables = tf.global_variables()

for var in variables:
    print(var.name, var.shape)

print("\n\n---------------------------------------------------------------------------\n\n")


def getkey(v):
    n: str = v.name
    path, _ = n.split(":")
    nodes = n.split("/")
    except_last = nodes[:-1]
    return "/".join(except_last)

# TODO check the padding


legos: T.List[nn.Module] = [
    # gen_secret/dense
    nn.Linear(100, 7500),

    # gen_encoder/*
    nn.Conv2d(6, 32, 3, padding=1),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.Conv2d(128, 256, 3, padding=1),
    nn.Conv2d(256, 128, 3, padding=1),
    nn.Conv2d(256, 128, 3, padding=1),
    nn.Conv2d(128, 64, 3, padding=1),
    nn.Conv2d(128, 64, 3, padding=1),
    nn.Conv2d(64, 32, 2),  # Downsampling??? TODO: check the padding
    nn.Conv2d(64, 32, 3, padding=1),
    nn.Conv2d(32, 32, 2),  # Downsampling??? TODO: check the padding
    nn.Conv2d(70, 32, 3, padding=1),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.Conv2d(32, 3, 3, padding=1),

    # gen_stn/*
    nn.Conv2d(3, 32, 3, padding=1),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.Linear(320000, 128),
    nn.Linear(128, 6),

    # gen_decoder/*
    nn.Conv2d(3, 32, 3, padding=1),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Linear(21632, 512),
    nn.Linear(512, 100),
]

for (group_name, values), th_module in zip(itertools.groupby(variables, getkey), legos):
    values: T.List[tf.Tensor] = list(values)
    print(group_name)
    for v in values:
        print("\t", v.name, v.shape)
    for k, v in th_module.state_dict().items():
        print("\t", k, v.size())

    if isinstance(th_module, nn.Linear):
        pass
    if isinstance(th_module, nn.Conv2d):
        # TF conv2d kernel = [kernel_height, kernel_width, in_channels, out_channels]
        # Torch conv2d kernel = [out_channels, in_channels, kernel_height, kernel_width]
        pass

    # TODO: Weight transmutation


# TODO: Do the forward pass

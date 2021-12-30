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
import random
import argparse
import pathlib
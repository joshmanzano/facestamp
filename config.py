
from easydict import EasyDict
import yaml

global args
with open('cfg/setting.yaml', 'r') as f:
    args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

def get_args(key):
    global args
    return args[key]

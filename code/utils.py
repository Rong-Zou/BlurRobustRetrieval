import os
import logging
import datetime
import json
import random
import numpy as np
import torch
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
def get_logger(logdir, mode="train"): 
    logger = logging.getLogger(mode)
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, '{}_{}.log'.format(mode, ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def save_config(logdir, config):
    param_path = os.path.join(logdir, "config.json")
    print("[*] Config Path: %s" % param_path)
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    image_paths = []
    for looproot, _, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.endswith(suffix):
                image_paths.append(os.path.join(looproot, filename))
    return image_paths

def assure_dir(dir: str):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def assure_brand_new_dir(dir: str):
    if os.path.isdir(dir):
        raise ValueError(f"This directory already exists: {dir}.")
    else:
        os.makedirs(dir)
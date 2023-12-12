import os
import random
import numpy as np
import PIL
from PIL import Image
from utils import print_and_log
seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


def get_info(image, background, use_rgb=False,clear_frame_alpha=None):
    # get the alpha channel of the image
    alpha = image[:, :, 3]
    if alpha.dtype == np.uint8:
        alpha = alpha.astype(np.float32) / 255.0 
    # get the mask of the object
    m = alpha > 0
    # get the number of pixels in the object
    toltal_num_pix_obj = np.count_nonzero(m)
    avg_alpha = np.sum(alpha[m]) / toltal_num_pix_obj

    obj_occ = toltal_num_pix_obj / (image.shape[0] * image.shape[1])
    
    # get the number of pixels in the object that are different from the background
    image = image[:, :, :3]
    background = background[:, :, :3]
    if use_rgb:
        diff = np.abs(image - background)
        diff = np.max(diff, axis=2)
    else:
        # convert the image and the background into grayscale images
        image = PIL.Image.fromarray(image).convert('L')
        background = PIL.Image.fromarray(background).convert('L')
        # convert the grayscale images into numpy arrays and compute the difference
        image = np.array(image)
        background = np.array(background)
        diff = np.abs(image - background) # size = (height, width)
        
    diff_avg = np.sum(diff[m]) / toltal_num_pix_obj

    # compute the decrease of alpha relative to the clear frame
    if clear_frame_alpha is not None:
        percent = (clear_frame_alpha - avg_alpha) / clear_frame_alpha
    else:
        percent = 0
    info = {
        "total_num_pix_obj": toltal_num_pix_obj,
        "pix_occupancy": obj_occ,
        "avg_alpha": avg_alpha,

        "rel_alpha_decrease": percent,
        "norm_avg_diff": diff_avg/255.0,
    }        
    return info, diff
    

def obj_occ_judger(path_to_subframes, filter_params, logger):
    # path_to_subframes: a path that stores all the subframes of an instance
    # open all the subframes and compute the object occupancy of each subframe
    # if the object occupancy of any subframe is less than min_occ_thresh, return False
    # otherwise, return True
    # if the difference between the object occupancy of any two consecutive subframes is greater than max_diff_thresh, return False
    # otherwise, return True
    
    # filter_params = {
    # "obj_occ_increment_threshold": obj_occ_increment_threshold,
    # "min_obj_occ_threshold": min_obj_occ_threshold,
    # "min_avg_alpha_threshold": min_avg_alpha_threshold}
    
    min_occ_thresh=filter_params["min_obj_occ_threshold"]
    max_occ_increment_thresh=filter_params["obj_occ_increment_threshold"]
    obj_occs = []
    toltal_num_pix_objs = []
    for subframe_name in os.listdir(path_to_subframes):
        subframe_path = os.path.join(path_to_subframes, subframe_name)
        subframe = np.array(Image.open(subframe_path))
        # get the alpha channel of the image
        alpha = subframe[:, :, 3]
        if alpha.dtype == np.uint8:
            alpha = alpha.astype(np.float32) / 255.0 
        # get the mask of the object
        m = alpha > 0.9
        # get the number of pixels in the object
        toltal_num_pix_obj = np.count_nonzero(m)
        obj_occ = toltal_num_pix_obj / (subframe.shape[0] * subframe.shape[1])
        obj_occs.append(obj_occ)
        toltal_num_pix_objs.append(toltal_num_pix_obj)
    
    min_obj_occ = min(obj_occs)
    max_obj_occ = max(obj_occs)
    diff = max_obj_occ - min_obj_occ

    if min_obj_occ == 0:
        increament = 1
    else:
        increament = diff / min_obj_occ
    
    condition1 = min_obj_occ > min_occ_thresh
    condition2 = increament < max_occ_increment_thresh
    if not condition1:
        # print min_obj_occ is less than min_occ_thresh with 4 decimal places for both values
        print_and_log("min_obj_occ: %.4f is less than required min_occ_thresh: %.4f" % (min_obj_occ, min_occ_thresh), logger)
    if not condition2:
        print_and_log("obj_occ_increament: %.4f is greater than required max_occ_increment_thresh: %.4f" % (increament, max_occ_increment_thresh), logger)
    valid = min_obj_occ > min_occ_thresh and increament < max_occ_increment_thresh
    return max_obj_occ, min_obj_occ, np.argmax(obj_occs), np.argmin(obj_occs), valid
    
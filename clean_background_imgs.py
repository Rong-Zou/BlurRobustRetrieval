# check duplicate images in the dataset, print out the duplicate images and delete them
import os
import numpy as np
from PIL import Image
import tqdm
import json
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from settings import *
import numpy as np

    
img_dir = g_background_path
img_list = os.listdir(img_dir)
img_list.sort()

def check_duplicate_images():
    # for every consecutive image pair, check if they are the same
    # if they are the same, print the name of the image
    duplicate_list = []
    for i in tqdm.tqdm(range(len(img_list)-1)):
        img1 = Image.open(os.path.join(img_dir, img_list[i]))
        img2 = Image.open(os.path.join(img_dir, img_list[i+1]))
        if np.array_equal(img1, img2):
            print(img_list[i], img_list[i+1])
            duplicate_list.append(img_list[i])

    print(len(duplicate_list))
    with open("./datasets/LHQ/duplicate_images.json", "w") as f:
        json.dump(duplicate_list, f, indent=4)
    # remove duplicate images
    for img in duplicate_list:
        os.remove(os.path.join(img_dir, img))

if __name__ == "__main__":
    check_duplicate_images()

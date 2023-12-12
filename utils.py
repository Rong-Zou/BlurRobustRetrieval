import os
import random
import fnmatch
import tempfile
from zipfile import ZipFile
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import logging
import datetime
import json
import bpy

seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)



def estimate_best_init_rot(obj_path, dim_factor=10):  
    context = bpy.context
    scene = context.scene
    
    obj = bpy.ops.import_scene.obj(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    obj.location = [0,0,0]
    obj.scale = [3]*3

    x = obj.dimensions[0]
    y = obj.dimensions[1]
    z = obj.dimensions[2]

    xy = x*y
    xz = x*z
    yz = y*z

    max_area = max([xy, xz, yz])

    if max_area == yz and z > dim_factor * x:
        # rot_y is the key
        # rotation ry could be any value between [-pi/2-pi/3, -pi/2+pi/3] or [pi/2-pi/3, pi/2+pi/3] (in degrees is +/-90 +/-60)
        # randomly sample a value from this even distribution
        ry = np.random.uniform(-np.pi/2-np.pi/3, -np.pi/2+np.pi/3) if np.random.uniform(0,1) < 0.5 else np.random.uniform(np.pi/2-np.pi/3, np.pi/2+np.pi/3)

        # rotation rx could be any value between [-pi/2-pi/3, -pi/2+pi/3] or [pi/2-pi/3, pi/2+pi/3]
        # randomly sample a value from this even distribution
        rx = np.random.uniform(-np.pi/2-np.pi/3, -np.pi/2+np.pi/3) if np.random.uniform(0,1) < 0.5 else np.random.uniform(np.pi/2-np.pi/3, np.pi/2+np.pi/3)
        # rotation rz could be any value between [-pi, pi]
        rz = np.random.uniform(-np.pi, np.pi)
    elif max_area == xy and y > dim_factor * z:
        # rot_x is the key
        # rotation rx could be any value between [-pi, -pi+pi/6] or [pi-pi/6, pi] (convert to degrees is [-180,-150] or [150,180], i.e.180 +/- 30)
        # randomly sample a value from this even distribution
        rx = np.random.uniform(-np.pi, -np.pi+np.pi/6) if np.random.uniform(0,1) < 0.5 else np.random.uniform(np.pi-np.pi/6, np.pi)

        # rotation ry could be any value between [-60,-15] or [15,60] (convert to radians is [-np.pi/3, -np.pi/12] or [np.pi/12, np.pi/3])
        # randomly sample a value from this even distribution
        ry = np.random.uniform(-np.pi/3, -np.pi/12) if np.random.uniform(0,1) < 0.5 else np.random.uniform(np.pi/12, np.pi/3)
        # rotation rz could be any value between [-pi, pi]
        rz = np.random.uniform(-np.pi, np.pi)
    elif max_area == xz and x > dim_factor * y:
        # rot_x is the key
        # rotation rx could be any value between [45, 135], convert to radians is [np.pi/4, 3*np.pi/4]
        # randomly sample a value from this even distribution
        rx = np.random.uniform(np.pi/4, 3*np.pi/4)
        # rotation ry could be any value between [-pi/4, pi/4]
        ry = np.random.uniform(-np.pi/4, np.pi/4)
        # rotation rz could be any value between [-pi, pi]
        rz = np.random.uniform(-np.pi, np.pi)
    else:
        # rotation rx could be any value between [135, 165], convert to radians is [3*np.pi/4, 5*np.pi/6]
        # randomly sample a value from this even distribution
        rx = np.random.uniform(3*np.pi/4, 5*np.pi/6)
        # rotation ry could be any value between [15, 60], convert to radians is [np.pi/12, np.pi/3]
        # randomly sample a value from this even distribution
        ry = np.random.uniform(np.pi/12, np.pi/3)

        rz = np.random.uniform(-np.pi, np.pi)
        
    best_init_rot = [rx,ry,rz]
    size_info = {}
    size_info['xyz'] = [x,y,z]
    size_info['rectangle_area_xy_xz_yz'] = [xy, xz, yz]
    size_info['max_area'] = 'xy' if max_area == xy else 'xz' if max_area == xz else 'yz'

    # delete the object
    bpy.ops.object.delete()
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.images]:
        for block in collection:
            if not block.users:
                collection.remove(block)

    return best_init_rot, size_info

def get_blur_subframe_indexes_for_fixed_num_subframes(n_frames, blur_level_num_frames):
    """Get the subframe indexes for each blur level.
    
    Args:
        n_frames (int): number of subframes
        blur_level_num_frames (list): number of subframes for each blur level
    """
    blur_subframe_indexes = []
    # for each blur level, randomly select a start frame, such that 
    # the start frame + number of subframes for the blur level is less than the total number of subframes
    # then append the start frame and the end frame of this blur level as a tuple (start, end) to the list
    
    # if blur_level_num_frames is a list:
    if isinstance(blur_level_num_frames, list):
        for i in range(len(blur_level_num_frames)):
            start = random.randint(0, n_frames - blur_level_num_frames[i])
            end = start + blur_level_num_frames[i] - 1 # max value is n_frames - 1
            blur_subframe_indexes.append((start, end)) # inclusive on both ends
    # if blur_level_num_frames is an int:
    if isinstance(blur_level_num_frames, int):
        start = random.randint(0, n_frames - blur_level_num_frames)
        end = start + blur_level_num_frames - 1 # max value is n_frames - 1
        blur_subframe_indexes = (start, end) # inclusive on both ends

    return blur_subframe_indexes

def get_blur_subframe_indexes(n_frames, traj_length, intersection_lengths, blur_level_traj_intersection_ratios, logger):
    """Get the subframe indexes for each blur level.
    
    Args:
        n_frames (int): number of subframes
        blur_level_num_frames (list): number of subframes for each blur level
    """
    blur_level_num_frames_range = range_num_frames_for_each_blur_level(n_frames, traj_length, intersection_lengths, blur_level_traj_intersection_ratios, logger)
    if blur_level_num_frames_range is None or len(blur_level_num_frames_range) == 0:
        print_and_log("Warning: Cannot generate complete blur_level_num_frames_range.", logger)
        return [None], [None]
    each_step_traj_length = traj_length / (n_frames - 1)
    # for each blur level, randomly select a value that lies in the range of the number of subframes for the blur level (n, m), inclusive on both ends
    blur_subframe_indexes = []
    blur_ratios = []
    for i in range(len(blur_level_num_frames_range)):
        incomplete = True
        sample_count = 0
        n, m = blur_level_num_frames_range[i]
        a, b = blur_level_traj_intersection_ratios[i]
        while sample_count < 2*n_frames:
            blur_level_num_frames = random.randint(n, m)
            blur_level_subframe_indexes = get_blur_subframe_indexes_for_fixed_num_subframes(n_frames, blur_level_num_frames)
            # get the average intersection length for these subframes, blur_level_subframe_indexes=(start, end), inclusive on both ends
            # notice when start = end, the average intersection length is just the intersection length of that subframe
            # len(intersection_lengths) = n_frames
            blur_level_intersection_lengths = intersection_lengths[blur_level_subframe_indexes[0]:blur_level_subframe_indexes[1]+1]
            avg_intersection_length = np.mean(np.array(blur_level_intersection_lengths))            
            
            # get the traj length for these subframes
            sub_traj_length = (blur_level_num_frames - 1) * each_step_traj_length
            # get the ratio 
            ratio = sub_traj_length / avg_intersection_length
            # if the ratio is not in the range of the blur level, then resample. we want a<=ratio<b
            if a==b:
                if ratio == a:
                    incomplete = False
                    blur_subframe_indexes.append(blur_level_subframe_indexes)
                    blur_ratios.append(ratio)
                    break
            if ratio >= a and ratio < b:
                incomplete = False
                blur_subframe_indexes.append(blur_level_subframe_indexes)
                blur_ratios.append(ratio)
                break
            sample_count += 1
        if incomplete:
            message1 = "Warning: for blur level ratio ({},{}), the range of num_subframes (n, m) is ({},{}), but we failed to sample a complete number of subframes.".format(a, b, n, m)
            message2 = "Maybe the trajectory length is too short to generate a blur level {}".format(i)
            message = [message1, message2]
            print_and_log(message, logger)
            blur_subframe_indexes.append((None, None))
            blur_ratios.append(None)
            
    return blur_subframe_indexes, blur_ratios

def range_num_frames_for_each_blur_level(n_frames, traj_length, intersection_lengths, blur_level_traj_intersection_ratios, logger):
    
    min_intersection_length = np.min(np.array(intersection_lengths))
    max_intersection_length = np.max(np.array(intersection_lengths))
    
    each_step_traj_length = traj_length / (n_frames - 1)
    
    r_num_frames_for_each_blur_level = []
    # for each blur level, compute the range of the number of frames (n, m)
    # such that number_of_steps * each_step_traj_length / avg_intersection_length lies in blur_level_traj_intersection_ratios[i], which is [a, b)
    for i in range(len(blur_level_traj_intersection_ratios)):
        a, b = blur_level_traj_intersection_ratios[i]
        n = int(np.ceil(a * min_intersection_length / each_step_traj_length))
        m = int(np.floor(b * max_intersection_length / each_step_traj_length))
        
        if n>m:
            message = "Warning: for blur level ratio ({},{}), the range of num_subframes (n, m) is ({},{}), n should <=m but n > m.".format(a, b, n, m)
            print_and_log(message, logger)
            return None
        if n+1 >= n_frames:
            message1 = "Error: n_frames {}, for blur level ratio ({},{}), range of num_subframes (n, m), n = {}, which is >= n_frames - 1.".format(n_frames, a, b, n)
            message2 = "The trajectory length is too short to generate a blur level {}".format(i)
            message = [message1, message2]
            print_and_log(message, logger)
            break
        if m+1 > n_frames:
            message = "Warning: n_frames {}, for blur level ratio ({},{}), range of num_subframes (n, m), m = {}, which is > n_frames - 1.".format(n_frames, a, b, m)
            print_and_log(message, logger)
            m = n_frames - 1
        
        r_num_frames_for_each_blur_level.append((n+1, m+1))
        
    return r_num_frames_for_each_blur_level
        
def assure_dir(dir: str):
    # create the directory if it does not exist 
    if not os.path.isdir(dir):
        os.makedirs(dir)

# assure a list of directories
def assure_dirs(dirs: list):
    for dir in dirs:
        assure_dir(dir)
        
def assure_brand_new_dir(dir: str):
    # create the directory if it does not exist and raise an error if it already exists
    if os.path.isdir(dir):
        raise ValueError(f"This directory already exists: {dir}.")
    else:
        os.makedirs(dir)
        
def assure_brand_new_dirs(dirs: list):
    for dir in dirs:
        assure_brand_new_dir(dir)

def get_sub_dirs(parent_dir_path):
    sub_dirs = []
    for file in sorted(os.listdir(parent_dir_path)):
        d = os.path.join(parent_dir_path, file)
        if os.path.isdir(d):
            sub_dirs.append(d)

    return sub_dirs

def get_sub_dirs_from_shapenet(g_shapenet_path, num_categories, method="sorted"):
    # get the subdirectories of g_shapenet_path
    # each subdirectory is a category
    # return a list of subdirectories
    # if method == "sorted", then the subdirectories are sorted by instances_with_textures_info[class_name]["numInstances"]
    # if method == "random", then the subdirectories are shuffled
    

    with open(os.path.join(g_shapenet_path, "instances_with_textures_info.json"), "r") as f:
        instances_with_textures_info = json.load(f)
    
    class_ids = list(instances_with_textures_info.keys())
    

    if method == "sorted":
        class_ids.sort(key=lambda x: instances_with_textures_info[x]["numInstances"], reverse=True)
    elif method == "random":
        random.shuffle(class_ids)
    else:
        raise ValueError("method must be either 'sorted' or 'random'.")
    # take the first num_categories, join with the g_shapenet_path path, and return
    class_ids = class_ids[:num_categories]
    class_names = [instances_with_textures_info[class_id]["name"] for class_id in class_ids]
    sub_dirs = [os.path.join(g_shapenet_path, class_id) for class_id in class_ids]
    return sub_dirs, class_ids, class_names

    
def check_data(folder):
    # for each subfolder in the folder, there are many subsubfolders
    # for each subsubfolder, if it contains 23 files, then it is a complete folder
    # if it contains less or more than 23 files, then it is an incomplete folder
    # save the complete folder names in a list, save the incomplete folder names in another list
    # return the two lists
    complete_folders = []
    incomplete_folders = []
    num_blur_levels_for_incomplete_folders  = []
    for subfolder in os.listdir(folder):
        # subfolder must be a folder and not a file
        if not os.path.isdir(os.path.join(folder, subfolder)):
            continue
        else:
            subfolder_path = os.path.join(folder, subfolder)
            for subsubfolder in os.listdir(subfolder_path):
                subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
                # if len(os.listdir(subsubfolder_path)) == 24:
                #         # append path relative to folder
                #         complete_folders.append(os.path.relpath(subsubfolder_path, folder))
                # else:
                #     incomplete_folders.append(os.path.relpath(subsubfolder_path, folder))
                #     # number of blur levels for incomplete folders is (the number of files in the subsubfolder - 2 ) / 3
                #     num_blur_levels_for_incomplete_folders.append((len(os.listdir(subsubfolder_path)) - 3) / 3)
                for trajectory in os.listdir(subsubfolder_path):
                    trajectory_path = os.path.join(subsubfolder_path, trajectory)
                    if len(os.listdir(trajectory_path)) == 24:
                        # append path relative to folder
                        complete_folders.append(os.path.relpath(trajectory_path, folder))
                    else:
                        incomplete_folders.append(os.path.relpath(trajectory_path, folder))
                        # number of blur levels for incomplete folders is (the number of files in the subsubfolder - 2 ) / 3
                        num_blur_levels_for_incomplete_folders.append((len(os.listdir(trajectory_path)) - 3) / 3)
    return complete_folders, incomplete_folders, num_blur_levels_for_incomplete_folders

def get_logger(logdir): 
    logger = logging.getLogger('emotion')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def print_and_log(message, logger):
    if not isinstance(message, list):
        # print(message)
        logger.info(message)
    else:
        for item in message:
            # print(item)
            logger.info(item)
# https://stackoverflow.com/questions/42710879/write-two-dimensional-list-to-json-file
from _ctypes import PyObj_FromPtr  # see https://stackoverflow.com/a/15012814/355230
import re

class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError('{} wrong type. Only lists and tuples can be wrapped'.format(value))
        self.value = value

class NoIndentWriter(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'  # Unique string pattern of NoIndent object ids.
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding NoIndent wrapped values.
        ignore = {'cls', 'indent'}

        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super(NoIndentWriter, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                    else super(NoIndentWriter, self).default(obj))

    def iterencode(self, obj, **kwargs):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.

        # Replace any marked-up NoIndent wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super(NoIndentWriter, self).iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_repr = json.dumps(no_indent.value, **self._kwargs)
                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                encoded = encoded.replace(
                            '"{}"'.format(format_spec.format(id)), json_repr)

            yield encoded

class ZipLoader:
    def __init__(self, zip, filter="*[!/]", balance_subdirs=False):
        self.zip = ZipFile(zip)
        self.names = fnmatch.filter(self.zip.namelist(), filter)
        self.dirtree = None

        if balance_subdirs:
            # create directory tree of zip contents
            dict_tree = lambda: defaultdict(dict_tree)
            self.dirtree = dict_tree()
            for name in self.names:
                node = self.dirtree
                for d in name.split("/")[:-1]:
                    node = node[d]
                node[name] = None

    @contextmanager
    def as_tempfile(self, name):
        """Extract a file from the zip file to a temporary file.
        
        Args:
            name (str): name of the file in the zip file
        
        Yields:
            str: path to the temporary file
        
        """
        _, ext = os.path.splitext(name)
        fd, path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, "wb") as f:
            f.write(self.zip.read(name))
        try:
            yield path
        finally:
            os.remove(path)

    def get_random(self):
        """Get a random file from the zip file.
        If balance_subdirs is True, then the file is sampled from the directory
        tree of the zip file.
        
        Returns:
            str: name of the file
        
        """
        if self.dirtree:
            # randomly sample at every level of directory tree
            node = self.dirtree
            while True:
                name = random.choice(list(node.keys()))
                node = node[name]
                if not node:
                    # leaf node
                    return name
        return random.choice(self.names)

    def get_random_seq(self, length):
        """
        Get a random sequence of files from the zip file.
        If balance_subdirs is True, then the sequence is sampled from the
        directory tree of the zip file. Otherwise, the sequence is sampled
        uniformly from the list of files in the zip file. If the sequence is
        longer than the number of files in the zip file, then an error is
        raised. """
        for _ in range(10000):
            seed = self.get_random()
            node = self.dirtree
            for d in seed.split("/")[:-1]:
                node = node[d]
            names = sorted(node.keys())
            if len(names) >= length:
                start = random.randint(0, len(names) - length)
                return names[start : start + length]
        raise ValueError(f"Failed to get random sequence of length {length}.")

if __name__ == "__main__":
    # g_shapenet_path = "/local/home/ronzou/euler/datasets/ShapeNetCore.v2"
    # sub_dirs = get_sub_dirs_from_shapenet(g_shapenet_path, 10, method="random")
    # print(sub_dirs)
    g_outdata_path = "/local/home/ronzou/euler/blur_data_final1"
    complete_folders, incomplete_folders, num_blur_levels_for_incomplete_folders = check_data(g_outdata_path)
    sequences = {"complete": [], "incomplete": [], "failed": []}
    for folder in complete_folders:
        sequences["complete"].append([folder, 7])
    for i in range(len(incomplete_folders)):
        sequences["incomplete"].append([incomplete_folders[i], num_blur_levels_for_incomplete_folders[i]])
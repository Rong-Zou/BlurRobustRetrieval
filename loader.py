import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import os
import random
from PIL import Image
import json
import copy
import cv2
import math
from models import *
from utils import *
import collections
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from settings import g_data_dir, g_distractor_dir
import bisect
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# set TF_ENABLE_ONEDNN_OPTS to 0 will 
from tqdm import tqdm
# manual seed
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic=True
os.environ['PYTHONHASHSEED'] = str(seed)
# os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    
class dataset_database_query(Dataset):
    def __init__(self, instance_paths, 
                 normalize=True, transform=None,
                 database_ratio=11/12,
                 take_blur_levels=[0,1,2,3,4,5],
                 save_load_imgs_dir=None,
                 ): 
        super().__init__()
        # manual seed
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic=True
        os.environ['PYTHONHASHSEED'] = str(seed)
        # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
        
        self.instance_paths = instance_paths
        # for each path in instance_paths, split by 'synthetic_data' and take the second part and join it with g_data_dir
        self.instance_paths = [os.path.join(g_data_dir, path.split('synthetic_data/')[1]) for path in self.instance_paths]

        self.database_ratio = database_ratio
        

        # sort blur_levels_alpha from large to small
        if isinstance(take_blur_levels, int):
            take_blur_levels = [take_blur_levels]
        self.blur_levels_alpha_large_to_small = sorted(take_blur_levels, reverse=True)
        self.num_blur_levels = len(self.blur_levels_alpha_large_to_small)

        self.bins = np.round(np.arange(0, 1.1, 0.1),3).tolist()
        
        self.instance_names = [os.path.basename(instance_path) for instance_path in instance_paths]
        self.instance_names = sorted(list(set(self.instance_names)))
        num_instances = len(self.instance_names)
        # map the unique instance ids to an integer from 0 to num_instances
        # map the unique instance ids to a one-hot label
        self.instance_id_to_label = {instance_id: np.eye(num_instances)[i] for i, instance_id in enumerate(self.instance_names)}

        
        self.normalize = normalize
        self.transform = transform
        
        self.save_load_imgs_dir = save_load_imgs_dir
        assure_dir(self.save_load_imgs_dir)
        
        saved_ins_im_db_mixed_path = os.path.join(save_load_imgs_dir, 'instance_images_db_mixed.npy')
        saved_ins_im_db_v2_mixed_path = os.path.join(save_load_imgs_dir, 'instance_images_db_v2_mixed.npy')
        saved_ins_im_q_mixed_path = os.path.join(save_load_imgs_dir, 'instance_images_q_mixed.npy')
        saved_ins_im_db_only_sharp_path = os.path.join(save_load_imgs_dir, 'instance_images_db_only_sharp.npy')
        saved_ins_im_q_only_sharp_path = os.path.join(save_load_imgs_dir, 'instance_images_q_only_sharp.npy')

        # if there are no saved files, then load the images and save them
        if not os.path.exists(saved_ins_im_db_mixed_path) or not os.path.exists(saved_ins_im_q_mixed_path) or not os.path.exists(saved_ins_im_db_only_sharp_path) or not os.path.exists(saved_ins_im_q_only_sharp_path) or not os.path.exists(saved_ins_im_db_v2_mixed_path):
            self.instance_traj_alpha_stats  = np.load(os.path.join(g_data_dir, 'stats/traj_stats', 'instance_traj_alpha_stats.npy'), allow_pickle=True).item()
            self.instance_traj_alpha_counts = np.load(os.path.join(g_data_dir, 'stats/traj_stats', 'instance_traj_alpha_counts.npy'), allow_pickle=True).item()
            # for k v in instance_traj_alpha_stats.items(), v is a list, keep only elements whose index is in blur_levels_alpha
            # instance_traj_alpha_stats = {k: [v[i] for i in blur_levels_alpha] for k, v in instance_traj_alpha_stats.items()}
            
            # erosion kernel size is 3
            avg_alphas_erosion = np.load(os.path.join(g_data_dir,'stats/erosion', 'avg_alphas_erosion_by_blurlevel.npz'), allow_pickle=True)
            avg_alphas_erosion = [avg_alphas_erosion[i] for i in avg_alphas_erosion]

            with open(os.path.join(g_data_dir,'stats/erosion', 'img_paths_erosion_by_blurlevel.json'), 'r') as f:
                img_paths = json.load(f)

            self.all_avg_alpha = avg_alphas_erosion
            self.all_img_paths = img_paths
            
            self.instance_images_db = {instance_name: [] for instance_name in self.instance_names}
            self.instance_images_db_v2 = {instance_name: [] for instance_name in self.instance_names}
            self.instance_images_q = {instance_name: [] for instance_name in self.instance_names}
            self.instance_images_db_only_sharp = {instance_name: [] for instance_name in self.instance_names}
            self.instance_images_q_only_sharp = {instance_name: [] for instance_name in self.instance_names}
            
            self.all_instance_images_db = []
            self.all_instance_images_db_v2 = []
            self.all_instance_images_q = []
            self.all_instance_images_db_only_sharp = []
            self.all_instance_images_q_only_sharp = []

            # for instance_path in self.instance_paths:             
            for instance_path in tqdm(self.instance_paths):
            # when use alpha to define blur level
                this_instance_db, this_instance_db_v2, this_instance_q, this_instance_db_only_sharp, this_instance_q_only_sharp = self.get_instance_imgs_db_q(instance_path, type='rendered', method='even')
                instance_name = os.path.basename(instance_path)
                
                self.instance_images_db[instance_name] = this_instance_db
                self.instance_images_db_v2[instance_name] = this_instance_db_v2
                self.instance_images_q[instance_name] = this_instance_q
                self.instance_images_db_only_sharp[instance_name] = this_instance_db_only_sharp
                self.instance_images_q_only_sharp[instance_name] = this_instance_q_only_sharp
                
                self.all_instance_images_db += this_instance_db
                self.all_instance_images_db_v2 += this_instance_db_v2
                self.all_instance_images_q  += this_instance_q
                self.all_instance_images_db_only_sharp += this_instance_db_only_sharp
                self.all_instance_images_q_only_sharp  += this_instance_q_only_sharp
            
            # save all the above lists
            np.save(saved_ins_im_db_mixed_path, self.instance_images_db)
            np.save(saved_ins_im_db_v2_mixed_path, self.instance_images_db_v2)
            np.save(saved_ins_im_q_mixed_path, self.instance_images_q)
            np.save(saved_ins_im_db_only_sharp_path, self.instance_images_db_only_sharp)
            np.save(saved_ins_im_q_only_sharp_path, self.instance_images_q_only_sharp)
            
            del self.instance_traj_alpha_stats
            del self.instance_traj_alpha_counts
            del self.all_avg_alpha
            del self.all_img_paths
        
        else: # load the above lists
            self.instance_images_db = np.load(saved_ins_im_db_mixed_path, allow_pickle=True).item()
            self.instance_images_db_v2 = np.load(saved_ins_im_db_v2_mixed_path, allow_pickle=True).item()
            self.instance_images_q = np.load(saved_ins_im_q_mixed_path, allow_pickle=True).item()
            self.instance_images_db_only_sharp = np.load(saved_ins_im_db_only_sharp_path, allow_pickle=True).item()
            self.instance_images_q_only_sharp = np.load(saved_ins_im_q_only_sharp_path, allow_pickle=True).item()
            self.all_instance_images_db = []
            self.all_instance_images_db_v2 = []
            self.all_instance_images_q = []
            self.all_instance_images_db_only_sharp = []
            self.all_instance_images_q_only_sharp = []
            for instance_name in self.instance_names:
                self.all_instance_images_db += self.instance_images_db[instance_name]
                self.all_instance_images_db_v2 += self.instance_images_db_v2[instance_name]
                self.all_instance_images_q  += self.instance_images_q[instance_name]
                self.all_instance_images_db_only_sharp += self.instance_images_db_only_sharp[instance_name]
                self.all_instance_images_q_only_sharp  += self.instance_images_q_only_sharp[instance_name]
        
        
        self.dataset_type = None
        self.data = None
        

    def get_instance_imgs_db_q(self, instance_path, type='rendered', method='even'):
        instance_images_db = []
        instance_alphas_db = [] 
        instance_blur_levels_db = []

        instance_images_db_only_sharp = []
        instance_alphas_db_only_sharp = []
        instance_blur_levels_db_only_sharp = []

        instance_images_db_v2 = []
        instance_alphas_db_v2 = []
        instance_blur_levels_db_v2 = []
        
        instance_images_q = []
        instance_alphas_q = []
        instance_blur_levels_q = []

        instance_images_q_only_sharp = []
        instance_alphas_q_only_sharp = []
        instance_blur_levels_q_only_sharp = []



        instance_name = os.path.basename(instance_path)
        label = self.instance_id_to_label[instance_name]
        # for database
        if method == 'even':
            # get the alpha stats and counts of this instance
            instance_traj_alpha_stats_this = self.instance_traj_alpha_stats[instance_name]
            instance_traj_alpha_counts_this = self.instance_traj_alpha_counts[instance_name]
            # sort the two lists by counts, we also need the indices of the sorted lists
            instance_traj_alpha_counts_this_sorted, blur_levels_sorted = torch.sort(torch.tensor(instance_traj_alpha_counts_this), descending=False) # 9857264310
            instance_traj_alpha_stats_this_sorted = [instance_traj_alpha_stats_this[i] for i in blur_levels_sorted]
            # get the idxes of the blur levels in blur_levels_sorted that are in blur_levels_alpha
            idxes = [i for i in range(len(blur_levels_sorted)) if blur_levels_sorted[i] in self.blur_levels_alpha_large_to_small]
            # keep only the blur levels in blur_levels_alpha
            instance_traj_alpha_counts_this_sorted = [instance_traj_alpha_counts_this_sorted[i].item() for i in idxes]
            instance_traj_alpha_stats_this_sorted = [instance_traj_alpha_stats_this_sorted[i] for i in idxes]
            blur_levels_sorted = [blur_levels_sorted[i].item() for i in idxes]
            
            take_trajs_for_diff_blur_levels = {blur_level: [] for blur_level in self.blur_levels_alpha_large_to_small}
            take_num_trajs_for_diff_blur_levels = {blur_level: 0 for blur_level in self.blur_levels_alpha_large_to_small}

            """query"""
            take_trajs_for_diff_blur_levels_q = {blur_level: [] for blur_level in self.blur_levels_alpha_large_to_small}
            take_num_trajs_for_diff_blur_levels_q = {blur_level: 0 for blur_level in self.blur_levels_alpha_large_to_small}

            # The following code is for database
            num_traj = max(instance_traj_alpha_counts_this)
            num_traj_each_blur_level = [math.floor(num_traj / self.num_blur_levels)] * self.num_blur_levels
            remains = num_traj % self.num_blur_levels
            # ranomly choose remains blur levels and add 1 to num_traj_each_blur_level
            if remains > 0:
                take_blur_levels = random.sample(range(self.num_blur_levels), remains)
                for i in take_blur_levels:
                    num_traj_each_blur_level[i] += 1

            for i in range(self.num_blur_levels):
                cur_blur_level = blur_levels_sorted[i]
                trajs_this_blur_level = instance_traj_alpha_stats_this_sorted[i]
                traj_count_this_blur_level = len(trajs_this_blur_level)
                
                if traj_count_this_blur_level < num_traj_each_blur_level[i]:
                    take_trajs_this_blur_level = trajs_this_blur_level

                    if i < self.num_blur_levels - 1:
                        # how less trajs do we take
                        spare_num_traj = num_traj_each_blur_level[i] - traj_count_this_blur_level
                        # distribute this num trajs evenly to j>i num_traj_each_blur_level[j]
                        for j in range(i+1, self.num_blur_levels):
                            num_traj_each_blur_level[j] += spare_num_traj // (self.num_blur_levels - i - 1)
                        spare_num_traj = spare_num_traj % (self.num_blur_levels - i - 1)
                        # randomly choose spare_num_traj blur levels from i+1 to self.num_blur_levels-1 and add 1 
                        if spare_num_traj > 0:
                            take_blur_levels = random.sample(range(i+1, self.num_blur_levels), spare_num_traj)
                            for j in take_blur_levels:
                                num_traj_each_blur_level[j] += 1

                elif traj_count_this_blur_level > num_traj_each_blur_level[i]:
                    take_trajs_this_blur_level = random.sample(trajs_this_blur_level, num_traj_each_blur_level[i])
                
                else:
                    take_trajs_this_blur_level = trajs_this_blur_level

                # if non zero
                if len(take_trajs_this_blur_level) > 0:
                    # remove the chosen trajs from instance_traj_alpha_stats_this_sorted[j] for all j > i
                    for j in range(i+1, self.num_blur_levels):
                        instance_traj_alpha_stats_this_sorted[j] = sorted(list(set(instance_traj_alpha_stats_this_sorted[j]) - set(take_trajs_this_blur_level)))

                    """query"""
                    # take some trajs from this blur level for query
                    num_q = math.ceil(len(take_trajs_this_blur_level) * (1 - self.database_ratio))
                    take_trajs_this_blur_level_q = random.sample(take_trajs_this_blur_level, num_q)
                    take_trajs_this_blur_level = sorted(list(set(take_trajs_this_blur_level) - set(take_trajs_this_blur_level_q)))
                    take_trajs_for_diff_blur_levels_q[cur_blur_level] = take_trajs_this_blur_level_q
                    take_num_trajs_for_diff_blur_levels_q[cur_blur_level] = num_q
                    """query"""

                    take_trajs_for_diff_blur_levels[cur_blur_level] = take_trajs_this_blur_level
                    take_num_trajs_for_diff_blur_levels[cur_blur_level] = len(take_trajs_this_blur_level)

            """database"""
            # get the images, one image from each traj
            for i in range(self.num_blur_levels):
                cur_blur_level = blur_levels_sorted[i]
                take_trajs_this_blur_level = take_trajs_for_diff_blur_levels[cur_blur_level]
                take_num_trajs_this_blur_level = take_num_trajs_for_diff_blur_levels[cur_blur_level]
                # get the images of this blur level
                if take_num_trajs_this_blur_level > 0:
                    # get the images of this blur level
                    # get the image paths of all the trajs in take_trajs_this_blur_level
                    take_img_paths_this_blur_level = []
                    take_img_alphas_this_blur_level = []
                    for traj in take_trajs_this_blur_level:
                        traj_path = os.path.join(instance_path, traj).split('/')[-3:] # ['02691156', '1c93b0eb9c313f5d9a6e43b878d5b335', '000']
                        traj_path = os.path.join(*traj_path)
                        # get the image paths of all the images in this traj
                        im_paths = [os.path.join(traj_path, str(i)+'_blurred.png') for i in range(0, 11)] 
                        
                        paths_ = []
                        alphas_ = []
                        for im_path in im_paths:
                            try:
                                im_idx = self.all_img_paths[cur_blur_level].index(im_path)
                            except:
                                continue
                            im_alpha = self.all_avg_alpha[cur_blur_level][im_idx]
                            paths_.append(im_path)
                            alphas_.append(im_alpha)
                        # randomly choose one image
                        idx_ = random.choice(range(len(paths_)))

                        take_img_alphas_this_blur_level.append(alphas_[idx_])
                        im_idx_in_traj = paths_[idx_].split('/')[-1].split('_')[0]
                        take_img_paths_this_blur_level.append(os.path.join(instance_path, traj, str(im_idx_in_traj)+'_'+type+'.png'))

                        """database with only sharp"""
                        im_idx = self.all_img_paths[0].index(im_paths[0])
                        instance_images_db_only_sharp.append(os.path.join(instance_path, traj, str(0)+'_'+type+'.png'))
                        instance_alphas_db_only_sharp.append(self.all_avg_alpha[0][im_idx])
                        instance_blur_levels_db_only_sharp.append(0) # must be 0

                    # add the images of this blur level to instance_images
                    instance_images_db.extend(take_img_paths_this_blur_level)
                    instance_alphas_db.extend(take_img_alphas_this_blur_level)
                    instance_blur_levels_db.extend([cur_blur_level] * len(take_img_paths_this_blur_level))

            """database version 2 - per blur level database"""
            # get the images, one image for each blur level in each traj, namely num_blur_levels images for each traj
            for i in range(self.num_blur_levels):
                cur_blur_level = blur_levels_sorted[i]
                take_trajs_this_blur_level = take_trajs_for_diff_blur_levels[cur_blur_level]
                take_num_trajs_this_blur_level = take_num_trajs_for_diff_blur_levels[cur_blur_level]
                # get the images
                if take_num_trajs_this_blur_level > 0:
                    # get the images of this blur level
                    # get the image paths of all the trajs in take_trajs_this_blur_level
                    take_img_paths_these_trajs = []
                    take_img_alphas_these_trajs = []
                    take_img_blur_levels_these_trajs = []
                    for traj in take_trajs_this_blur_level:
                        traj_path = os.path.join(instance_path, traj).split('/')[-3:]
                        traj_path = os.path.join(*traj_path)
                        # get the image paths of all the images in this traj
                        im_paths = [os.path.join(traj_path, str(i)+'_blurred.png') for i in range(0, 11)]
                        
                        im_alphas=[]
                        im_blur_levels=[]
                        for im_path in im_paths:
                            for bl in range(len(self.bins)-1):
                                try:
                                    im_idx = self.all_img_paths[bl].index(im_path)
                                except:
                                    continue
                                im_alpha = self.all_avg_alpha[bl][im_idx]
                                im_alphas.append(im_alpha)
                                im_blur_levels.append(bl)
                                break
                        
                        """take only one image for each blur level in blur_levels_alpha"""
                        for blur_level in self.blur_levels_alpha_large_to_small:
                            idxes_this_blur_level = [i for i in range(len(im_blur_levels)) if im_blur_levels[i] == blur_level]
                            if len(idxes_this_blur_level) > 0:
                                # randomly choose one image from idxes_this_blur_level
                                idx = random.choice(idxes_this_blur_level)
                                take_img_paths_these_trajs.append(os.path.join(instance_path, traj, str(idx)+'_'+type+'.png'))
                                take_img_alphas_these_trajs.append(im_alphas[idx])
                                take_img_blur_levels_these_trajs.append(blur_level)

                    # add the images of this blur level to instance_images
                    instance_images_db_v2.extend(take_img_paths_these_trajs)
                    instance_alphas_db_v2.extend(take_img_alphas_these_trajs)
                    instance_blur_levels_db_v2.extend(take_img_blur_levels_these_trajs)

                
            """query"""
            # get the images, one image for each blur level in each traj, namely num_blur_levels images for each traj
            for i in range(self.num_blur_levels):
                cur_blur_level = blur_levels_sorted[i]
                take_trajs_this_blur_level = take_trajs_for_diff_blur_levels_q[cur_blur_level]
                take_num_trajs_this_blur_level = take_num_trajs_for_diff_blur_levels_q[cur_blur_level]
                # get the images
                if take_num_trajs_this_blur_level > 0:
                    # get the images of this blur level
                    # get the image paths of all the trajs in take_trajs_this_blur_level
                    take_img_paths_these_trajs = []
                    take_img_alphas_these_trajs = []
                    take_img_blur_levels_these_trajs = []
                    for traj in take_trajs_this_blur_level:
                        traj_path = os.path.join(instance_path, traj).split('/')[-3:]
                        traj_path = os.path.join(*traj_path)
                        # get the image paths of all the images in this traj
                        im_paths = [os.path.join(traj_path, str(i)+'_blurred.png') for i in range(0, 11)]
                        
                        im_alphas=[]
                        im_blur_levels=[]
                        for im_path in im_paths:
                            for bl in range(len(self.bins)-1):
                                try:
                                    im_idx = self.all_img_paths[bl].index(im_path)
                                except:
                                    continue
                                im_alpha = self.all_avg_alpha[bl][im_idx]
                                im_alphas.append(im_alpha)
                                im_blur_levels.append(bl)
                                break

                        """query"""
                        """take only one image for each blur level in blur_levels_alpha"""
                        
                        for blur_level in self.blur_levels_alpha_large_to_small:
        
                            idxes_this_blur_level = [i for i in range(len(im_blur_levels)) if im_blur_levels[i] == blur_level]
                            if len(idxes_this_blur_level) > 0:
                                # randomly choose one image from idxes_this_blur_level
                                idx = random.choice(idxes_this_blur_level)
                                take_img_paths_these_trajs.append(os.path.join(instance_path, traj, str(idx)+'_'+type+'.png'))
                                take_img_alphas_these_trajs.append(im_alphas[idx])
                                take_img_blur_levels_these_trajs.append(blur_level)
                        """query only sharp"""
                        instance_images_q_only_sharp.append(os.path.join(instance_path, traj, str(0)+'_'+type+'.png'))
                        instance_alphas_q_only_sharp.append(im_alphas[0])
                        instance_blur_levels_q_only_sharp.append(im_blur_levels[0]) # must be 0

                    # add the images of this blur level to instance_images
                    instance_images_q.extend(take_img_paths_these_trajs)
                    instance_alphas_q.extend(take_img_alphas_these_trajs)
                    instance_blur_levels_q.extend(take_img_blur_levels_these_trajs)

        else:
            raise NotImplementedError
        
        # zip into a list of tuples (image_path, alpha, blur_level, label)
        database = list(zip(instance_images_db, instance_alphas_db, instance_blur_levels_db, [label] * len(instance_images_db)))
        database_v2 = list(zip(instance_images_db_v2, instance_alphas_db_v2, instance_blur_levels_db_v2, [label] * len(instance_images_db_v2)))
        query = list(zip(instance_images_q, instance_alphas_q, instance_blur_levels_q, [label] * len(instance_images_q)))

        database_only_sharp = list(zip(instance_images_db_only_sharp, instance_alphas_db_only_sharp, instance_blur_levels_db_only_sharp, [label] * len(instance_images_db_only_sharp)))
        query_only_sharp = list(zip(instance_images_q_only_sharp, instance_alphas_q_only_sharp, instance_blur_levels_q_only_sharp, [label] * len(instance_images_q_only_sharp)))

        return database, database_v2, query, database_only_sharp, query_only_sharp

    
    def set_dataset_type(self, dataset_type):

        assert len(dataset_type) == 2
        # convert to list
        dataset_type = list(dataset_type)
        # dataset_type[0] must be "database" or "db", or "query" or "q"
        assert dataset_type[0] in ['database', 'db', 'query', 'q', 'database_v2', 'db_v2']
        # if it is "db" or "q", change it to "database" or "query"
        if dataset_type[0] == 'db':
            dataset_type[0] = 'database'
        elif dataset_type[0] == 'q':
            dataset_type[0] = 'query'
        elif dataset_type[0] == 'db_v2':
            dataset_type[0] = 'database_v2'

        # dataset_type[1] must be "sharp" or "s", or "mixed" or "m"
        assert dataset_type[1] in ['sharp', 's', 'mixed', 'm']
        # if it is "s" or "m", change it to "sharp" or "mixed"
        if dataset_type[1] == 's':
            dataset_type[1] = 'sharp'
        elif dataset_type[1] == 'm':
            dataset_type[1] = 'mixed'
        # set self.dataset_type
        dataset_type = tuple(dataset_type)
        self.dataset_type = dataset_type
        # set self.data
        if dataset_type == ("database", "sharp"):
            self.data = self.all_instance_images_db_only_sharp
        elif dataset_type == ("database", "mixed"):
            self.data = self.all_instance_images_db
        elif dataset_type == ("database_v2", "sharp"):
            self.data = self.all_instance_images_db_only_sharp
        elif dataset_type == ("database_v2", "mixed"):
            self.data = self.all_instance_images_db_v2
        elif dataset_type == ("query", "sharp"):
            self.data = self.all_instance_images_q_only_sharp
        elif dataset_type == ("query", "mixed"):
            self.data = self.all_instance_images_q
        
        print("Setting data type to {}.".format(dataset_type))
        print("The data contain {} images.".format(len(self.data)))


    def get_bbox(self, image_path):
        # first get the mask
        mask_path = image_path.replace("_rendered.png", "_blurred.png")
        mask = torch.from_numpy(np.array(Image.open(mask_path))).float().permute(2, 0, 1) # [4 x H x W]
        mask = torch.where(mask[3,:,:] > 0, torch.tensor(255.0), torch.tensor(0.0)).unsqueeze(0)
        
        # find the first and last non-zero row and column
        col_min_idx = torch.nonzero(mask[0,:,:].sum(dim=0) > 0)[0]
        col_max_idx = torch.nonzero(mask[0,:,:].sum(dim=0) > 0)[-1]
        row_min_idx = torch.nonzero(mask[0,:,:].sum(dim=1) > 0)[0]
        row_max_idx = torch.nonzero(mask[0,:,:].sum(dim=1) > 0)[-1]
        
        # get the bbox [center_x, center_y, width, height]
        center_x = (col_min_idx + col_max_idx) / 2.0
        norm_center_x = center_x / mask.shape[2]
        center_y = (row_min_idx + row_max_idx) / 2.0
        norm_center_y = center_y / mask.shape[1]
        width = col_max_idx - col_min_idx
        norm_width = width / mask.shape[2]
        height = row_max_idx - row_min_idx
        norm_height = height / mask.shape[1]
        bbox_denorm = [col_min_idx, row_min_idx, col_max_idx, row_max_idx]
        # bbox = torch.tensor([norm_center_x, norm_center_y, norm_width, norm_height])

        # get the bbox [x1, y1, x2, y2], normalized
        # x1 = col_min_idx / mask.shape[2]
        # y1 = row_min_idx / mask.shape[1]
        # x2 = col_max_idx / mask.shape[2]
        # y2 = row_max_idx / mask.shape[1]
        # bbox = torch.tensor([x1, y1, x2, y2])
        
        return bbox_denorm


    def __getitem__(self, idx):

        img_path, alpha, blur_level, label = self.data[idx]
        img_path = os.path.join(g_data_dir, img_path.split('synthetic_data/')[1])

        im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        
        img = torch.from_numpy(img).float().permute(2, 0, 1) # [3 x H x W]

        # normalize the image
        if self.normalize:
            img = img / 255.0

        # transform the image
        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_path, blur_level
        
    def __len__(self):
        if self.dataset_type is None:
            raise ValueError("You need to set self.dataset_type before getting the length. Use self.set_dataset_type().")

        return len(self.data)

              
class dataset_distractor(Dataset):
    def __init__(self, data_dir=None, stats_dir=None, 
                 normalize=True, transform=None,
                 ): 
        super().__init__()
        # manual seed
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic=True
        os.environ['PYTHONHASHSEED'] = str(seed)
        # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
        
        self.data_dir = data_dir
        self.stats_dir = stats_dir
        self.normalize = normalize
        self.transform = transform

        if self.data_dir is None:
            self.data_dir = g_distractor_dir

        if self.stats_dir is None:
            self.stats_dir = os.path.join(g_distractor_dir, 'stats', 'erosion')


        self.All_alpha_erosion = np.load(os.path.join(self.stats_dir, 'alphas_erosion.npy'))
        self.All_blur_levels = np.load(os.path.join(self.stats_dir, 'blur_levels.npy'))
        with open(os.path.join(self.stats_dir, 'img_paths.json'), 'r') as f:
            self.All_img_paths = json.load(f)
        
        for i in range(len(self.All_img_paths)):
            self.All_img_paths[i] = self.All_img_paths[i].replace("_blurred.png", "_rendered.png")
        
        self.all_alpha_erosion = None
        self.all_blur_levels = None
        self.all_img_paths = None
        self.dataset_type = None

    
    def set_dataset_type(self, dataset_type):
        # set self.dataset_type
        self.dataset_type = dataset_type
        # set self.data
        if dataset_type == "a" or dataset_type == "all":
            idxes = None
        elif dataset_type == "s" or dataset_type == "sharp": 
            idxes = [i for i in range(len(self.All_alpha_erosion)) if self.All_img_paths[i].split('/')[-1].split('_')[0] == '0']
        elif isinstance(dataset_type, tuple) or isinstance(dataset_type, list):
            dataset_type = list(dataset_type)
            idxes = [i for i in range(len(self.All_alpha_erosion)) if self.All_blur_levels[i] in dataset_type]
        elif isinstance(dataset_type, int) or (dataset_type.isdigit() and len(dataset_type) == 1):
            dataset_type = int(dataset_type)
            idxes = [i for i in range(len(self.All_alpha_erosion)) if self.All_blur_levels[i] == dataset_type]
            
        else:
            raise NotImplementedError

        if idxes is None:
            self.all_alpha_erosion = self.All_alpha_erosion
            self.all_blur_levels = self.All_blur_levels
            self.all_img_paths = self.All_img_paths
        else:
            self.all_alpha_erosion = self.All_alpha_erosion[idxes]
            self.all_blur_levels = self.All_blur_levels[idxes]
            self.all_img_paths = [self.All_img_paths[i] for i in idxes]

    def __getitem__(self, idx):

        blur_level = self.all_blur_levels[idx]
        img_path = os.path.join(g_distractor_dir, self.all_img_paths[idx])

        img = np.array(Image.open(img_path))

        img = torch.from_numpy(img).float().permute(2, 0, 1) # [3 x H x W]

        # normalize the image
        if self.normalize:
            img = img / 255.0

        # transform the image
        if self.transform is not None:
            img = self.transform(img)

        return img, img_path, blur_level
        
    def __len__(self):
        if self.dataset_type is None:
            raise ValueError("You need to set self.dataset_type before getting the length. Use self.set_dataset_type().")

        return len(self.all_alpha_erosion)



class dataset_database_query_real(Dataset):
    def __init__(self, 
                 data_dir='/local/home/ronzou/real_data', 
                 normalize=True, transform=None,
                 ): 
        super().__init__()
        # manual seed
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic=True
        os.environ['PYTHONHASHSEED'] = str(seed)
        # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)        
        self.data_dir = data_dir
        self.normalize = normalize
        self.transform = transform
        self.dataset_type = None
        self.data = None

        self.database, self.queries = get_test_image_and_label_real(data_dir=self.data_dir)
    def set_dataset_type(self, dataset_type):
        
        if isinstance(dataset_type, tuple) or isinstance(dataset_type, list):
            dataset_type = dataset_type[0]

        assert dataset_type in ['database', 'db', 'query', 'q', 'database_v2', 'db_v2']
        # if it is "db" or "q", change it to "database" or "query"
        if dataset_type == 'db':
            dataset_type = 'database'
            # Do not use tuple, o.w. the above line gives error: TypeError: 'tuple' object does not support item assignment
        elif dataset_type == 'q':
            dataset_type = 'query'
        
        self.dataset_type = dataset_type
        # set self.data
        if dataset_type == "database":
            self.data = self.database
        elif dataset_type == "query":
            self.data = self.queries
        
        print("Setting data type to {}.".format(dataset_type))
        print("The data contain {} images.".format(len(self.data)))

    def __getitem__(self, idx):
        blur_level = 0
        
        if len(self.data[idx]) == 2:
            img_path, label = self.data[idx]
        elif len(self.data[idx]) == 4:
            img_path, blur_value, blur_level, label = self.data[idx]
        elif len(self.data[idx]) == 3:
            img_path, blur_level, label = self.data[idx]
        img = torch.from_numpy(np.array(Image.open(img_path))).float().permute(2, 0, 1) # [3 x H x W]

        # normalize the image
        if self.normalize:
            img = img / 255.0

        # transform the image
        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_path, blur_level
    
    def __len__(self):
        if self.dataset_type is None:
            raise ValueError("You need to set self.dataset_type before getting the length. Use self.set_dataset_type().")

        return len(self.data)



class dataset_train_val(Dataset): 
    def __init__(self, cls_ids, instance_paths, 
                 num_pos = 1, num_negs = 1, 
                 normalize=True, transform=None, device=None,
                 logger_dir=None,
                 take_blur_levels=[0,1,2,3,4,5],
                 take_only_sharp=False,
                 save_load_imgs_dir=None,
                 pred_blur_level_type=None,
                 localization_method=None,
                 pred_cls_label_type=None,
                 contrastive_bl_range=None,
                 get_contrastive_samples=True,
                 mode='train',
                 ): 
        super().__init__()
        # manual seed
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic=True
        os.environ['PYTHONHASHSEED'] = str(seed)
        # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
        
        self.pred_cls_label_type=None
        if pred_cls_label_type is not None:
            if pred_cls_label_type == 'cls' or pred_cls_label_type == 'class':
                self.pred_cls_label_type = 'cls'
            elif pred_cls_label_type == 'ins' or pred_cls_label_type == 'instance':
                self.pred_cls_label_type = 'ins'
            else:
                assert False, "pred_cls_label_type must be either 'cls' or 'ins'"

        self.num_cls = len(cls_ids)
        self.cls_ids = cls_ids
        # map class id to index from 0 to num_cls-1
        self.cls_id_to_idx = {cls_id: idx for idx, cls_id in enumerate(self.cls_ids)}

        self.pred_blur_level_type = pred_blur_level_type
        # assert self.pred_blur_level_type in ['discrete', 'continuous_erosion', 'continuous_original]
        self.localization_method = localization_method
        
        self.instance_paths = instance_paths
        # for each path in instance_paths, split by 'synthetic_data' and take the second part and join it with g_data_dir
        self.instance_paths = [os.path.join(g_data_dir, path.split('synthetic_data/')[1]) for path in self.instance_paths]

        # sort blur_levels_alpha from large to small
        # if take blur levels is an integer, make it a list
        if isinstance(take_blur_levels, int):
            take_blur_levels = [take_blur_levels]

        self.blur_levels_alpha_large_to_small = sorted(take_blur_levels, reverse=True)
        self.num_blur_levels = len(self.blur_levels_alpha_large_to_small)

        self.instance_names = [os.path.basename(instance_path) for instance_path in instance_paths]
        self.instance_names = sorted(list(set(self.instance_names)))
        self.instance_names_to_idx = {instance_name: idx for idx, instance_name in enumerate(self.instance_names)}
        
        self.bins = np.round(np.arange(0, 1.1, 0.1),3).tolist()
        
        self.normalize = normalize
        self.transform = transform
        self.num_negs = num_negs
        self.num_pos = num_pos
         
        self.save_load_imgs_dir = save_load_imgs_dir if save_load_imgs_dir is not None else os.path.join(g_data_dir, 'stats/loader/'+mode)    
        instance_alphas_path = os.path.join(save_load_imgs_dir, 'instance_alphas_{}.npy'.format('only_sharp' if take_only_sharp else 'mixed'))
        instance_alphas_original_path = os.path.join(save_load_imgs_dir, 'instance_alphas_original_{}.npy'.format('only_sharp' if take_only_sharp else 'mixed'))
        instance_images_path = os.path.join(save_load_imgs_dir, 'instance_images_{}.npy'.format('only_sharp' if take_only_sharp else 'mixed'))
        
        if not os.path.exists(instance_alphas_path) or not os.path.exists(instance_images_path) or not os.path.exists(instance_alphas_original_path):
            self.instance_traj_alpha_stats  = np.load(os.path.join(g_data_dir, 'stats/traj_stats', 'instance_traj_alpha_stats.npy'), allow_pickle=True).item()
            self.instance_traj_alpha_counts = np.load(os.path.join(g_data_dir, 'stats/traj_stats', 'instance_traj_alpha_counts.npy'), allow_pickle=True).item()
            # for k v in instance_traj_alpha_stats.items(), v is a list, keep only elements whose index is in blur_levels_alpha
            # instance_traj_alpha_stats = {k: [v[i] for i in blur_levels_alpha] for k, v in instance_traj_alpha_stats.items()}

            avg_alphas_erosion = np.load(os.path.join(g_data_dir,'stats/erosion', 'avg_alphas_erosion_by_blurlevel.npz'), allow_pickle=True)
            avg_alphas_erosion = [avg_alphas_erosion[i] for i in avg_alphas_erosion]
            
            with open(os.path.join(g_data_dir,'stats/erosion', 'img_paths_erosion_by_blurlevel.json'), 'r') as f:
                img_paths = json.load(f)

            self.all_avg_alpha = avg_alphas_erosion
            self.all_img_paths = img_paths
            
            self.instance_images = {instance_name: [] for instance_name in self.instance_names}
            self.instance_alphas = {instance_name: [] for instance_name in self.instance_names}
            self.instance_blur_levels = {instance_name: [] for instance_name in self.instance_names}
            self.all_instance_images = []
            self.all_instance_alphas = []
            self.all_instance_blur_levels = []

            # for instance_path in self.instance_paths:             
            for instance_path in tqdm(self.instance_paths):
            # when use alpha to define blur level
                instance_images_this, instance_alphas_this = self.get_instance_images(instance_path, type="rendered", method="only_sharp" if take_only_sharp else "even")
                
                blur_level_this = [len(self.bins)-1 - bisect.bisect(self.bins, a) for a in instance_alphas_this]

                instance_name = os.path.basename(instance_path)
                
                self.instance_images[instance_name] += instance_images_this
                self.instance_alphas[instance_name] += instance_alphas_this
                self.instance_blur_levels[instance_name] += blur_level_this

                self.all_instance_images += instance_images_this
                self.all_instance_alphas += instance_alphas_this
                self.all_instance_blur_levels += blur_level_this

            self.all_instance_alphas_original = []
            self.instance_alphas_original = copy.deepcopy(self.instance_alphas)
            for instance_name in self.instance_names:
                instance_images = self.instance_images[instance_name]
                instance_alphas = []
                for im in instance_images:
                    subf = im.split('/')[-1].split('_')[0]
                    traj_path = im.split('/')[-5:-1]
                    traj_path = os.path.join(*traj_path)
                    try:
                        blur_info = os.path.join('/cluster/project/infk/cvg/students/rzoran/', traj_path, 'blur_infos.json')
                        data_params = json.load(open(blur_info))
                    except:
                        blur_info = os.path.join('/local/home/ronzou/euler/', traj_path, 'blur_infos.json')
                        data_params = json.load(open(blur_info))
                    instance_alphas.append(data_params[subf]["avg_alpha"])
                self.instance_alphas_original[instance_name] = instance_alphas
                self.all_instance_alphas_original += instance_alphas
                        
            np.save(instance_alphas_original_path, self.instance_alphas_original)
            np.save(instance_images_path, self.instance_images)
            np.save(instance_alphas_path, self.instance_alphas)
            
            del self.instance_traj_alpha_stats
            del self.instance_traj_alpha_counts
            del self.all_avg_alpha
            del self.all_img_paths
            
        else: # load from disk
            self.instance_images = np.load(instance_images_path, allow_pickle=True).item()
            self.instance_alphas = np.load(instance_alphas_path, allow_pickle=True).item()
            self.instance_alphas_original = np.load(instance_alphas_original_path, allow_pickle=True).item()
            self.instance_blur_levels  = {instance_name: [len(self.bins)-1 - bisect.bisect(self.bins, a) for a in alphas] for instance_name, alphas in self.instance_alphas.items()}
            
            self.all_instance_images = []
            self.all_instance_alphas = []
            self.all_instance_alphas_original = []
            self.all_instance_blur_levels = []
            for instance_name in self.instance_names:
                instance_images = self.instance_images[instance_name]
                instance_images =  [os.path.join(g_data_dir, path.split('synthetic_data/')[1]) for path in instance_images]
                self.instance_images[instance_name] = instance_images
                self.all_instance_images += instance_images
                self.all_instance_alphas += self.instance_alphas[instance_name]
                self.all_instance_alphas_original += self.instance_alphas_original[instance_name]
                self.all_instance_blur_levels += self.instance_blur_levels[instance_name]
            
        # sort the images by blur level
        self.all_instance_images_by_blur_level = {blur_level: [] for blur_level in self.blur_levels_alpha_large_to_small}
        self.all_instance_alphas_by_blur_level = {blur_level: [] for blur_level in self.blur_levels_alpha_large_to_small}
        for i in range(len(self.all_instance_images)):
            blur_level = self.all_instance_blur_levels[i]
            self.all_instance_images_by_blur_level[blur_level].append(self.all_instance_images[i])
            self.all_instance_alphas_by_blur_level[blur_level].append(self.all_instance_alphas[i])

        self.instance_images_by_blur_level = {instance_name: {blur_level: [] for blur_level in self.blur_levels_alpha_large_to_small} for instance_name in self.instance_names}
        self.instance_alphas_by_blur_level = {instance_name: {blur_level: [] for blur_level in self.blur_levels_alpha_large_to_small} for instance_name in self.instance_names}
        for instance_name in self.instance_names:
            for i in range(len(self.instance_images[instance_name])):
                blur_level = self.instance_blur_levels[instance_name][i]
                self.instance_images_by_blur_level[instance_name][blur_level].append(self.instance_images[instance_name][i])
                self.instance_alphas_by_blur_level[instance_name][blur_level].append(self.instance_alphas[instance_name][i])
                
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.instance_descriptors = {}

        if logger_dir is not None:
            # if 'train' is contained in logger_dir, then it is a training logger, set mode to train
            if 'train' in logger_dir:
                self.logger = get_logger(logger_dir, "train_set")
                self.logger.info("Initialize Train set logger.")
                
                # self.writter = SummaryWriter(log_dir=logger_dir)
            elif 'val' in logger_dir:
                self.logger = get_logger(logger_dir, "val_set")
                self.logger.info("Initialize Val set logger.")
                
                # self.writter = SummaryWriter(log_dir=logger_dir)
        else:
            self.logger = None
            # self.writter = None
        self.img_step=-1
        self.blur_levels_alpha_small_to_large = sorted(self.blur_levels_alpha_large_to_small)
        self.max_blur_level = self.blur_levels_alpha_small_to_large[-1]
        self.min_blur_level = self.blur_levels_alpha_small_to_large[0]
        self.get_contrastive_samples = get_contrastive_samples
        self.contrastive_bl_range = contrastive_bl_range

    def get_instance_images(self, instance_path, type='rendered', method='even'):
        instance_images = []
        instance_alphas = []

        if method == 'even':
            instance_name = os.path.basename(instance_path)
            # get the alpha stats and counts of this instance
            instance_traj_alpha_stats_this = self.instance_traj_alpha_stats[instance_name]
            instance_traj_alpha_counts_this = self.instance_traj_alpha_counts[instance_name]
            # sort the two lists by counts, we also need the indices of the sorted lists
            instance_traj_alpha_counts_this_sorted, blur_levels_sorted = torch.sort(torch.tensor(instance_traj_alpha_counts_this), descending=False) # 9857264310
            instance_traj_alpha_stats_this_sorted = [instance_traj_alpha_stats_this[i] for i in blur_levels_sorted]
            # get the idxes of the blur levels in blur_levels_sorted that are in blur_levels_alpha
            idxes = [i for i in range(len(blur_levels_sorted)) if blur_levels_sorted[i] in self.blur_levels_alpha_large_to_small]
            # keep only the blur levels in blur_levels_alpha
            instance_traj_alpha_counts_this_sorted = [instance_traj_alpha_counts_this_sorted[i].item() for i in idxes]
            instance_traj_alpha_stats_this_sorted = [instance_traj_alpha_stats_this_sorted[i] for i in idxes]
            blur_levels_sorted = [blur_levels_sorted[i].item() for i in idxes]
                    

            take_trajs_for_diff_blur_levels = {blur_level: [] for blur_level in self.blur_levels_alpha_large_to_small}
            take_num_trajs_for_diff_blur_levels = {blur_level: 0 for blur_level in self.blur_levels_alpha_large_to_small}

        
            num_traj = max(instance_traj_alpha_counts_this)
            num_traj_each_blur_level = [math.floor(num_traj / self.num_blur_levels)] * self.num_blur_levels
            remains = num_traj % self.num_blur_levels
            # ranomly choose remains blur levels and add 1 to num_traj_each_blur_level
            if remains > 0:
                take_blur_levels = random.sample(range(self.num_blur_levels), remains)
                for i in take_blur_levels:
                    num_traj_each_blur_level[i] += 1
            
            for i in range(self.num_blur_levels):
                cur_blur_level = blur_levels_sorted[i]
                trajs_this_blur_level = instance_traj_alpha_stats_this_sorted[i]
                traj_count_this_blur_level = len(trajs_this_blur_level)
                
                if traj_count_this_blur_level < num_traj_each_blur_level[i]:
                    take_trajs_this_blur_level = trajs_this_blur_level

                    if i < self.num_blur_levels - 1:
                        # how less trajs do we take
                        spare_num_traj = num_traj_each_blur_level[i] - traj_count_this_blur_level
                        # distribute this num trajs evenly to j>i num_traj_each_blur_level[j]
                        for j in range(i+1, self.num_blur_levels):
                            num_traj_each_blur_level[j] += spare_num_traj // (self.num_blur_levels - i - 1)
                        spare_num_traj = spare_num_traj % (self.num_blur_levels - i - 1)
                        # randomly choose spare_num_traj blur levels from i+1 to self.num_blur_levels-1 and add 1 
                        if spare_num_traj > 0:
                            take_blur_levels = random.sample(range(i+1, self.num_blur_levels), spare_num_traj)
                            for j in take_blur_levels:
                                num_traj_each_blur_level[j] += 1

                elif traj_count_this_blur_level > num_traj_each_blur_level[i]:
                    take_trajs_this_blur_level = random.sample(trajs_this_blur_level, num_traj_each_blur_level[i])
                
                else:
                    take_trajs_this_blur_level = trajs_this_blur_level

                # if non zero
                if len(take_trajs_this_blur_level) > 0:
                    take_trajs_for_diff_blur_levels[cur_blur_level] = take_trajs_this_blur_level
                    take_num_trajs_for_diff_blur_levels[cur_blur_level] = len(take_trajs_this_blur_level)

                    # remove the chosen trajs from instance_traj_alpha_stats_this_sorted[j] for all j > i
                    for j in range(i+1, self.num_blur_levels):
                        instance_traj_alpha_stats_this_sorted[j] = sorted(list(set(instance_traj_alpha_stats_this_sorted[j]) - set(take_trajs_this_blur_level)))

            # get the images

            for i in range(self.num_blur_levels):
                cur_blur_level = blur_levels_sorted[i]
                take_trajs_this_blur_level = take_trajs_for_diff_blur_levels[cur_blur_level]
                take_num_trajs_this_blur_level = take_num_trajs_for_diff_blur_levels[cur_blur_level]
                # get the images of this blur level
                if take_num_trajs_this_blur_level > 0:
                    # get the images of this blur level
                    # get the image paths of all the trajs in take_trajs_this_blur_level
                    take_img_paths_this_blur_level = []
                    take_img_alphas_this_blur_level = []
                    for traj in take_trajs_this_blur_level:
                        traj_path = os.path.join(instance_path, traj).split('/')[-3:] # ['02691156', '1c93b0eb9c313f5d9a6e43b878d5b335', '000']
                        traj_path = os.path.join(*traj_path)
                        # get the image paths of all the images in this traj
                        im_paths = [os.path.join(traj_path, str(i)+'_blurred.png') for i in range(0, 11)]
                        
                        paths_ = []
                        alphas_ = []
                        for im_path in im_paths:
                            try:
                                im_idx = self.all_img_paths[cur_blur_level].index(im_path)
                            except:
                                continue
                            im_alpha = self.all_avg_alpha[cur_blur_level][im_idx]
                            paths_.append(im_path)
                            alphas_.append(im_alpha)
                        # randomly choose one image
                        idx_ = random.choice(range(len(paths_)))
                        
                        take_img_alphas_this_blur_level.append(alphas_[idx_])
                        im_idx_in_traj = paths_[idx_].split('/')[-1].split('_')[0]
                        take_img_paths_this_blur_level.append(os.path.join(instance_path, traj, str(im_idx_in_traj)+'_'+type+'.png'))

                    # add the images of this blur level to instance_images
                    instance_images.extend(take_img_paths_this_blur_level)
                    instance_alphas.extend(take_img_alphas_this_blur_level)
        elif method == 'only_sharp':
            num_traj = 120
            # join instance_path, traj, 0_blurred.png for all trajs from 0 to num_traj-1
            for traj in range(num_traj):
                traj_id = str(traj).zfill(3)
                im_path = os.path.join(instance_path, traj_id, '0_'+type+'.png')  # full path
                
                im_path_blurred = im_path.split('/')[-4:] # ['02691156', '1c93b0eb9c313f5d9a6e43b878d5b335', '000', '0_rendered.png']
                im_path_blurred = os.path.join(*im_path_blurred).replace('_rendered.png', '_blurred.png')
                try:
                    im_idx = self.all_img_paths[0].index(im_path_blurred)
                except:
                    continue
                im_alpha = self.all_avg_alpha[0][im_idx]
                instance_images.append(im_path)
                instance_alphas.append(im_alpha)
        else:
            raise NotImplementedError

        # shuffle the images and alphas
        shuffled_idxes = list(range(len(instance_images)))
        random.shuffle(shuffled_idxes)
        instance_images = [instance_images[i] for i in shuffled_idxes]
        instance_alphas = [instance_alphas[i] for i in shuffled_idxes]
        
        return instance_images, instance_alphas

    def get_pos_pool(self, image_path):
        # image_path is the path of a rendered image: .../02691156/1c93b0eb9c313f5d9a6e43b878d5b335/000/1_rendered.png
        instance_name_this = image_path.split("/")[-3]
        pos_pool = self.instance_images[instance_name_this].copy()
        # remove the image_path itself
        pos_pool.remove(image_path)
        return pos_pool
    
    def get_pos_pool_by_blur_level(self, image_path, blur_level):
        # image_path is the path of a rendered image: .../02691156/1c93b0eb9c313f5d9a6e43b878d5b335/000/1_rendered.png
        instance_name_this = image_path.split("/")[-3]
        pos_pool = self.instance_images_by_blur_level[instance_name_this][blur_level].copy()
        # remove the image_path itself if it is in pos_pool
        if image_path in pos_pool:
            pos_pool.remove(image_path)
        
        return pos_pool
    
    def get_pos_pool_by_blur_levels(self, image_path, blur_levels: list):
        blur_levels = sorted(blur_levels)
        pos_pool = []
        if blur_levels == self.blur_levels_alpha_small_to_large:
            # if the blur levels are all the blur levels, then we can just return the pos pool of this image
            pos_pool = self.get_pos_pool(image_path)
            return pos_pool
        
        for blur_level in blur_levels:
            pos_pool.extend(self.get_pos_pool_by_blur_level(image_path, blur_level))
        
        return pos_pool

    def get_neg_pool(self, image_path):
        instance_name_this = image_path.split("/")[-3]
        neg_pool = self.all_instance_images.copy()
        # remove the images of the same instance
        for image_path_this_instance in self.instance_images[instance_name_this]:
            neg_pool.remove(image_path_this_instance)

        return neg_pool
    
    def get_neg_pool_by_blur_level(self, image_path, blur_level):
        instance_name_this = image_path.split("/")[-3]
        
        neg_pool = self.all_instance_images_by_blur_level[blur_level].copy()
        # remove the images of the same instance and the same blur level
        for image_path_this_instance in self.instance_images_by_blur_level[instance_name_this][blur_level]:
            neg_pool.remove(image_path_this_instance)
        
        return neg_pool
    
    def get_neg_pool_by_blur_levels(self, image_path, blur_levels: list):      
        blur_levels = sorted(blur_levels)  
        neg_pool = []
        if blur_levels == self.blur_levels_alpha_small_to_large:
            # if the blur levels are all the blur levels, then we can just return the neg pool of this image
            neg_pool = self.get_neg_pool(image_path)
            return neg_pool
        
        for blur_level in blur_levels:
            neg_pool.extend(self.get_neg_pool_by_blur_level(image_path, blur_level))
        
        return neg_pool
    
    def get_image(self, image_path, with_batch_dim=False):
        img = torch.from_numpy(np.array(Image.open(image_path))).float().permute(2, 0, 1)
        if self.normalize:
            img = img / 255.0
        if self.transform is not None:
            img = self.transform(img)
        
        if with_batch_dim:
            # make batch size 1
            img = img.unsqueeze(0)

            # move to device
            img = img.to(self.device)
        return img

    def get_mask(self, image_path):
        mask_path = image_path.replace("_rendered.png", "_blurred.png")
        mask = torch.from_numpy(np.array(Image.open(mask_path))).float().permute(2, 0, 1) # [4 x H x W]
        # take the alpha channel of mat_alpha1 and turn it into binary, if the value is non-zero, set it to 255, otherwise set it to 0
        mask = torch.where(mask[3,:,:] > 0, torch.tensor(255.0), torch.tensor(0.0)).unsqueeze(0)
        
        if self.normalize:
            mask = mask / 255.0

        if self.transform:
            mask = self.transform(mask)

        # mask = mask.to(self.device)
        return mask
    
    def get_bbox(self, image_path):
        # first get the mask
        mask = self.get_mask(image_path)
        
        # find the first and last non-zero row and column
        col_min_idx = torch.nonzero(mask[0,:,:].sum(dim=0) > 0)[0]
        col_max_idx = torch.nonzero(mask[0,:,:].sum(dim=0) > 0)[-1]
        row_min_idx = torch.nonzero(mask[0,:,:].sum(dim=1) > 0)[0]
        row_max_idx = torch.nonzero(mask[0,:,:].sum(dim=1) > 0)[-1]
        
        # get the bbox [center_x, center_y, width, height]
        center_x = (col_min_idx + col_max_idx) / 2.0
        norm_center_x = center_x / mask.shape[2]
        center_y = (row_min_idx + row_max_idx) / 2.0
        norm_center_y = center_y / mask.shape[1]
        width = col_max_idx - col_min_idx
        norm_width = width / mask.shape[2]
        height = row_max_idx - row_min_idx
        norm_height = height / mask.shape[1]
        # bbox_denorm = torch.tensor([center_x, center_y, width, height])
        bbox = torch.tensor([norm_center_x, norm_center_y, norm_width, norm_height])

        # get the bbox [x1, y1, x2, y2], normalized
        # x1 = col_min_idx / mask.shape[2]
        # y1 = row_min_idx / mask.shape[1]
        # x2 = col_max_idx / mask.shape[2]
        # y2 = row_max_idx / mask.shape[1]
        # bbox = torch.tensor([x1, y1, x2, y2])
        
        return bbox
        
    def get_pos_imgs(self, image_path):
        
        idx = self.all_instance_images.index(image_path)
        blur_level = self.all_instance_blur_levels[idx]

        r = 1 if self.contrastive_bl_range == 'S' else 3 if self.contrastive_bl_range == 'M' else 5 if self.contrastive_bl_range == 'L' else None
        
        pos_blur_level_range_min = max(blur_level - r, self.min_blur_level)
        pos_blur_level_range_max = min(blur_level + r, self.max_blur_level)
        pos_pool = self.get_pos_pool_by_blur_levels(image_path, range(pos_blur_level_range_min, pos_blur_level_range_max+1))
        
        try:
            pos_img_paths = random.sample(pos_pool, self.num_pos)
        except:
            pos_img_paths = random.choices(pos_pool, k=self.num_pos)

        blur_level_pos = [self.all_instance_blur_levels[self.all_instance_images.index(pos_img_path)] for pos_img_path in pos_img_paths]
        
        if self.logger is not None:
            self.logger.info("IMG_STEP {}, pos_img_paths: {}".format(self.img_step, pos_img_paths))

        pos_imgs = [self.get_image(pos_img_path) for pos_img_path in pos_img_paths]
        # make pos_imgs into a tensor of shape (num_pos, 3, H, W)
        pos_imgs = torch.stack(pos_imgs, dim=0)
            
        return pos_imgs
    
    def get_neg_imgs(self, image_path):

        idx = self.all_instance_images.index(image_path)
        blur_level = self.all_instance_blur_levels[idx]
        
        r = 1 if self.contrastive_bl_range == 'S' else 3 if self.contrastive_bl_range == 'M' else 5 if self.contrastive_bl_range == 'L' else None
        
        neg_blur_level_range_min = max(blur_level - r, self.min_blur_level)
        neg_blur_level_range_max = min(blur_level + r, self.max_blur_level)
        neg_pool = self.get_neg_pool_by_blur_levels(image_path, range(neg_blur_level_range_min, neg_blur_level_range_max+1))
        
        try:
            neg_img_paths = random.sample(neg_pool, self.num_negs)
        except:
            neg_img_paths = random.choices(neg_pool, k=self.num_negs)      
            
        blur_level_neg = [self.all_instance_blur_levels[self.all_instance_images.index(neg_img_path)] for neg_img_path in neg_img_paths]
        
        if self.logger is not None:
            self.logger.info("IMG_STEP {}, neg_img_paths: {}".format(self.img_step,neg_img_paths))
       
        neg_imgs = [self.get_image(neg_img_path) for neg_img_path in neg_img_paths]
        # make neg_imgs into a tensor of shape (num_neg, 3, H, W)
        neg_imgs = torch.stack(neg_imgs, dim=0)
            
        return neg_imgs

    def get_num_instances(self):
        return len(self.instance_paths)
    
    def get_num_images(self):
        return len(self.all_instance_images)
    
    def get_num_images_of_instance(self, instance_idx):
        instance_name = self.instance_names[instance_idx]
        return len(self.instance_images[instance_name])
    
    def __len__(self):
        return self.get_num_images()
    
    def __getitem__(self, idx):
        # get image
        image_path = self.all_instance_images[idx]
        img = self.get_image(image_path)
        
        labels = {"img_path": image_path}

        if self.get_contrastive_samples:
            pos_imgs = self.get_pos_imgs(image_path)
            neg_imgs = self.get_neg_imgs(image_path)
            contractive_label = torch.tensor([-1] + [1] * self.num_pos + [0] * self.num_negs, dtype=torch.int64)
            labels["contrastive_label"] = contractive_label
        else:
            pos_imgs = torch.tensor(0.0)
            neg_imgs = torch.tensor(0.0)
        
        if self.pred_cls_label_type is not None:

            if self.pred_cls_label_type == 'cls':
                cls_id = image_path.split("/")[-4]
                cls_idx = self.cls_id_to_idx[cls_id]
                cls_idx = torch.tensor(cls_idx, dtype=torch.int64)
                labels["cls_idx"] = cls_idx
            elif self.pred_cls_label_type == 'ins':
                ins_name = image_path.split("/")[-3]
                ins_idx = self.instance_names_to_idx[ins_name]
                ins_idx = torch.tensor(ins_idx, dtype=torch.int64)
                labels["cls_idx"] = ins_idx

        if self.pred_blur_level_type is not None:
            if self.pred_blur_level_type == 'discrete':
                blur_level = self.all_instance_blur_levels[idx]
                labels["blur_level"] = blur_level
            elif self.pred_blur_level_type == 'continuous_erosion':
                blur_level = self.all_instance_alphas[idx]
                labels["blur_level"] = blur_level
            elif self.pred_blur_level_type == 'continuous_original':
                blur_level = self.all_instance_alphas_original[idx]
                labels["blur_level"] = blur_level
            else:
                raise NotImplementedError
        
        if self.localization_method is not None:
            if self.localization_method == 'bbox':
                bbox = self.get_bbox(image_path)
                labels["bbox"] = bbox
            else:
                raise NotImplementedError

        return img, pos_imgs, neg_imgs, labels

def split_data(data_dir, train_ratio=0.8, val_ratio=0.1, database_ratio=0.8, save_results=True, save_dir=None,
                  train_dir = "train", val_dir = "val", test_dir = "test", database_dir = "database", query_dir = "query"):
    """
    Train, Val and Test are different instances.
    Each cls_folder has a list of instance folders, take train_ratio of the instance folders as train, 
    take val_ratio of the instance folders as val, take the rest as test
    for each test instance folder, randomly sample database_ratio of the trajectory folders as database, take the rest as query
    """

    # check if the data_dir contains train, val, test/database, test/query folders simultaneously
    if os.path.isdir(os.path.join(data_dir, train_dir)) and os.path.isdir(os.path.join(data_dir, val_dir)) and os.path.isdir(os.path.join(data_dir, test_dir)):
        # get the cls ids and instance paths from the train, val, test folders
        print("data_dir contains train, val, test folders, will only read and return, no split.")
        train_cls_ids, train_instance_folders = get_cls_ids_and_instance_paths(os.path.join(data_dir, train_dir), type_of_data="train")
        val_cls_ids, val_instance_folders = get_cls_ids_and_instance_paths(os.path.join(data_dir, val_dir), type_of_data="val")
        test_cls_ids, test_instance_folders = get_cls_ids_and_instance_paths(os.path.join(data_dir, test_dir), type_of_data="test")
        # assert two lists contain same elements without considering the order
        assert set(train_cls_ids) == set(val_cls_ids) == set(test_cls_ids[0]) == set(test_cls_ids[1]), "train, val, test folders contain different cls ids"
        cls_ids = train_cls_ids
        database = test_instance_folders[0]
        queries = test_instance_folders[1]

        return cls_ids, train_instance_folders, val_instance_folders, database, queries
    
    else:
        print("data_dir does not contain train, val, test folders, will split the data_dir into train, val, test/database, test/query.")
        # read the params.json file in the data_dir
        try:
            data_params_json_path = os.path.join(data_dir, 'params1.json') # You may need to change json file name here.
            assert os.path.isfile(data_params_json_path), "No json configuration file found at {}".format(data_params_json_path)
        except:
            data_params_json_path = os.path.join(data_dir, 'params.json')
            assert os.path.isfile(data_params_json_path), "No json configuration file found at {}".format(data_params_json_path)
        
        data_params = json.load(open(data_params_json_path))
        # get the param "category_id_names"
        # "category_id_names": [["04379243", "knives"], ["03001627", "lamps"], ["02691156", "guns"], [02958343", "board"] ...]
        category_id_names = data_params["category_id_names"]
        num_cls = len(category_id_names)
        cls_ids = [category_id_name[0] for category_id_name in category_id_names]
        cls_names = [category_id_name[1] for category_id_name in category_id_names]
        # get the cls_folders by join the cls_ids with the data_dir
        cls_folders = [os.path.join(data_dir, cls_id) for cls_id in cls_ids]

        train_instance_folders = []
        val_instance_folders = []
        test_instance_folders = []
        database = []
        queries = []

        database_val = []
        queries_val = []
        
        for cls_folder in cls_folders:
            # get the instance folders for each cls_folder
            cls_instance_folders = glob.glob(os.path.join(cls_folder, "*"))
            # randomly shuffle the cls_instance_folders
            random.shuffle(cls_instance_folders)
            # get the number of instance folders
            num_instance_folders = len(cls_instance_folders)
            # get the number of train, val, test folders
            num_train_instance_folders = int(num_instance_folders * train_ratio)
            num_val_instance_folders = int(num_instance_folders * val_ratio)
            num_test_instance_folders = num_instance_folders - num_train_instance_folders - num_val_instance_folders
            # get the train, val, test instance folders
            train_instance_folders_ = cls_instance_folders[:num_train_instance_folders]
            val_instance_folders_ = cls_instance_folders[num_train_instance_folders:num_train_instance_folders+num_val_instance_folders]
            test_instance_folders_ = cls_instance_folders[num_train_instance_folders+num_val_instance_folders:]
            # append the train, val, test instance folders to the train, val, test instance folders list
            train_instance_folders.extend(train_instance_folders_)
            val_instance_folders.extend(val_instance_folders_)
            test_instance_folders.extend(test_instance_folders_)

            # for each test instance folder, randomly sample database_ratio of the trajectory folders as database, take the rest as query
            for test_instance_folder in test_instance_folders_:
                # get the trajectory folders for each test_instance_folder
                test_instance_trajectory_folders = glob.glob(os.path.join(test_instance_folder, "*"))
                # randomly shuffle the test_instance_trajectory_folders
                random.shuffle(test_instance_trajectory_folders)
                # get the number of trajectory folders
                num_test_instance_trajectory_folders = len(test_instance_trajectory_folders)
                # get the number of database, query folders
                num_database = int(num_test_instance_trajectory_folders * database_ratio)
                num_query = num_test_instance_trajectory_folders - num_database
                # get the database, query instance folders
                database_ = test_instance_trajectory_folders[:num_database]
                queries_ = test_instance_trajectory_folders[num_database:]
                # append the database, query instance folders to the database, query instance folders list
                database.extend(database_)
                queries.extend(queries_)

            for val_instance_folder in val_instance_folders_:
                # get the trajectory folders for each val_instance_folder
                val_instance_trajectory_folders = glob.glob(os.path.join(val_instance_folder, "*"))
                # randomly shuffle the val_instance_trajectory_folders
                random.shuffle(val_instance_trajectory_folders)
                # get the number of trajectory folders
                num_val_instance_trajectory_folders = len(val_instance_trajectory_folders)
                # get the number of database, query folders
                num_database = int(num_val_instance_trajectory_folders * database_ratio)
                num_query = num_val_instance_trajectory_folders - num_database
                # get the database, query instance folders
                database_val_ = val_instance_trajectory_folders[:num_database]
                queries_val_ = val_instance_trajectory_folders[num_database:]
                # append the database, query instance folders to the database, query instance folders list
                database_val.extend(database_val_)
                queries_val.extend(queries_val_)
            
        if save_results:
           
            num_database_imgs = len(database)
            num_query_imgs = len(queries)

            num_database_imgs_val = len(database_val)
            num_query_imgs_val = len(queries_val)
                
            # save the cls_ids and train_instance_folders, val_instance_folders, test_instance_folders, database, queries into json files
            if save_dir is not None:
                json_file = os.path.join(save_dir, "data_split_info.json")
                with open(json_file, 'w') as f:
                    json.dump({"cls_ids": cls_ids, 
                            "cls_names": cls_names, 
                            "num_cls": num_cls,
                            "num_train_instances": len(train_instance_folders),
                            "num_val_instances": len(val_instance_folders),
                            "num_test_instances": len(test_instance_folders),
                            "num_database_imgs": num_database_imgs,
                            "num_query_imgs": num_query_imgs,
                            "num_database_imgs_val": num_database_imgs_val,
                            "num_query_imgs_val": num_query_imgs_val,
                            "train_instance_folders": train_instance_folders, 
                            "val_instance_folders": val_instance_folders, 
                            "test_instance_folders": test_instance_folders, 
                            "database": database, 
                            "queries": queries,
                            "database_val": database_val,
                            "queries_val": queries_val
                            }, f, indent=4)

        return cls_ids, train_instance_folders, val_instance_folders, test_instance_folders, database, queries, database_val, queries_val

def get_cls_ids_and_instance_paths(data_dir, type_of_data):
    # get the cls ids and instance paths from the train, val, test folders
    # data_dir is the path to the train, val, test folders

    if type_of_data == "train" or type_of_data == "val":
        cls_ids = []
        instance_paths = []
        for cls_folder in glob.glob(os.path.join(data_dir, "*")):
            cls_id = cls_folder.split("/")[-1]
            cls_ids.append(cls_id)
            for instance_folder in glob.glob(os.path.join(cls_folder, "*")):
                instance_paths.append(instance_folder)

    elif type_of_data == "test":
        cls_ids_db = []
        instance_paths_db = []
        cls_ids_query = []
        instance_paths_query = []
        instance_paths = []
        for cls_folder in glob.glob(os.path.join(data_dir, "database", "*")):
            cls_id = cls_folder.split("/")[-1]
            cls_ids_db.append(cls_id)
            for instance_folder in glob.glob(os.path.join(cls_folder, "*")):
                # append all subdirs in instance_folder to instance_paths_db
                instance_paths_db.extend(glob.glob(os.path.join(instance_folder, "*")))
        
        for cls_folder in glob.glob(os.path.join(data_dir, "query", "*")):
            cls_id = cls_folder.split("/")[-1]
            cls_ids_query.append(cls_id)
            for instance_folder in glob.glob(os.path.join(cls_folder, "*")):
                instance_paths_query.extend(glob.glob(os.path.join(instance_folder, "*")))

        cls_ids = [cls_ids_db, cls_ids_query]
        instance_paths = [instance_paths_db, instance_paths_query]

    return cls_ids, instance_paths

def get_test_image_and_label_real(data_dir):
    
    # read images
    all_images = json.load(open(os.path.join(data_dir, "all_images.json")))
    # get the unique instance ids
    unique_instance_ids = [img.split('/')[-1].split('_')[0] for img in all_images]
    unique_instance_ids = sorted(list(set(unique_instance_ids)))
    num_instances = len(unique_instance_ids)
    # map the unique instance ids to a one-hot label
    instance_id_to_label = {instance_id: np.eye(num_instances)[i] for i, instance_id in enumerate(unique_instance_ids)}

    # construct a dict like this: {instance_id: {traj_id: [(img_path, blur_level), ...], ...}, ...}
    instance_imgs = {instance_id: {} for instance_id in unique_instance_ids}
    for img_path in all_images:
        instance_id = img_path.split('/')[-1].split('_')[0]
        traj_id = img_path.split('/')[-1].split('_')[1]
        blur_level = int(img_path.split('/')[-2].split('_')[-1])-1
        if traj_id not in instance_imgs[instance_id].keys():
            instance_imgs[instance_id][traj_id] = []
        instance_imgs[instance_id][traj_id].append((img_path, blur_level))
    
    # print the number of imgs for each instance
    for instance_id in instance_imgs.keys():
        print("instance_id: {}, num_imgs: {}".format(instance_id, sum([len(instance_imgs[instance_id][traj_id]) for traj_id in instance_imgs[instance_id].keys()])))

    # construct database and queries, for each instance, take the trajectory with the least number of images as query and the rest as database
    database = []
    queries = []
    # count for each instance id, how many imgs are there in database and query
    num_database_imgs = {instance_id: 0 for instance_id in unique_instance_ids}
    num_query_imgs = {instance_id: 0 for instance_id in unique_instance_ids}
    for instance_id in unique_instance_ids:
        # get the label
        label = instance_id_to_label[instance_id]
        # get the trajectories
        trajectories = instance_imgs[instance_id]
        # get the trajectory with the least number of images as query
        query_traj_id = min(trajectories, key=lambda x: len(trajectories[x]))
        # get the query images
        query_imgs = trajectories[query_traj_id]
        # get the database images
        database_imgs = []
        for traj_id in trajectories.keys():
            if traj_id != query_traj_id:
                database_imgs.extend(trajectories[traj_id])
        # append the images and labels to database
        database.extend([(image, blur_level, label) for image, blur_level in database_imgs])
        queries.extend([(image, blur_level, label) for image, blur_level in query_imgs])
        # update the num_database_imgs and num_query_imgs
        num_database_imgs[instance_id] += len(database_imgs)
        num_query_imgs[instance_id] += len(query_imgs)

    # print database and queries size
    print("database size: {}".format(len(database)))
    print("queries size: {}".format(len(queries)))

    # print the number of imgs for each instance in database and query
    for instance_id in unique_instance_ids:
        print("instance_id: {}, num_database_imgs: {}, num_query_imgs: {}".format(instance_id, num_database_imgs[instance_id], num_query_imgs[instance_id]))


    return database, queries

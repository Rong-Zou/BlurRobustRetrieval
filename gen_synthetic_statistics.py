import numpy as np
import torch
import os
import cv2
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import json
from collections import OrderedDict
import bisect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def erosion_alphas(data_dir="/local/home/ronzou/euler/synthetic_data"):
  
  num_blur_levels = 11
  erosion_iter = 1
  kernel = np.ones((3,3), np.uint8)

  avg_alphas = [[] for i in range(num_blur_levels)]
  img_paths = [[] for i in range(num_blur_levels)]
  avg_alphas_erosion = {k: [[] for i in range(num_blur_levels)] for k in range(erosion_iter)}

  for cls_dir in tqdm(os.listdir(data_dir)):
      if os.path.isdir(os.path.join(data_dir, cls_dir)):
          for ins_dir in tqdm(os.listdir(os.path.join(data_dir, cls_dir))):
              if os.path.isdir(os.path.join(data_dir, cls_dir, ins_dir)):
                  for traj_dir in os.listdir(os.path.join(data_dir, cls_dir, ins_dir)):
                      if os.path.isdir(os.path.join(data_dir, cls_dir, ins_dir, traj_dir)):
                          # if it is not empty, read the blur_infos.json in this dir
                          if len(os.listdir(os.path.join(data_dir, cls_dir, ins_dir, traj_dir))) > 0:
                              with open(os.path.join(data_dir, cls_dir, ins_dir, traj_dir, "blur_infos.json"), 'r') as f:
                                  blur_infos = json.load(f)

                              for BL in blur_infos:
                                  avg_alphas[int(BL)].append(blur_infos[BL]['avg_alpha'])
                                  im_p = os.path.join(cls_dir, ins_dir, traj_dir, BL+'_blurred.png')
                                  img_paths[int(BL)].append(im_p)

                                  # read im_p, erosion, compute avg_alpha_erosion, append to avg_alphas[int(BL)]
                                  img = cv2.imread(os.path.join(data_dir, im_p), cv2.IMREAD_UNCHANGED)
                                  alpha = img[:,:,3]
                                  binary_mask = np.zeros(alpha.shape, dtype=np.uint8)
                                  binary_mask[alpha>0] = 255
                                  alpha = alpha.astype(np.float32)/255.0
                                  for erosion_i in range(erosion_iter):
                                      erosion = cv2.erode(binary_mask, kernel, iterations=erosion_i+1)
                                      alpha_erosion = alpha * erosion.astype(np.float32)/255.0
                                      m = alpha_erosion > 0
                                      num_pix_obj = np.count_nonzero(m)
                                      avg_alpha_erosion = np.sum(alpha_erosion[m])/num_pix_obj
                                      # could be that num_pix_obj is 0, then avg_alpha_erosion is nan
                                      if np.isnan(avg_alpha_erosion):
                                          avg_alpha_erosion = 0

                                      avg_alphas_erosion[erosion_i][int(BL)].append(avg_alpha_erosion)

  for i in range(num_blur_levels):
      sorted_idx = np.argsort(avg_alphas[i])
      avg_alphas[i] = [avg_alphas[i][idx] for idx in sorted_idx]
      img_paths[i] = [img_paths[i][idx] for idx in sorted_idx]
      for erosion_i in range(erosion_iter):
          avg_alphas_erosion[erosion_i][i] = [avg_alphas_erosion[erosion_i][i][idx] for idx in sorted_idx]

                                  
  # save 
  np.save(os.path.join(data_dir, 'stats', 'erosion', 'avg_alphas_original_by_subframe.npy'), avg_alphas)
  np.save(os.path.join(data_dir, 'stats', 'erosion', 'avg_alphas_erosion_by_subframe.npy'), avg_alphas_erosion)
  # save img_paths as json
  with open(os.path.join(data_dir, 'stats', 'erosion', 'img_paths_by_subframe.json'), 'w') as f:
    json.dump(img_paths, f, indent=4)
    

def erosion_alphas_distractor(data_dir="/local/home/ronzou/euler/synthetic_data_distractor"):
    cls_dirs = os.listdir(data_dir)
    # keep only the dirs
    cls_dirs = [d for d in cls_dirs if os.path.isdir(os.path.join(data_dir, d))]
    # remove stats dir
    cls_dirs = [d for d in cls_dirs if d != 'stats']


    img_paths = []
    alphas = []
    kernel = np.ones((3,3), np.uint8)
    
    alphas_erosion = []
    blur_levels = []
    
    bins = np.round(np.arange(0, 1.1, 0.1),3).tolist()
    
    for cls_dir in tqdm(cls_dirs):
        print(cls_dir)
        cls_img_paths = []
        cls_alphas = []
        cls_alphas_erosion = []
        cls_blur_levels = []
        for ins_dir in tqdm(os.listdir(os.path.join(data_dir, cls_dir))):
            if os.path.isdir(os.path.join(data_dir, cls_dir, ins_dir)):
                for traj_dir in os.listdir(os.path.join(data_dir, cls_dir, ins_dir)):
                    if os.path.isdir(os.path.join(data_dir, cls_dir, ins_dir, traj_dir)):
                        if len(os.listdir(os.path.join(data_dir, cls_dir, ins_dir, traj_dir))) > 0:
                            with open(os.path.join(data_dir, cls_dir, ins_dir, traj_dir, "blur_infos.json"), 'r') as f:
                                blur_infos = json.load(f)

                            for BL in blur_infos:
                                cls_alphas.append(blur_infos[BL]['avg_alpha'])
                                im_p = os.path.join(cls_dir, ins_dir, traj_dir, BL+'_blurred.png')
                                cls_img_paths.append(im_p)

                                # read im_p, erosion, compute avg_alpha_erosion, append to avg_alphas[int(BL)]
                                img = cv2.imread(os.path.join(data_dir, im_p), cv2.IMREAD_UNCHANGED)
                                alpha = img[:,:,3]
                                binary_mask = np.zeros(alpha.shape, dtype=np.uint8)
                                binary_mask[alpha>0] = 255
                                alpha = alpha.astype(np.float32)/255.0
                                erosion = cv2.erode(binary_mask, kernel)
                                alpha_erosion = alpha * erosion.astype(np.float32)/255.0
                                m = alpha_erosion > 0
                                num_pix_obj = np.count_nonzero(m)
                                avg_alpha_erosion = np.sum(alpha_erosion[m])/num_pix_obj
                                # could be that num_pix_obj is 0, then avg_alpha_erosion is nan
                                if np.isnan(avg_alpha_erosion):
                                    avg_alpha_erosion = 0

                                cls_alphas_erosion.append(avg_alpha_erosion)
                                cls_blur_levels.append(len(bins)-1 - bisect.bisect(bins, avg_alpha_erosion))
        # save in cls dir
        np.save(os.path.join(data_dir, cls_dir, 'alphas_origial.npy'), cls_alphas)
        np.save(os.path.join(data_dir, cls_dir, 'alphas_erosion.npy'), cls_alphas_erosion)
        np.save(os.path.join(data_dir, cls_dir, 'blur_levels.npy'), cls_blur_levels)
        # save img_paths as json
        try:
            with open(os.path.join(data_dir, cls_dir, 'img_paths.json'), 'w') as f:
                json.dump(cls_img_paths, f, indent=4)
        except:
            np.save(os.path.join(data_dir, cls_dir, 'img_paths.npy'), cls_img_paths)
        # append to all
        img_paths.extend(cls_img_paths)
        alphas.extend(cls_alphas)
        alphas_erosion.extend(cls_alphas_erosion)
        blur_levels.extend(cls_blur_levels)
        
    # save
    np.save(os.path.join(data_dir, 'stats', 'erosion', 'alphas_original.npy'), alphas)   
    np.save(os.path.join(data_dir, 'stats', 'erosion', 'alphas_erosion.npy'), alphas_erosion)
    np.save(os.path.join(data_dir, 'stats', 'erosion', 'blur_levels.npy'), blur_levels)
    # save img_paths as json
    with open(os.path.join(data_dir, 'stats', 'erosion', 'img_paths.json'), 'w') as f:
        json.dump(img_paths, f, indent=4)


def erosion_alpha_traj_stats(data_dir="/local/home/ronzou/euler/synthetic_data/stats/erosion", threshold=0.0, gap=0.1):
    avg_alphas_erosion = np.load(os.path.join(data_dir, 'avg_alphas_erosion_by_subframe.npy'), allow_pickle=True).item()
    avg_alphas_erosion = avg_alphas_erosion[0]
    with open(os.path.join(data_dir, 'img_paths_by_subframe.json'), 'r') as f:
        img_paths = json.load(f)

    instance_ids = []
    for i in range(len(img_paths)):
        for j in range(len(img_paths[i])):
            cls_id_ij = img_paths[i][j].split('/')[0]      
            if cls_id_ij == '04401088':
                assert False
            ins_id_ij = img_paths[i][j].split('/')[1]
            ins_id_ij = cls_id_ij + '/' + ins_id_ij
            instance_ids.append(ins_id_ij)
    instance_ids = sorted(list(set(instance_ids)))
    
    # both avg_alphas_erosion and img_paths are lists of lists
    # flatten them to make it easier to use, namely concatenate all the lists in avg_alphas_erosion and img_paths
    avg_alpha_flatten = []
    img_paths_flatten = []
    for i in range(len(avg_alphas_erosion)):
        avg_alpha_flatten.extend(avg_alphas_erosion[i])
        img_paths_flatten.extend(img_paths[i])

    # bins = from threshold to 1.0, with gap 0.1
    bins = np.round(np.arange(threshold, 1.0+gap, gap),3).tolist()
    # for each img, get blur level by avg_alpha, and store 
    img_paths_blur_level = [[] for i in range(len(bins)-1)]
    avg_alphas_blur_level = [[] for i in range(len(bins)-1)]
    for i in range(len(avg_alpha_flatten)):
        # find out which bin it belongs to
        b = len(bins)-1 - bisect.bisect(bins, avg_alpha_flatten[i])
        img_paths_blur_level[b].append(img_paths_flatten[i])
        avg_alphas_blur_level[b].append(avg_alpha_flatten[i])
    
    img_paths_without_suffix = ['/'.join(x.split('/')[:-1]) for x in img_paths_flatten]
    
    
    traj_groups = OrderedDict()
    for i, v in enumerate(img_paths_without_suffix):
        try:
            traj_groups[v].append(i)
        except KeyError:
            traj_groups[v] = [i]

    data_dir=data_dir.split('stats')[0]
    
    instance_traj_alpha_stats = {i: [[] for j in range(len(bins)-1)] for i in instance_ids}
    instance_traj_alpha_counts = {i: [0 for j in range(len(bins)-1)] for i in instance_ids}
    

    for instance_id in tqdm(instance_ids):
        for traj_dir in tqdm(range(120)):
            # make traj_dir into 3 digits
            traj_dir = str(traj_dir).zfill(3)

            traj_id = instance_id + '/' + traj_dir
            # img_paths_without_suffix have multiple positions of traj_id, find the indexes
            try:
                idxes = traj_groups[traj_id]
            except KeyError: # if traj_id not in traj_groups, continue
                continue
            
            # avg_alphas of instance_id/traj_dir
            avg_alphas = [avg_alpha_flatten[i] for i in idxes]

            # for each avg_alpha, find out which bin it belongs to
            blur_levels = []
            for a in avg_alphas:
                b = len(bins)-1 - bisect.bisect(bins, a)
                blur_levels.append(b)
            # unique
            blur_levels = list(set(blur_levels))
            # append to instance_traj_alpha_stats
            for b in blur_levels:
                instance_traj_alpha_stats[instance_id][b].append(traj_dir)
                instance_traj_alpha_counts[instance_id][b] += 1

    # change the keys of instance_traj_alpha_stats and instance_traj_alpha_counts, keep split('/')[-1]
    instance_traj_alpha_stats = {k.split('/')[-1]: v for k, v in instance_traj_alpha_stats.items()}
    instance_traj_alpha_counts = {k.split('/')[-1]: v for k, v in instance_traj_alpha_counts.items()}

    # save
    # avg_alphas_blur_level is a list of lists, each list has different length
    # so we save it as a list of np arrays
    avg_alphas_blur_level = [np.array(avg_alphas_blur_level[i]) for i in range(len(avg_alphas_blur_level))]
    # need to use savez
    np.savez(os.path.join(data_dir, 'stats/erosion', 'avg_alphas_erosion_by_blurlevel.npz'), *avg_alphas_blur_level)
    # loading code example:
    # avg_alphas_blur_level = np.load(os.path.join(data_dir, 'avg_alphas_erosion_by_blurlevel.npz'), allow_pickle=True)
    # avg_alphas_blur_level = [avg_alphas_blur_level[i] for i in avg_alphas_blur_level]
    
    # save img_paths_blur_level as json
    with open(os.path.join(data_dir, 'stats/erosion', 'img_paths_erosion_by_blurlevel.json'), 'w') as f:
        json.dump(img_paths_blur_level, f, indent=4)
        
    np.save(os.path.join(data_dir, 'stats/traj_stats', 'instance_traj_alpha_stats.npy'), instance_traj_alpha_stats)
    np.save(os.path.join(data_dir, 'stats/traj_stats', 'instance_traj_alpha_counts.npy'), instance_traj_alpha_counts)
    # loading code example:
    # instance_traj_alpha_stats1 = np.load(os.path.join(data_dir, 'stats', 'instance_traj_alpha_stats.npy'), allow_pickle=True).item()
    # instance_traj_alpha_counts1 = np.load(os.path.join(data_dir, 'stats', 'instance_traj_alpha_counts.npy'), allow_pickle=True).item()

if __name__ == "__main__":
    try:
        erosion_alphas()
    except:
        print('Could not generate statistics for the dataset. (error in running erosion_alphas())')
    
    try:    
        erosion_alphas_distractor()
    except:
        print('Could not generate statistics for the dataset. (error in running erosion_alphas_distractor())')
    
    try:
        erosion_alpha_traj_stats()
    except:
        print('Could not generate statistics for the dataset. (error in running erosion_alpha_traj_stats())')
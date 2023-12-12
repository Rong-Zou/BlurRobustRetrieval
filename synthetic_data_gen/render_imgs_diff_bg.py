import os
import sys
import json
import numpy as np
from datetime import datetime
from glob import glob
import json
# enable importing from current dir when running with Blender
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from settings import *
import render
import utils
from utils import NoIndent, NoIndentWriter, get_logger, print_and_log, check_data, estimate_best_init_rot
import random
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

logger = get_logger(g_outdata_path)
print_and_log("Start rendering", logger)


################################################ PARAMETERS ##################################################
n_frames= 24 # number of subframes
num_categories = 40
num_instances = 40
num_traj_per_instance = 70
num_processes = 1 #8
this_process_id = 1

resolution=(320, 240) # frame resolution
motion_blur_steps = 40 # in each render, splits the render into multiple time steps and accumulates the result
render_subframe_step=4
frustum_depth_range=(-6, -3) # frustum depth range. 
start_end_posistions_max_delta_z=1  # trajectory start-end point pair max depth difference
start_end_posistions_delta_xy_range=(2.1, 2.5) # trajectory start-end point pair projected on image plane, the distance between the projected points must in this range
start_end_orientations_max_diff=np.pi/6 # start angle is random, end angle is start angle + random rotation less than start_end_orientations_max_diff

# for filter
obj_occ_increment_threshold = 0.8 
min_obj_occ_threshold = 0.015 
min_avg_alpha_threshold = 0.3
filter_params = {
    "obj_occ_increment_threshold": obj_occ_increment_threshold,
    "min_obj_occ_threshold": min_obj_occ_threshold,
    "min_avg_alpha_threshold": min_avg_alpha_threshold
}

use_only_instances_with_builtin_textures = True
# otherwise use all instances
use_extra_texture_for_no_texture_instances = 0 
use_extra_texture_for_all_instances = 1 
################################################ PARAMETERS ##################################################


instance_texs = None
if use_only_instances_with_builtin_textures:
    print_and_log("Using only instances with builtin textures.", logger)
    use_extra_texture_for_no_texture_instances = 0
    use_extra_texture_for_all_instances = 0
else:
    instance_texs = []
    assert use_extra_texture_for_no_texture_instances + use_extra_texture_for_all_instances == 1, "One and only one of use_extra_texture_for_no_texture_instances and use_extra_texture_for_all_instances must be True."
    if use_extra_texture_for_no_texture_instances:
        print_and_log("Using all instances, using extra texture for instances without builtin textures.", logger)
    else:
        print_and_log("Using all instances, using extra texture for all instances.", logger)
    
categories_dirs, category_ids, category_names = utils.get_sub_dirs_from_shapenet(g_shapenet_path, num_categories, method="sorted")

remove_idx = category_ids.index("04401088")
categories_dirs.pop(remove_idx)
category_ids.pop(remove_idx)
category_names.pop(remove_idx)
num_categories -= 1

instance_paths = []
out_dirs_each_instance = []
out_dirs_each_instance_each_traj = []

blur_renders = 11 # generated images for each instance, each image has different background
bg_paths_full = glob(os.path.join(g_background_path, "*.png")) #   89733
num_bg_needed = num_categories * num_instances * num_traj_per_instance * blur_renders
repeat_times = num_bg_needed // len(bg_paths_full) + 1
bg_paths_full = bg_paths_full * repeat_times
# shuffle the bg_paths
random.shuffle(bg_paths_full)

tex_paths = glob(os.path.join(g_texture_path, "*/*.jpg"))
# shuffle the tex_paths
random.shuffle(tex_paths)

num_total_seqs = num_categories * num_instances * num_traj_per_instance

assert len(bg_paths_full)*4 >= num_total_seqs, f"Number of background images is less than {num_total_seqs}." # each seq share the same bg

# assume textures can be reused, so we don't need to check the number of textures
    

# number of instances w tex: [2253, 164, 64, 93, 280, 175, 1126, 57, 307, 249, 83, 733, 1230, 72, 69, 34, 1477, 669, 3817, 461, 23, 72, 751, 41, 294, 207, 622, 71, 345, 285, 1216, 349, 1189, 63, 45,116 ,262 ,78 ,178 ,46 ,187 ,390 ,71 ,30 ,1459 ,40 ,129 ,2163 ,169 ,6352 ,854 ,88 ,299 ,1482 ,99]

do_not_select_instances = os.path.join(g_outdata_path, "not_select_instances.json")
# open the file as dict
if os.path.exists(do_not_select_instances):
    with open(do_not_select_instances, "r") as f:
        do_not_select_instances_dict = json.load(f)
else:
    # empty list
    do_not_select_instances_dict = {}


for i, category_dir in enumerate(categories_dirs):
    # get the category name
    category_id = category_ids[i]
    category_name = category_names[i]

    # load the instance.json file in the category folder
    instance_with_textures = None
    with open(os.path.join(category_dir, "instances_with_textures.json"), "r") as instance_file:
        instance_with_textures = json.load(instance_file)

    # remove instances in do_not_select_instances[category_id] from instance_with_textures["instanceWithTexturesNames"]
    if category_id in do_not_select_instances_dict:
        for instance_name in do_not_select_instances_dict[category_id]:
            if instance_name in instance_with_textures["instanceWithTexturesNames"]:
                instance_with_textures["instanceWithTexturesNames"].remove(instance_name)
                instance_with_textures["numInstancesWithTextures"] -= 1
                


    if use_only_instances_with_builtin_textures:
        print_and_log(f"Category {category_id} ({category_name}) has {instance_with_textures['numInstancesWithTextures']} instances.", logger)
        
        assert instance_with_textures["numInstancesWithTextures"] > num_instances, f"Category {category_dir} has less than {num_instances} instances."
        
        # randomly select 10 instances from the category
        instance_names = random.sample(instance_with_textures["instanceWithTexturesNames"], num_instances)
        # generate the object path and output dir for each instance, 
        for instance_name in instance_names:
            instance_path = os.path.join(category_dir, instance_name, "models", "model_normalized.obj")
            out_dir = os.path.join(os.path.basename(g_outdata_path), category_id, instance_name)
            instance_paths.append(instance_path)
            out_dirs_each_instance.append(out_dir)
            for t in range(num_traj_per_instance):
                out_dirs_each_instance_each_traj.append(os.path.join(out_dir, str(t).zfill(3)))
    else:
        all_instance_dict = None
        with open(os.path.join(category_dir, "instances.json"), "r") as instance_file:
            all_instance_dict = json.load(instance_file)
        print_and_log(f"Category {category_id} ({category_name}) has {all_instance_dict['numInstances']} instances.", logger)
        
        assert all_instance_dict["numInstances"] > num_instances, f"Category {category_dir} has less than {num_instances} instances."
        
        # randomly select 10 instances from the category
        instance_names = random.sample(all_instance_dict["instanceNames"], num_instances)
        # generate the object path and output dir for each instance, 
        for instance_name in instance_names:
            instance_path = os.path.join(category_dir, instance_name, "models", "model_normalized.obj")
            if use_extra_texture_for_no_texture_instances:
                if instance_name in instance_with_textures["instanceWithTexturesNames"]:
                    instance_texs.append(None)
                else:
                    instance_texs.append(random.choice(tex_paths))
            elif use_extra_texture_for_all_instances:
                instance_texs.append(random.choice(tex_paths))

            out_dir = os.path.join(os.path.basename(g_outdata_path), category_id, instance_name)
            instance_paths.append(instance_path)
            out_dirs_each_instance.append(out_dir)
            for t in range(num_traj_per_instance):
                out_dirs_each_instance_each_traj.append(os.path.join(out_dir, str(t).zfill(3)))
        
# bg_paths_each_instance = random.choices(bg_paths, k=len(instance_paths)) # randomly select 100 background images with replacement
n_sequences = len(instance_paths)

n_sequences_per_process = n_sequences // num_processes

n_bgs_per_process = n_sequences_per_process * num_traj_per_instance * blur_renders

this_process_seq_start_idx = n_sequences_per_process*(this_process_id-1)
this_process_seq_end_idx = n_sequences_per_process*this_process_id


# save all variables in a json file, use NoIndent for lists and tuples
params = {
    "blur_renders": blur_renders,
    "num_categories": num_categories,
    "num_instances": num_instances,
    "num_traj_per_instance": num_traj_per_instance,
    "resolution": NoIndent(resolution),
    "n_frames": n_frames,
    "motion_blur_steps": motion_blur_steps,
    "frustum_depth_range": NoIndent(frustum_depth_range),
    "start_end_posistions_max_delta_z": start_end_posistions_max_delta_z,
    "start_end_posistions_delta_xy_range": NoIndent(start_end_posistions_delta_xy_range),
    "start_end_orientations_max_diff": start_end_orientations_max_diff * 180 / np.pi,
    "category_id_names": NoIndent(list(zip(category_ids, category_names))),
    "instance_paths": instance_paths[this_process_seq_start_idx:this_process_seq_end_idx],
    "instance_texs": instance_texs,
    "use_only_instances_with_builtin_textures": use_only_instances_with_builtin_textures,
    "use_extra_texture_for_no_texture_instances": use_extra_texture_for_no_texture_instances,
    "use_extra_texture_for_all_instances": use_extra_texture_for_all_instances,
    "filter_params": filter_params,
}    
    
with open(os.path.join(g_outdata_path, "params{}.json".format(this_process_id)), 'w') as fp:
    json.dump(params, fp, cls=NoIndentWriter, sort_keys=False, indent=4)
    fp.write('\n') 
     
render.init(n_frames, resolution, motion_blur_steps, (0.6, 0.6, 0.6)) # Color of the background = (0.6, 0.6, 0.6)
frustum = render.Frustum(frustum_depth_range, resolution)

time = datetime.now()

instance_param_dict = {}

try_for_each_instance_times = 5
sequences = {"num_comp_incomp_failed": [], "complete": [], "incomplete": [], "failed": []}

tex_path_this_instance = None

bg_path_flag = this_process_id
bg_paths = bg_paths_full[(bg_path_flag-1)*n_bgs_per_process : bg_path_flag*n_bgs_per_process]


for i in range(this_process_seq_start_idx, this_process_seq_end_idx):

    for t in range(num_traj_per_instance):
        # take blur_renders number of bg images from bg_paths, and delete them from bg_paths
        bg_paths_this_instance_b = bg_paths[:blur_renders]
        bg_paths = bg_paths[blur_renders:]

        print_and_log(f"[START] Rendering sequence {t + 1} / {num_traj_per_instance} of instance {i + 1} / {n_sequences} in  {out_dirs_each_instance_each_traj[i * num_traj_per_instance + t]}", logger)
    
        # if out_dirs_each_instance_each_traj already exists and it is not empty, skip this trajectory of this instance
        if os.path.exists(out_dirs_each_instance_each_traj[i * num_traj_per_instance + t]) and os.listdir(out_dirs_each_instance_each_traj[i * num_traj_per_instance + t]):
            print_and_log(f"Skipping sequence {t + 1} of instance {i + 1} because it already exists.", logger)
            continue
        
        else:
            blur_infos = {}
            instance_param_dict = {}
            blur_render_num_frames_ = []
            consective_failed = 0
            
            current_try = 0
            while current_try < try_for_each_instance_times:
                loc_pair = frustum.gen_point_pair(start_end_posistions_max_delta_z, start_end_posistions_delta_xy_range)
                if loc_pair is None:
                    print_and_log(f"Generating start_end_position_pair for sequence {t+1} of instance {i+1} failed at try {current_try + 1} of {try_for_each_instance_times}.", logger)
                    # print_and_log(f"Generating start_end_position_pair for sequence {out_dirs_each_instance[i]} failed at try {current_try + 1} of {try_for_each_instance_times}.", logger)
                    current_try += 1
                    continue
                
                rot_start, obj_size_info = estimate_best_init_rot(instance_paths[i])
                rot_end = rot_start + start_end_orientations_max_diff * (np.random.rand(3) * 2 - 1) 
                rot_pair = (list(rot_start), list(rot_end))
                
                utils.assure_dir(out_dirs_each_instance_each_traj[i * num_traj_per_instance + t])
                # valid = render.render(out_dirs_each_instance[i], instance_paths[i], tex_path_this_instance, bg_paths_this_instance, loc_pair, rot_pair, blur_subframe_indexes, use_filter=True, render_subframe_step=render_subframe_step, filter_params=filter_params, logger=logger)
                valid, blur_infos = render.render_diff_bg(out_dirs_each_instance_each_traj[i * num_traj_per_instance + t], 
                                                          instance_paths[i], 
                                                          tex_path_this_instance, 
                                                          bg_paths_this_instance_b, 
                                                          loc_pair, rot_pair, 
                                                          blurs=None, use_filter=True, 
                                                          render_subframe_step=render_subframe_step, 
                                                          filter_params=filter_params, 
                                                          logger=logger, 
                                                          bg_path_flag = bg_path_flag,
                                                          num_frames=n_frames)
                
                if not valid:
                    print_and_log(f"Rendering sequence {t+1} of instance {i+1} failed.", logger)
                    current_try += 1
                    # no need to run the following code
                    continue
                    # consective_failed += 1
                    # break
                else:

                    rotation_srat_in_degrees = np.array(rot_pair[0]) * 180 / np.pi
                    rotation_end_in_degrees = np.array(rot_pair[1]) * 180 / np.pi
                    location_diff = np.array(loc_pair[1]) - np.array(loc_pair[0])
                    location_xy_dist = np.sqrt(location_diff[0] ** 2 + location_diff[1] ** 2)

                    rot_pair_in_degrees = (list(rotation_srat_in_degrees), list(rotation_end_in_degrees))
                    obj_sizes = obj_size_info["xyz"]
                    obj_areas = obj_size_info["rectangle_area_xy_xz_yz"]
                    obj_max_area_side = obj_size_info["max_area"]
                    instance_param_dict = {
                        "location_start_end": NoIndent(list(loc_pair)),
                        "location_diff": NoIndent(list(location_diff)),
                        "location_xy_dist": location_xy_dist,
                        "rotation_start_end": NoIndent(list(rot_pair)),
                        "rotation_start_end_in_degrees": NoIndent(list(rot_pair_in_degrees)),
                        "rotation_diff_in_degrees": NoIndent(list(np.array(rot_pair_in_degrees[1]) - np.array(rot_pair_in_degrees[0]))),
                        "obj_size_xyz": NoIndent(list(obj_sizes)),
                        "obj_area_xy_xz_yz": NoIndent(list(obj_areas)),
                        "obj_max_area_side": obj_max_area_side,
                        "bg_path": bg_paths_this_instance_b,
                        "bg_path_flag": "upper left" if bg_path_flag == 1 else "upper right" if bg_path_flag == 2 else "lower left" if bg_path_flag == 3 else "lower right",
                    }
                    print_and_log(f"Rendering sequence {t+1} of instance {i+1} succeeded at try {current_try + 1} of {try_for_each_instance_times}.", logger)
                    break
                
            if valid:
                with open(os.path.join(out_dirs_each_instance_each_traj[i * num_traj_per_instance + t], "instance_params.json"), "w") as fp:
                    json.dump(instance_param_dict, fp, cls=NoIndentWriter, sort_keys=False, indent=4)
                    fp.write('\n')  # Add a newline to very end (optional).

                with open(os.path.join(out_dirs_each_instance_each_traj[i * num_traj_per_instance + t], "blur_infos.json"), "w") as fp:
                    json.dump(blur_infos, fp, cls=NoIndentWriter, sort_keys=False, indent=4)
                    fp.write('\n')
        

duration = datetime.now() - time
message = f"Rendered {num_total_seqs} sequences in {duration}, {duration.total_seconds() / num_total_seqs:.2f}s per sequence"
complete_folders, incomplete_folders, num_blur_renders_for_incomplete_folders = check_data(g_outdata_path)
print_and_log(message, logger)


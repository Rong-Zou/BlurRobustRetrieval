import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import random
import argparse
import glob
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import *
from models import *
from loader import *
from metrics import *
import gc
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
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def get_distractor_descriptors(model, distractor_loader, device, mode, save_load_path=None):
    P = save_load_path
    distractor_loader.dataset.set_dataset_type(mode)
    print('Distractor mode: {}'.format(mode))
    print('Distractor set contains {} images'.format(len(distractor_loader.dataset)))
    
    # if mode is a list
    if isinstance(mode, list):
        # for example mode is [0,1,2], make it to be '0_1_2'
        mode = '_'.join([str(m) for m in mode])
    
    if os.path.exists(os.path.join(P, f'distractor_descriptors_{mode}.pt')):
        print("Distractor descriptors already exist. Loading descriptors from the disk")
        descriptors = torch.load(os.path.join(P, f'distractor_descriptors_{mode}.pt'), map_location=device)
        blur_levels = torch.load(os.path.join(P, f'distractor_blur_levels_{mode}.pt'), map_location=device)

        with open(os.path.join(P, f'distractor_img_paths_{mode}.json'), 'r') as f:
            distractor_img_paths = json.load(f)
        img_paths = distractor_img_paths['distractor_img_paths']
                
    else:
        assure_dir(P)
        model.to(device)
        model.eval()
        descriptors = []
        blur_levels = []
        img_paths = []
        with torch.no_grad():
            for i, (images, img_path, blur_level) in tqdm(enumerate(distractor_loader), total=len(distractor_loader)):
                images = images.to(device)
                descriptors_ = model(images, only_descriptor=True)
                descriptors_ = F.normalize(descriptors_, p=2, dim=1)
                descriptors.append(descriptors_)
                blur_levels.append(blur_level)
                
                img_paths.extend(img_path)
            descriptors = torch.cat(descriptors, dim=0)
            blur_levels = torch.cat(blur_levels, dim=0)
        
        torch.save(descriptors, os.path.join(P, f'distractor_descriptors_{mode}.pt'))
        torch.save(blur_levels, os.path.join(P, f'distractor_blur_levels_{mode}.pt'))
        # save the image paths to the disk for future use, save to a json file
        distractor_img_paths_ = {'len_distractor': len(img_paths), 'distractor_img_paths': img_paths}
        with open(os.path.join(P, f'distractor_img_paths_{mode}.json'), 'w') as f:
            json.dump(distractor_img_paths_, f, indent=4)
            
    return descriptors, blur_levels, img_paths


def get_db_q_descriptors(model, db_q_loader, device, q_db_mode='ss', save_load_path=None, no_per_BL_db=False, continuous_BL=False):
    

    q_s_m = q_db_mode[0]
    db_s_m = q_db_mode[1]
    assert q_s_m in ['s', 'm']
    assert db_s_m in ['s', 'm']
    P = save_load_path
    if q_s_m == 's':
        q_id = 'sharp'
    elif q_s_m == 'm':
        q_id = 'mixed'
    if db_s_m == 's':
        db_id = 'sharp'
    elif db_s_m == 'm':
        db_id = 'mixed'

    # if paths exists, load from the disk
    if os.path.exists(os.path.join(P, f'queries_descriptors_{q_id}.pt')):
        print("Query descriptors already exist. Loading descriptors from the disk")
        # load from the disk
        queries_descriptors = torch.load(os.path.join(P, f'queries_descriptors_{q_id}.pt'), map_location=device)
        queries_labels = torch.load(os.path.join(P, f'queries_labels_{q_id}.pt'), map_location=device)
        queries_blur_levels = torch.load(os.path.join(P, f'queries_blur_levels_{q_id}.pt'), map_location=device)
        
        with open(os.path.join(P, f'q_img_paths_{q_id}.json'), 'r') as f:
            q_img_paths = json.load(f)
        q_img_paths = q_img_paths['q_img_paths']
    else:
        assure_dir(P)
        model.to(device)
        model.eval()
        db_q_loader.dataset.set_dataset_type(('q', q_s_m))
        queries_descriptors, queries_labels, queries_blur_levels, q_img_paths = get_descriptor(model, db_q_loader, device, continuous_BL)
        
        torch.save(queries_descriptors, os.path.join(P, f'queries_descriptors_{q_id}.pt'))
        
        torch.save(queries_labels, os.path.join(P, f'queries_labels_{q_id}.pt'))
        if queries_blur_levels is not None:
            torch.save(queries_blur_levels, os.path.join(P, f'queries_blur_levels_{q_id}.pt'))
        # save the image paths to the disk for future use, save to a json file
        q_img_paths_ = {'len_q': len(q_img_paths), 'q_img_paths': q_img_paths}
        with open(os.path.join(P, f'q_img_paths_{q_id}.json'), 'w') as f:
            json.dump(q_img_paths_, f, indent=4)
    
    if os.path.exists(os.path.join(P, f'database_descriptors_{db_id}.pt')):
        print("Database descriptors already exist. Loading descriptors from the disk")
        database_descriptors = torch.load(os.path.join(P, f'database_descriptors_{db_id}.pt'), map_location=device)
        database_labels = torch.load(os.path.join(P, f'database_labels_{db_id}.pt'), map_location=device)
        database_blur_levels = torch.load(os.path.join(P, f'database_blur_levels_{db_id}.pt'), map_location=device)
        
        with open(os.path.join(P, f'db_img_paths_{db_id}.json'), 'r') as f:
            db_img_paths = json.load(f)
        db_img_paths = db_img_paths['db_img_paths']
    else:
        assure_dir(P)
        model.to(device)
        model.eval()
        db_q_loader.dataset.set_dataset_type(('db', db_s_m))
        database_descriptors, database_labels, database_blur_levels, db_img_paths = get_descriptor(model, db_q_loader, device, continuous_BL)
        
        torch.save(database_descriptors, os.path.join(P, f'database_descriptors_{db_id}.pt'))
        torch.save(database_labels, os.path.join(P, f'database_labels_{db_id}.pt'))
        if database_blur_levels is not None:
            torch.save(database_blur_levels, os.path.join(P, f'database_blur_levels_{db_id}.pt'))
        db_img_paths_ = {'len_db': len(db_img_paths), 'db_img_paths': db_img_paths}
        with open(os.path.join(P, f'db_img_paths_{db_id}.json'), 'w') as f:
            json.dump(db_img_paths_, f, indent=4)

    if no_per_BL_db:
        return queries_descriptors, queries_labels, queries_blur_levels, database_descriptors, database_labels, database_blur_levels, q_img_paths, db_img_paths
        
    if os.path.exists(os.path.join(P, f'database_v2_descriptors_{db_id}.pt')):
        print("Database v2 descriptors already exist. Loading descriptors from the disk")
        database_v2_descriptors = torch.load(os.path.join(P, f'database_v2_descriptors_{db_id}.pt'), map_location=device)
        database_v2_labels = torch.load(os.path.join(P, f'database_v2_labels_{db_id}.pt'), map_location=device)
        database_v2_blur_levels = torch.load(os.path.join(P, f'database_v2_blur_levels_{db_id}.pt'), map_location=device)
        
        with open(os.path.join(P, f'db_v2_img_paths_{db_id}.json'), 'r') as f:
            db_v2_img_paths = json.load(f)
        db_v2_img_paths = db_v2_img_paths['db_v2_img_paths']
    else:
        assure_dir(P)
        model.to(device)
        model.eval()
        
        db_q_loader.dataset.set_dataset_type(('db_v2', db_s_m))
        database_v2_descriptors, database_v2_labels, database_v2_blur_levels, db_v2_img_paths = get_descriptor(model, db_q_loader, device, continuous_BL)
        
        torch.save(database_v2_descriptors, os.path.join(P, f'database_v2_descriptors_{db_id}.pt'))
        
        torch.save(database_v2_labels, os.path.join(P, f'database_v2_labels_{db_id}.pt'))
        
        if database_v2_blur_levels is not None:
            torch.save(database_v2_blur_levels, os.path.join(P, f'database_v2_blur_levels_{db_id}.pt'))
        db_v2_img_paths_ = {'len_db_v2': len(db_v2_img_paths), 'db_v2_img_paths': db_v2_img_paths}
        with open(os.path.join(P, f'db_v2_img_paths_{db_id}.json'), 'w') as f:
            json.dump(db_v2_img_paths_, f, indent=4)


    return queries_descriptors, queries_labels, queries_blur_levels, database_descriptors, database_labels, database_blur_levels, q_img_paths, db_img_paths, database_v2_descriptors, database_v2_labels, database_v2_blur_levels, db_v2_img_paths

def get_descriptor(model, loader, device, continuous_BL=False):
    model.to(device)
    model.eval()
    img_paths=[]
    bins = np.round(np.arange(0, 1.1, 0.1),3).tolist()

    with torch.no_grad():
        for i, (images, labels, img_path, blur_level) in tqdm(enumerate(loader), total=len(loader)):
                images = images.to(device) # [B, 3, H, W]
                descriptors = model(images, only_descriptor=True) # [B, descriptor_size]
                descriptors = F.normalize(descriptors, p=2, dim=1) # [B, descriptor_size]

                labels = labels.to(device) # [B, num_instances_in_testset]
                blur_level = None if blur_level[0] == -1 else blur_level.to(device) # [B]
                img_paths.extend(img_path)
                if i == 0:
                    D = descriptors
                    L = labels
                    B = blur_level
                else:
                    D = torch.cat((D, descriptors), 0) # [N, descriptor_size]
                    L = torch.cat((L, labels), 0) # [N, num_instances_in_testset]
                    if blur_level is not None:
                        B = torch.cat((B, blur_level), 0) # [N]
    return D, L, B, img_paths

def parse_args():
    parser = argparse.ArgumentParser(description='Blur Retrieval Testing')
    # -----------------------------DATA-----------------------------
    parser.add_argument('--syn_data_dir', type=str, default='./overfit_test', help='path to dataset')
    parser.add_argument('--train_log_dir', type=str, default='./results/overfit_test/four_ins', help='path to save training logs')
    parser.add_argument('--train_folder', type=str, default='train', help='train folder name')
    parser.add_argument('--val_folder', type=str, default='val', help='val folder name')
    parser.add_argument('--test_folder', type=str, default='test', help='test folder name')
    
    parser.add_argument('--real_data_dir', type=str, default='./real_data/annotated', help='path to real dataset')

    parser.add_argument('--train_ratio', type=float, default=0.8, help='train split')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='val split')    
    parser.add_argument('--database_ratio', type=float, default=10/12, help='database in test data')

    parser.add_argument('--dataset_transforms', type=str, default=None, help='dataset transforms')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataloader')

    # -----------------------------MODEL-----------------------------
    parser.add_argument('--pred_cls', type=bool, default=False, help='whether to predict classes')
    parser.add_argument('--pred_blur_level', type=bool, default=False, help='whether to predict blur levels')
    parser.add_argument('--pred_descriptor', type=bool, default=True, help='whether to predict descriptors')
    parser.add_argument('--pred_loc', type=bool, default=False, help='whether to predict object bounding box')
    parser.add_argument('--pred_blur_level_type', type=str, default=None, help='blur level type, discrete or continuous_erosion or continuous_original')

    parser.add_argument('--descriptor_size', type=int, default=128, help='descriptor size')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--num_blur_levels', type=int, default=24, help='number of blur levels')
    parser.add_argument('--image_height', type=int, default=240, help='image height')
    parser.add_argument('--image_width', type=int, default=320, help='image width')
    
    parser.add_argument('--encoder_pretrained', type=bool, default=True, help='whether to use pretrained model')
    parser.add_argument('--encoder_norm_type', type=str, default=None, help='encoder normalization type, None, BatchNorm, InstanceNorm, LayerNorm')

    # ------------------------------TESTING-----------------------------
    parser.add_argument('--test_synthetic_data', type=bool, default=True, help='whether to test on synthetic data')
    parser.add_argument('--test_log_dir', type=str, default='./results/overfit_test/test_results', help='path to save test logs')
    parser.add_argument('--test_results_dir', type=str, default='./results/overfit_test/test_results', help='path to save test results')
    parser.add_argument('--test_batch_size', type=int, default=32, help='test batch size')
    parser.add_argument('--top_k', type=int, default=500, help='test batch size')
    parser.add_argument('--test_save_results_imgs', type=bool, default=False, help='whether to save test results images, if False, will only save the results txt file')
    parser.add_argument('--mAP_batch_compute', type=bool, default=False, help='whether to compute mAP in batch')
    parser.add_argument('--take_distractors', type=bool, default=False, help='take distractors')
    parser.add_argument('--test_per_blur_level_db', type=bool, default=False, help='test per BL database')
    
    # -------------------------------OTHERS-------------------------------
    parser.add_argument('--debug', type=bool, default=False, help='whether to debug')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    args = parse_args()
    args.syn_data_dir = '/cluster/project/infk/cvg/students/rzoran/synthetic_data'
    if not os.path.exists(args.syn_data_dir):
        args.syn_data_dir = '/local/home/ronzou/euler/synthetic_data'
        
    args.real_data_dir = './real_data'

    args.test_synthetic_data = False
    args.take_distractors = False 
    args.test_save_results_imgs = False
    args.test_per_blur_level_db = False

    model_dirs = [
        './weights/des_w1.0cls_w0.1bbox_w10.0continuous_erosion_w1.0'
        ]
    model_paths = []

    failed_model_paths = []
    alredy_tested_model_paths = []

    for model_dir in model_dirs:
        model_paths += glob(os.path.join(model_dir, 'train_results', 'best_*.pkl'))
    model_paths = sorted(model_paths)

    dataset_transforms = args.dataset_transforms
    
    image_size = [args.image_height, args.image_width]
    
    for model_idx, model_path in enumerate(model_paths):
        print('"++++++++++++++++++++++ MODEL: {}/{} ++++++++++++++++++++++"'.format(model_idx+1, len(model_paths)))
        print('model_path: {}'.format(model_path))

        if 'continuous_erosion' in model_path:
            args.pred_blur_level = True
            args.pred_blur_level_type = 'continuous_erosion'
        elif 'continuous_original' in model_path:
            args.pred_blur_level = True
            args.pred_blur_level_type = 'continuous_original'
        elif 'discrete' in model_path:
            args.pred_blur_level = True
            args.pred_blur_level_type = 'discrete'
        else:
            args.pred_blur_level = False
            args.pred_blur_level_type = None

        if 'bbox' in model_path:
            args.pred_loc = True
        else:
            args.pred_loc = False

        if 'cls' in model_path:
            args.pred_cls = True
            args.num_classes = 792
            args.arcface_s = 30.0
            args.arcface_m = 0.15
        else:
            args.pred_cls = False
            args.num_classes = None

        
        print('args.pred_blur_level: {}'.format(args.pred_blur_level))
        print('args.pred_blur_level_type: {}'.format(args.pred_blur_level_type))
        print('args.pred_loc: {}'.format(args.pred_loc))
        print('args.pred_cls: {}'.format(args.pred_cls))

        
        model_name = model_path.split('/')[-1].split('.')[0]

        print('model_name: {}'.format(model_name))

        model = BlurRetrievalNet(args.num_classes if args.pred_cls else None,
                                args.num_blur_levels if args.pred_blur_level else None,
                                args.descriptor_size,
                                image_size,
                                args.pred_loc,
                                args.encoder_pretrained,
                                args.encoder_norm_type,
                                pred_blur_level_type = args.pred_blur_level_type
                                )
        model.to(device)

        try:
            ckpt = torch.load(model_path, map_location=device)
        except:
            # if still fails, skip and append to failed_model_paths
            failed_model_paths.append(model_path)
            print('Failed to load model from {}, skip.'.format(model_path))
            continue
        
        model.load_state_dict(ckpt["model_state"])

        args.test_log_dir = model_path.replace('train_results', 'test_results').split('.pkl')[0]
        assure_dir(args.test_log_dir)

        if args.test_synthetic_data:
            
            train_folder = args.train_folder
            val_folder = args.val_folder
            test_folder = args.test_folder

            data_split_dict = os.path.join(args.syn_data_dir, 'stats/loader/data_split_info.json')
            if os.path.exists(data_split_dict):
                print_str = 'Loading data split info from {}'.format(data_split_dict)
                data_split_dict = json.load(open(data_split_dict, 'r'))
                cls_ids = data_split_dict['cls_ids']
                test_instance_folders = data_split_dict['test_instance_folders']

                if args.debug:
                    test_instance_folders = random.sample(test_instance_folders, 3)
                print(print_str)
            else:
                print_str = 'Creating data split info and save it to {}'.format(os.path.join(args.train_log_dir, 'data_split_info.json'))
                cls_ids, train_instance_folders, val_instance_folders, test_instance_folders, test_database, test_query_set, val_database, val_query = split_data(args.syn_data_dir, train_dir = train_folder, val_dir = val_folder, test_dir = test_folder,
                                                                                                                train_ratio=args.train_ratio, val_ratio=args.val_ratio, database_ratio=args.database_ratio,
                                                                                                                save_dir=args.train_log_dir)
                print(print_str)       
        
            blur_levels = [[0, 1, 2, 3, 4, 5]]
            
            for blur_levels_ in blur_levels:
                print('blur_levels_: {}'.format(blur_levels_))
                args.take_blur_levels = blur_levels_
                id_ = '_'.join([str(i) for i in args.take_blur_levels])
                
                args.test_results_dir = os.path.join(args.test_log_dir, 'BL_{}'.format(id_))
                
                assure_dir(args.test_results_dir)
                test_database_query = dataset_database_query(test_instance_folders, 
                                                            normalize=True, transform=dataset_transforms, 
                                                            database_ratio=args.database_ratio, 
                                                            take_blur_levels=args.take_blur_levels,
                                                            save_load_imgs_dir=os.path.join(args.syn_data_dir, 'stats/loader/test/BL_{}_with_dbv2'.format(id_)),
                                                            )
                
                # create the dataloader
                test_database_query_loader = DataLoader(test_database_query, batch_size=args.test_batch_size,
                                                        shuffle=False, num_workers=args.num_workers)   
                if args.take_distractors:
                    distractors = dataset_distractor(normalize=True, transform=dataset_transforms)
                    distractors_loader = DataLoader(distractors, batch_size=args.test_batch_size,
                                                        shuffle=False, num_workers=args.num_workers)

                test_database_query.set_dataset_type(("db", "s"))
                test_dbs_len = len(test_database_query)
                test_database_query.set_dataset_type(("db", "m"))
                test_dbm_len = len(test_database_query)

                test_database_query.set_dataset_type(("q", "s"))
                test_qs_len = len(test_database_query)
                test_database_query.set_dataset_type(("q", "m"))
                test_qm_len = len(test_database_query)
                print_str = 'test database and query set loaded, num imgs: \n db_sharp: {}, db_mixed: {}, q_sharp: {}, q_mixed: {}'.format(test_dbs_len, test_dbm_len, test_qs_len, test_qm_len)
                
                print(print_str)

                for q_db_mode in ['mm']:

                    print("Getting descriptors for {}...".format(q_db_mode))
                    values = get_db_q_descriptors(model, test_database_query_loader, device, q_db_mode, args.test_results_dir, no_per_BL_db=False)
                    queries_descriptors, queries_labels, queries_blur_levels, database_descriptors, database_labels, database_blur_levels, q_img_paths, db_img_paths, database_v2_descriptors, database_v2_labels, database_v2_blur_levels, db_v2_img_paths = values

                    assure_dir(os.path.join(args.test_results_dir, q_db_mode, 'db_blur_level_all'))

                    print("Get query, database, and database_v2 descriptors done!")
                    
                    nom_db = torch.norm(database_descriptors, p=2, dim=1)
                    nom_q = torch.norm(queries_descriptors, p=2, dim=1)
                    compute_mAP(database_descriptors, queries_descriptors, 
                                database_labels, queries_labels, 
                                database_blur_levels, queries_blur_levels,
                                top_ks=[None],
                                batch_compute=args.mAP_batch_compute,
                                results_path=os.path.join(args.test_results_dir, q_db_mode, 'db_blur_level_all'),
                                query_image_paths=q_img_paths, database_image_paths=db_img_paths,
                                save_images=args.test_save_results_imgs,
                                save_num_imges_per_blur_level=10,
                                consider_only_db_blur_level=None)
                                        
                    if args.take_distractors:
                        print("Getting distractor descriptors...")
                        distractor_mode = q_db_mode[1]
                        if distractor_mode == 'm':
                            distractor_mode = args.take_blur_levels
                        distractor_descriptors, distractor_blur_levels, distractor_img_paths = get_distractor_descriptors(model, distractors_loader, device, distractor_mode,
                                                                                                                                                            args.test_results_dir)
                        assure_dir(os.path.join(args.test_results_dir, q_db_mode, 'db_blur_level_all', 'with_distractors'))
                        print("Get distractor descriptors done!")
                        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                        # concat database and distractors
                        database_descriptors = torch.cat((database_descriptors, distractor_descriptors), dim=0)
                        database_blur_levels = torch.cat((database_blur_levels, distractor_blur_levels.to(device)), dim=0)
                        
                        # distractor_labels is all zeros, shape is [num_distractors, len(instance_folders))]
                        distractor_labels = torch.zeros((distractor_descriptors.shape[0], len(test_instance_folders))).to(device)
                        database_labels = torch.cat((database_labels, distractor_labels), dim=0)

                        db_img_paths = db_img_paths + distractor_img_paths
                        
                        compute_mAP(database_descriptors, queries_descriptors, 
                                    database_labels, queries_labels, 
                                    database_blur_levels, queries_blur_levels,
                                    top_ks=[100],
                                    batch_compute=args.mAP_batch_compute,
                                    results_path=os.path.join(args.test_results_dir, q_db_mode, 'db_blur_level_all', 'with_distractors'),
                                    query_image_paths=q_img_paths, database_image_paths=db_img_paths,
                                    save_images=args.test_save_results_imgs,
                                    save_num_imges_per_blur_level=10,
                                    consider_only_db_blur_level=None)
                        
                    if args.test_per_blur_level_db:
                        
                        # compute mAP for each blur level
                        for db_blur_level in blur_levels_:

                            assure_dir(os.path.join(args.test_results_dir, q_db_mode, 'db_blur_level_{}'.format(db_blur_level)))
                            compute_mAP(database_v2_descriptors, queries_descriptors, 
                                        database_v2_labels, queries_labels, 
                                        database_v2_blur_levels, queries_blur_levels,
                                        top_ks=[None],
                                        batch_compute=args.mAP_batch_compute,
                                        results_path=os.path.join(args.test_results_dir, q_db_mode, 'db_blur_level_{}'.format(db_blur_level)),
                                        query_image_paths=q_img_paths, database_image_paths=db_v2_img_paths,
                                        save_images=args.test_save_results_imgs,
                                        save_num_imges_per_blur_level=10,
                                        consider_only_db_blur_level=db_blur_level)
                            
                            if args.take_distractors:
                                assure_dir(os.path.join(args.test_results_dir, q_db_mode, 'db_blur_level_{}'.format(db_blur_level), 'with_distractors'))
                                # get the indexes of distractor data with blur level db_blur_level
                                distractor_indexes = torch.where(distractor_blur_levels == db_blur_level)[0]
                                if len(distractor_indexes) > 0:
                                    # get distractor_descriptors, distractor_blur_levels, distractor_img_paths 
                                    distractor_descriptors_tmp = distractor_descriptors[distractor_indexes]
                                    distractor_blur_levels_tmp = distractor_blur_levels[distractor_indexes]

                                    distractor_img_paths_tmp = [distractor_img_paths[i] for i in distractor_indexes]
                                    # concat database and distractors, v2
                                    database_v2_descriptors_tmp = torch.cat((database_v2_descriptors, distractor_descriptors_tmp), dim=0)
                                    database_v2_blur_levels_tmp = torch.cat((database_v2_blur_levels, distractor_blur_levels_tmp.to(device)), dim=0)

                                    # distractor_labels is all zeros, shape is [num_distractors, len(instance_folders))]
                                    distractor_labels_tmp = torch.zeros((distractor_descriptors_tmp.shape[0], len(test_instance_folders))).to(device)
                                    database_v2_labels_tmp = torch.cat((database_v2_labels, distractor_labels_tmp), dim=0)

                                    db_v2_img_paths_tmp = db_v2_img_paths + distractor_img_paths_tmp

                                    compute_mAP(database_v2_descriptors_tmp, queries_descriptors, 
                                                database_v2_labels_tmp, queries_labels, 
                                                database_v2_blur_levels_tmp, queries_blur_levels,
                                                top_ks=[100],
                                                batch_compute=args.mAP_batch_compute,
                                                results_path=os.path.join(args.test_results_dir, q_db_mode, 'db_blur_level_{}'.format(db_blur_level), 'with_distractors'),
                                                query_image_paths=q_img_paths, database_image_paths=db_v2_img_paths_tmp,
                                                save_images=args.test_save_results_imgs,
                                                save_num_imges_per_blur_level=10,
                                                consider_only_db_blur_level=db_blur_level)

        else:
            args.test_results_dir = os.path.join(args.test_log_dir, 'real_data')
            assure_dir(args.test_results_dir)
            q_db_mode = 'mm'
            
            test_database_query = dataset_database_query_real(data_dir=args.real_data_dir,
                                                              normalize=True, transform=dataset_transforms)

            # ###create the dataloader
            test_database_query_loader = DataLoader(test_database_query, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)
            test_database_query.set_dataset_type('db')
            test_db_len = len(test_database_query)
            test_database_query.set_dataset_type('q')
            test_q_len = len(test_database_query)
            print_str = 'test database and query set loaded, num imgs: \n db: {}, q: {}'.format(test_db_len, test_q_len)
            print(print_str)

            # test_database_query_loader = None
            print("Getting descriptors for {}...".format(q_db_mode))
            queries_descriptors, queries_labels, queries_blur_levels, database_descriptors, database_labels, database_blur_levels, q_img_paths, db_img_paths= get_db_q_descriptors(model, test_database_query_loader, device, q_db_mode, args.test_results_dir, no_per_BL_db=True, continuous_BL=True)
            print("Get query, database descriptors done!")
            
            assure_dir(os.path.join(args.test_results_dir, q_db_mode))
            nom_db = torch.norm(database_descriptors, p=2, dim=1)
            nom_q = torch.norm(queries_descriptors, p=2, dim=1)
            print('Computing mAP...')
            compute_mAP(database_descriptors, queries_descriptors, 
                        database_labels, queries_labels, 
                        database_blur_levels, queries_blur_levels,
                        top_ks=[None], batch_compute=args.mAP_batch_compute,
                        results_path=os.path.join(args.test_results_dir, q_db_mode),
                        query_image_paths=q_img_paths, database_image_paths=db_img_paths,
                        save_images=args.test_save_results_imgs,
                        save_num_imges_per_blur_level=10,
                        consider_only_db_blur_level=None)
            print('mAP computation done!')

        del model
        del ckpt
        if test_database_query_loader is not None:
            del test_database_query_loader

        del test_database_query
        del queries_descriptors
        del queries_labels
        del queries_blur_levels
        del database_descriptors
        del database_labels
        if database_blur_levels is not None:
            del database_blur_levels
        del q_img_paths
        del db_img_paths

        if args.take_distractors and args.test_synthetic_data:
            del distractors
            del distractors_loader
            del distractor_descriptors
            del distractor_blur_levels
            del distractor_img_paths
            del distractor_labels

        gc.collect()
        torch.cuda.empty_cache()
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')



    print('failed_model_paths: {}'.format(failed_model_paths))
    print('alredy_tested_model_paths: {}'.format(alredy_tested_model_paths))
    model_paths = [model_path for model_path in model_paths if model_path not in failed_model_paths and model_path not in alredy_tested_model_paths]
    print('tested_model_paths: {}'.format(model_paths))

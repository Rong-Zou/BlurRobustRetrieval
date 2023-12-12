import os
import sys
import numpy as np
import torch
import cv2
import random
import time
import argparse
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import *
from models import *
from losses import BlurRetrievalLoss
from loader import *
from metrics import *
from torch.utils.tensorboard import SummaryWriter
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

def main(args, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  -----------------------------WRITTER, LOGGER-----------------------------
    writer = SummaryWriter(log_dir=args.train_log_dir)

    train_logger = get_logger(args.train_log_dir, mode="train")
    
    print_str = 'Start training. \nUsing device: {}'.format(device)
    train_logger.info(print_str)
    print(print_str)

    train_logger.info(args)

    # -----------------------------MODEL, LOSS, OPTIMIZER, SCHEDULER-----------------------------
    image_size = [args.image_height, args.image_width]
    
    model = BlurRetrievalNet(args.num_classes if args.pred_cls else None,
                            args.num_blur_levels if args.pred_blur_level else None,
                            args.descriptor_size,
                            image_size,
                            args.pred_loc,
                            args.encoder_pretrained,
                            args.encoder_norm_type,
                            pred_blur_level_type = args.pred_blur_level_type
                            )
                     
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    params_retuire_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_str = 'The model has {} parameters ({} of them requires grad)'.format(total_params, params_retuire_grad)
    train_logger.info(print_str)
    print(print_str)
    
    loss_weights = {
        'cls_loss_weight': args.cls_loss_weight,
        'blur_estimation_loss_weight': args.blur_estimation_loss_weight,
        'contrastive_loss_weight': args.contrastive_loss_weight,
        'loc_loss_weight': args.obj_loc_loss_weight
    }
    loss_fn = BlurRetrievalLoss(loss_weights, contrastive_margin = args.contrastive_margin)

    model = model.to(device)
    loss_fn = loss_fn.to(device)
    
    optimizer_name = args.optim
    lr = args.lr
    weight_decay = args.weight_decay
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    

    if args.lr_scheduler == 'step':
        scheduler_stepLR_step_size = args.scheduler_stepLR_step_size if args.scheduler_stepLR_step_size is not None else args.num_epochs // 5
        scheduler_stepLR_gamma = args.scheduler_stepLR_gamma if args.scheduler_stepLR_gamma is not None else 0.1
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_stepLR_step_size, gamma=scheduler_stepLR_gamma)
    elif args.lr_scheduler == 'exp':
        scheduler_gamma = args.scheduler_exp_gamma if args.scheduler_exp_gamma is not None else 0.1
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler_cosine_T_max = args.scheduler_cosine_T_max if args.scheduler_cosine_T_max is not None else args.num_epochs 
        scheduler_cosine_eta_min = args.scheduler_cosine_eta_min if args.scheduler_cosine_eta_min is not None else 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_cosine_T_max, eta_min=scheduler_cosine_eta_min)
    else:
        scheduler = None

    # -----------------------------RESUME-----------------------------
    init_epoch = 0
    cur_step = 0
    best_val_mAP = 0
    ckpt = None
    resume_path = args.resume_path
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)
        init_epoch = ckpt["epoch"]
        from_step = ckpt["cur_step"]
        
        print(model)
        model.load_state_dict(ckpt["model_state"])

        best_val_mAP = ckpt["best_val_mAP"]

        # delete ckpt to save memory
        del ckpt

        print_str = 'Resume training from epoch {}'.format(init_epoch)
        train_logger.info(print_str)
        print(print_str)

    # -----------------------------DATA SPLIT, LOADER-----------------------------
    dataset_transforms = args.dataset_transforms

    data_dir = args.data_dir    
    train_folder = args.train_folder
    val_folder = args.val_folder
    test_folder = args.test_folder
    # split instances of each class into train, val, test
    # load from datadir/data_split.json if exists, otherwise create the split and save it to datadir/data_split.json

    data_split_dict = os.path.join(args.data_dir, 'stats/loader/data_split_info.json')
    if os.path.exists(data_split_dict):
        print_str = 'Loading data split info from {}'.format(data_split_dict)
        data_split_dict = json.load(open(data_split_dict, 'r'))
        cls_ids = data_split_dict['cls_ids']
        train_instance_folders = data_split_dict['train_instance_folders']
        # randomly select 2
        if args.debug:
            train_instance_folders = random.sample(train_instance_folders, 10)
        val_instance_folders = data_split_dict['val_instance_folders']
        # randomly select 2
        if args.debug:
            val_instance_folders = random.sample(val_instance_folders, 10)

        # test_instance_folders = data_split_dict['test_instance_folders']
        
        train_logger.info(print_str)
        print(print_str)
    else:
        print_str = 'Creating data split info and save it to {}'.format(os.path.join(args.train_log_dir, 'data_split_info.json'))
        cls_ids, train_instance_folders, val_instance_folders, test_instance_folders, test_database, test_query_set, val_database, val_query = split_data(data_dir, train_dir = train_folder, val_dir = val_folder, test_dir = test_folder,
                                                                                                        train_ratio=args.train_ratio, val_ratio=args.val_ratio, database_ratio=args.database_ratio,
                                                                                                        save_dir=args.train_log_dir)
        train_logger.info(print_str)
        print(print_str)

    print_str = 'Loading training set...'
    train_logger.info(print_str)
    print(print_str)
    train_set = dataset_train_val(cls_ids, train_instance_folders, 
                            num_pos = args.num_pos_per_tuple, num_negs = args.num_negs_per_tuple, 
                            normalize=True, transform=dataset_transforms,
                            logger_dir=None,
                            take_blur_levels=args.take_blur_levels,
                            take_only_sharp=args.train_take_only_sharp,
                            save_load_imgs_dir=os.path.join(args.data_dir, 'stats/loader/train'),
                            pred_blur_level_type = args.pred_blur_level_type,
                            localization_method=args.localization_method,
                            pred_cls_label_type=args.pred_cls_label_type,
                            contrastive_bl_range=args.contrastive_bl_range,
                            get_contrastive_samples = args.pred_descriptor,
                            mode = "train")

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    
    num_batches = len(train_loader)
    print_str = 'Training set loaded, it has {} batches'.format(num_batches)
    train_logger.info(print_str)
    print(print_str)

    
    if args.resume_path is not None:
        if args.resume_save_loading_time:
            if from_step == (init_epoch+1) * num_batches:
                init_epoch += 1
                cur_step = from_step
            else:
                cur_step = init_epoch * num_batches
        else:
            assert args.resume_same_data_order == True
            init_epoch = 0
            cur_step = 0

    if args.val:

        print_str = 'Loading validation set...'
        train_logger.info(print_str)
        print(print_str)

        val_set = dataset_train_val(cls_ids, val_instance_folders, 
                            num_pos = args.num_pos_per_tuple, num_negs = args.num_negs_per_tuple, 
                            normalize=True, transform=dataset_transforms,
                            logger_dir=None,
                            take_blur_levels=args.take_blur_levels,
                            take_only_sharp=args.train_take_only_sharp,
                            save_load_imgs_dir=os.path.join(args.data_dir, 'stats/loader/val'),
                            pred_blur_level_type = args.pred_blur_level_type,
                            localization_method=args.localization_method,
                            pred_cls_label_type=args.pred_cls_label_type,
                            contrastive_bl_range=args.contrastive_bl_range,
                            get_contrastive_samples = args.pred_descriptor,
                            mode = "val")
        
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)
        print_str = 'Validation set loaded, it has {} batches'.format(len(val_loader))
        train_logger.info(print_str)
        print(print_str)   


        print_str = 'Loading validation database and query set...'
        dataset_transforms = args.dataset_transforms
        val_database_query = dataset_database_query(val_instance_folders, 
                                                    normalize=True, transform=dataset_transforms, 
                                                    database_ratio=args.database_ratio, 
                                                    take_blur_levels=args.take_blur_levels,
                                                    save_load_imgs_dir=os.path.join(args.data_dir, 'stats/loader/val_dbq'))
        # create the dataloader
        val_database_query_loader = DataLoader(val_database_query, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)   
        
        val_database_query.set_dataset_type(("db", "s"))
        val_dbs_len = len(val_database_query)
        val_database_query.set_dataset_type(("db", "m"))
        val_dbm_len = len(val_database_query)

        val_database_query.set_dataset_type(("q", "s"))
        val_qs_len = len(val_database_query)
        val_database_query.set_dataset_type(("q", "m"))
        val_qm_len = len(val_database_query)
        print_str = 'Val database and query set loaded, num imgs: \n db_sharp: {}, db_mixed: {}, q_sharp: {}, q_mixed: {}'.format(val_dbs_len, val_dbm_len, val_qs_len, val_qm_len)
        train_logger.info(print_str)
        print(print_str)
        
    if args.train:
        #***************-----------------------------MAIN-----------------------------****************
        num_epochs = args.num_epochs 

        step_logging_total_loss = 0
        step_logging_loss_cls = 0
        step_logging_loss_blur_estimation = 0
        step_logging_loss_contrastive = 0
        step_logging_loss_loc = 0
        step_num_imgs = 0

        for epoch in range(init_epoch, num_epochs):
            print_str = 'Epoch {}/{}'.format(epoch+1, num_epochs)
            train_logger.info(print_str)
            print(print_str)
                        
            print_str = 'Update Epoch {}'.format(epoch+1)
            train_logger.info(print_str)
            print(print_str)

            t_start = time.time()

            total_loss_epoch = 0
            loss_cls_epoch = 0
            loss_blur_estimation_epoch = 0
            loss_contrastive_epoch = 0
            loss_loc_epoch = 0

            # batch_idx is step
            for batch_idx, (query, pos, negs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
                cur_step += 1
                if resume_path is not None:
                    if cur_step <= from_step:
                        continue
                    elif cur_step == from_step + 1:
                        print_str = 'Resume training from epoch {}, step {}'.format(epoch+1, cur_step)
                        train_logger.info(print_str)
                        print(print_str)
                        resume_path = None
                                
                query = query.to(device)

                if args.pred_cls:
                    gt_cls_idx = labels["cls_idx"].to(device)
                else:
                    gt_cls_idx = None

                if args.pred_blur_level:
                    gt_blur_level = labels["blur_level"].to(device)
                else:
                    gt_blur_level = None
                    
                if args.pred_descriptor: 
                    pos, negs = pos.to(device), negs.to(device)
                    gt_contrastive_label = labels["contrastive_label"].to(device)
                else:
                    gt_contrastive_label = None

                if args.pred_loc:
                    gt_loc = labels["bbox"].to(device)
                else:
                    gt_loc = None

                #-----------------------------TRAIN-----------------------------
                model.train()
                optimizer.zero_grad()

                total_loss_batch = 0
                loss_cls_batch = 0
                loss_blur_estimation_batch = 0
                loss_contrastive_batch = 0
                loss_loc_batch = 0

                if args.pred_descriptor:
                    for tup_idx in range(query.shape[0]): # for each tuple
                        descriptors = torch.zeros(args.descriptor_size, 1+args.num_pos_per_tuple+args.num_negs_per_tuple).to(device)
                        for i in range(1+args.num_pos_per_tuple+args.num_negs_per_tuple):
                            if i == 0:
                                blur_classes_logits, blur_estimation_logits, descriptor_query, loc_logits = model(query[tup_idx].unsqueeze(0), only_descriptor=False, cls_target=gt_cls_idx[tup_idx].unsqueeze(0) if args.pred_cls else None)         
                                descriptors[:, i] = descriptor_query
                            elif i <= args.num_pos_per_tuple:
                                descriptors[:, i] = model(pos[tup_idx, i-1].unsqueeze(0), only_descriptor=True).squeeze(0)
                            else:
                                descriptors[:, i] = model(negs[tup_idx, i-1-args.num_pos_per_tuple].unsqueeze(0), only_descriptor=True).squeeze(0)
                            
                        # make blur_estimation_logits shape: [batch_size]
                        if blur_estimation_logits is not None:
                            blur_estimation_logits = blur_estimation_logits.squeeze(1)

                        pred = (blur_classes_logits, blur_estimation_logits, descriptors, loc_logits)                        

                        # if classification, blur_estimation_logits shape is [batch_size, num_classes],
                        gt = []
                        if args.pred_cls:
                            gt.append(gt_cls_idx[tup_idx].unsqueeze(0))
                        else:
                            gt.append(None)

                        if args.pred_blur_level:
                            gt.append(gt_blur_level[tup_idx].unsqueeze(0))
                        else:
                            gt.append(None)

                        gt.append(gt_contrastive_label[tup_idx])

                        if args.pred_loc:
                            gt.append(gt_loc[tup_idx].unsqueeze(0))
                        else:
                            gt.append(None)

                        gt = tuple(gt)

                        total_loss, loss_cls, loss_blur_estimation, loss_contrastive, loss_loc = loss_fn(pred, gt)
                        
                        total_loss.backward()

                        total_loss_batch += total_loss.item()
                        loss_cls_batch += loss_cls.item()
                        loss_blur_estimation_batch += loss_blur_estimation.item()
                        loss_contrastive_batch += loss_contrastive.item()
                        loss_loc_batch += loss_loc.item()
                         
                else:
                    # when pred descriptor is False, only need to forward once
                    blur_classes_logits, blur_estimation_logits, descriptor_query, loc_logits = model(query, only_descriptor=False, cls_target=gt_cls_idx if args.pred_cls else None)         

                    if blur_estimation_logits is not None:
                        blur_estimation_logits = blur_estimation_logits.squeeze(1)
                    
                    pred = (blur_classes_logits, blur_estimation_logits, None, loc_logits)
                    gt = []
                    if args.pred_cls:
                        gt.append(gt_cls_idx)
                    else:
                        gt.append(None)

                    if args.pred_blur_level:
                        gt.append(gt_blur_level)
                    else:
                        gt.append(None)

                    gt.append(None)

                    if args.pred_loc:
                        gt.append(gt_loc)
                    else:
                        gt.append(None)

                    gt = tuple(gt)

                    total_loss, loss_cls, loss_blur_estimation, loss_contrastive, loss_loc = loss_fn(pred, gt)
                    
                    total_loss.backward()

                    # since we only forward once, we need to multiply the batch size
                    total_loss_batch += total_loss.item() * query.shape[0]
                    loss_cls_batch += loss_cls.item() * query.shape[0]
                    loss_blur_estimation_batch += loss_blur_estimation.item() * query.shape[0]
                    loss_contrastive_batch += loss_contrastive.item() * query.shape[0]
                    loss_loc_batch += loss_loc.item() * query.shape[0]
                
                total_loss_epoch += total_loss_batch
                loss_cls_epoch += loss_cls_batch
                loss_blur_estimation_epoch += loss_blur_estimation_batch
                loss_contrastive_epoch += loss_contrastive_batch
                loss_loc_epoch += loss_loc_batch

                step_logging_total_loss += total_loss_batch
                step_logging_loss_cls += loss_cls_batch
                step_logging_loss_blur_estimation += loss_blur_estimation_batch
                step_logging_loss_contrastive += loss_contrastive_batch
                step_logging_loss_loc += loss_loc_batch
                step_num_imgs += query.shape[0]
                                            
                optimizer.step()

                #----------------------------STEP-WISE LOGGING----------------------------
                log_step = args.log_step
                if cur_step % log_step == 0:
                    print_str = '[Epoch {:d}/{:d}, Log Step {:d}:{:d}:{:d}],  average loss since last step: {} ({} steps).'.format(epoch+1, num_epochs, cur_step, log_step, num_batches*num_epochs, step_logging_total_loss/step_num_imgs, log_step)
                    train_logger.info(print_str)
                    print(print_str)
                    writer.add_scalar('train_step/total_loss', step_logging_total_loss/step_num_imgs, cur_step)
                    if args.pred_cls:
                        writer.add_scalar('train_step/loss_cls', step_logging_loss_cls/step_num_imgs, cur_step)
                    if args.pred_blur_level:
                        writer.add_scalar('train_step/loss_blur_estimation', step_logging_loss_blur_estimation/step_num_imgs, cur_step)
                    if args.pred_descriptor:
                        writer.add_scalar('train_step/loss_contrastive', step_logging_loss_contrastive/step_num_imgs, cur_step)
                    if args.pred_loc:
                        writer.add_scalar('train_step/loss_loc', step_logging_loss_loc/step_num_imgs, cur_step)

                    # reset
                    step_logging_total_loss = 0
                    step_logging_loss_cls = 0
                    step_logging_loss_blur_estimation = 0
                    step_logging_loss_contrastive = 0
                    step_logging_loss_loc = 0
                    step_num_imgs = 0
                    if args.pred_loc:
                        query_ = query[0].detach().clone().cpu().numpy()
                        query_ = np.transpose(query_, (1, 2, 0))
                        query_ = np.round(query_ * 255. ).astype('uint8')
                        # get the bbox, denormalize by image size, and visualize on the query image
                        gt_loc_ = gt_loc[0].detach().clone().cpu().numpy() # [4]
                        # denormalize by image size
                        gt_loc_[0] *= query.shape[3]
                        gt_loc_[1] *= query.shape[2]
                        gt_loc_[2] *= query.shape[3]
                        gt_loc_[3] *= query.shape[2]
                        gt_loc_ = gt_loc_.astype('int')

                        # pred bbox
                        loc_logits_ = loc_logits[0].detach().clone().cpu().numpy() # [4]
                        # denormalize by image size
                        loc_logits_[0] *= query.shape[3]
                        loc_logits_[1] *= query.shape[2]
                        loc_logits_[2] *= query.shape[3]
                        loc_logits_[3] *= query.shape[2]
                        loc_logits_ = loc_logits_.astype('int')

                        # show the query image with bbox
                        # show the image as RGB, gt_loc is green, pred_loc is red
                        img = cv2.cvtColor(query_, cv2.COLOR_BGR2RGB)
                        img = cv2.rectangle(img, 
                                            (gt_loc_[0] - gt_loc_[2]//2, gt_loc_[1] - gt_loc_[3]//2),
                                            (gt_loc_[0] + gt_loc_[2]//2, gt_loc_[1] + gt_loc_[3]//2),
                                            (0, 255, 0), 2)
                        img = cv2.rectangle(img, 
                                            (loc_logits_[0] - loc_logits_[2]//2, loc_logits_[1] - loc_logits_[3]//2),
                                            (loc_logits_[0] + loc_logits_[2]//2, loc_logits_[1] + loc_logits_[3]//2),
                                            (0, 0, 255), 2)
                        writer.add_images('train_step/query_loc', img, cur_step, dataformats='HWC')

                                        
                #---------------------------------VALIDATE---------------------------------
                val_step = args.val_step
                
                if args.val and (cur_step % val_step == 0 or batch_idx == len(train_loader)-1):
                    if cur_step % val_step == 0:
                        print_str = '[Epoch {:d}/{:d}, Val Step {:d}:{:d}:{:d}], Validating...'.format(epoch+1, num_epochs, cur_step, val_step, num_batches*num_epochs)
                    if batch_idx == len(train_loader)-1:
                        print_str = '[Epoch {:d}/{:d}, Val at the end of this epoch], Validating...'.format(epoch+1, num_epochs)

                    train_logger.info(print_str)
                    print(print_str)

                    model.eval()
                    with torch.no_grad():
                        val_loss_epoch = 0
                        val_loss_cls_epoch = 0
                        val_loss_blur_estimation_epoch = 0
                        val_loss_contrastive_epoch = 0
                        val_loss_loc_epoch = 0

                        for _, (query, pos, negs, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
                            query = query.to(device)

                            if args.pred_cls:
                                gt_cls_idx = labels["cls_idx"].to(device)
                            else:
                                gt_cls_idx = None

                            if args.pred_blur_level:
                                gt_blur_level = labels["blur_level"].to(device)
                            else:
                                gt_blur_level = None
                                
                            if args.pred_descriptor:
                                pos, negs = pos.to(device), negs.to(device)
                                gt_contrastive_label = labels["contrastive_label"].to(device)
                            else:
                                gt_contrastive_label = None

                            if args.pred_loc:
                                gt_loc = labels["bbox"].to(device)
                            else:
                                gt_loc = None

                            val_total_loss_batch = 0
                            val_loss_cls_batch = 0
                            val_loss_blur_estimation_batch = 0
                            val_loss_contrastive_batch = 0
                            val_loss_loc_batch = 0

                            if args.pred_descriptor:
                                for tup_idx in range(query.shape[0]):
                                    descriptors = torch.zeros(args.descriptor_size, 1+args.num_pos_per_tuple+args.num_negs_per_tuple).to(device)
                                    for i in range(1+args.num_pos_per_tuple+args.num_negs_per_tuple):
                                        if i == 0:
                                            blur_classes_logits, blur_estimation_logits, descriptor_query, loc_logits = model(query[tup_idx].unsqueeze(0), only_descriptor=False, cls_target=gt_cls_idx[tup_idx].unsqueeze(0) if args.pred_cls else None)
                                            descriptors[:, i] = descriptor_query
                                        elif i <= args.num_pos_per_tuple:
                                            descriptors[:, i] = model(pos[tup_idx, i-1].unsqueeze(0), only_descriptor=True).squeeze(0)
                                        else:
                                            descriptors[:, i] = model(negs[tup_idx, i-1-args.num_pos_per_tuple].unsqueeze(0), only_descriptor=True).squeeze(0)
                                        
                                    # loss
                                    pred = (blur_classes_logits, blur_estimation_logits, descriptors, loc_logits) #
                                    gt = []

                                    if args.pred_cls:
                                        gt.append(gt_cls_idx[tup_idx].unsqueeze(0))
                                    else:
                                        gt.append(None)

                                    if args.pred_blur_level:
                                        gt.append(gt_blur_level[tup_idx].unsqueeze(0))
                                    else:
                                        gt.append(None)

                                    gt.append(gt_contrastive_label[tup_idx])

                                    if args.pred_loc:
                                        gt.append(gt_loc[tup_idx].unsqueeze(0))
                                    else:
                                        gt.append(None)

                                    gt = tuple(gt)

                                    val_total_loss, val_loss_cls, val_loss_blur_estimation, val_loss_contrastive, val_loss_loc = loss_fn(pred, gt)

                                    val_total_loss_batch += val_total_loss.item()
                                    val_loss_cls_batch += val_loss_cls.item()
                                    val_loss_blur_estimation_batch += val_loss_blur_estimation.item()
                                    val_loss_contrastive_batch += val_loss_contrastive.item()
                                    val_loss_loc_batch += val_loss_loc.item()
                            else:
                                # when pred descriptor is False, only need to forward once
                                blur_classes_logits, blur_estimation_logits, descriptor_query, loc_logits = model(query, only_descriptor=False, cls_target=gt_cls_idx if args.pred_cls else None)

                                if blur_estimation_logits is not None:
                                    blur_estimation_logits = blur_estimation_logits.squeeze(1)

                                pred = (blur_classes_logits, blur_estimation_logits, None, loc_logits) #
                                gt = []

                                if args.pred_cls:
                                    gt.append(gt_cls_idx)
                                else:
                                    gt.append(None)

                                if args.pred_blur_level:
                                    gt.append(gt_blur_level)
                                else:
                                    gt.append(None)

                                gt.append(None)

                                if args.pred_loc:
                                    gt.append(gt_loc)
                                else:
                                    gt.append(None)

                                gt = tuple(gt)

                                val_total_loss, val_loss_cls, val_loss_blur_estimation, val_loss_contrastive, val_loss_loc = loss_fn(pred, gt)

                                # since we only forward once, we need to multiply the batch size
                                val_total_loss_batch += val_total_loss.item() * query.shape[0]
                                val_loss_cls_batch += val_loss_cls.item() * query.shape[0]
                                val_loss_blur_estimation_batch += val_loss_blur_estimation.item() * query.shape[0]
                                val_loss_contrastive_batch += val_loss_contrastive.item() * query.shape[0]
                                val_loss_loc_batch += val_loss_loc.item() * query.shape[0]

                            val_loss_epoch += val_total_loss_batch
                            val_loss_cls_epoch += val_loss_cls_batch
                            val_loss_blur_estimation_epoch += val_loss_blur_estimation_batch
                            val_loss_contrastive_epoch += val_loss_contrastive_batch
                            val_loss_loc_epoch += val_loss_loc_batch
                        
                        val_loss_epoch = val_loss_epoch / len(val_set)
                        val_loss_cls_epoch = val_loss_cls_epoch / len(val_set)
                        val_loss_blur_estimation_epoch = val_loss_blur_estimation_epoch / len(val_set)
                        val_loss_contrastive_epoch = val_loss_contrastive_epoch / len(val_set)
                        val_loss_loc_epoch = val_loss_loc_epoch / len(val_set)
                        

                        print_str = 'Validate loss: {}'.format(val_loss_epoch)
                        train_logger.info(print_str)
                        print(print_str)

                        writer.add_scalar('val/total_loss', val_loss_epoch, cur_step)
                        if args.pred_cls:
                            writer.add_scalar('val/loss_cls', val_loss_cls_epoch, cur_step)
                        if args.pred_blur_level:
                            writer.add_scalar('val/loss_blur_estimation', val_loss_blur_estimation_epoch, cur_step)
                        if args.pred_descriptor:
                            writer.add_scalar('val/loss_contrastive', val_loss_contrastive_epoch, cur_step)
                        if args.pred_loc:
                            writer.add_scalar('val/loss_loc', val_loss_loc_epoch, cur_step)

                        if args.pred_loc:
                            query_ = query[0].detach().clone().cpu().numpy()
                            query_ = np.transpose(query_, (1, 2, 0))
                            query_ = np.round(query_ * 255. ).astype('uint8')
                            # get the bbox, denormalize by image size, and visualize on the query image
                            gt_loc_ = gt_loc[0].detach().clone().cpu().numpy() # [4]
                            # denormalize by image size
                            gt_loc_[0] *= query.shape[3]
                            gt_loc_[1] *= query.shape[2]
                            gt_loc_[2] *= query.shape[3]
                            gt_loc_[3] *= query.shape[2]
                            gt_loc_ = gt_loc_.astype('int')

                            # pred bbox
                            loc_logits_ = loc_logits[0].detach().clone().cpu().numpy() # [4]
                            # denormalize by image size
                            loc_logits_[0] *= query.shape[3]
                            loc_logits_[1] *= query.shape[2]
                            loc_logits_[2] *= query.shape[3]
                            loc_logits_[3] *= query.shape[2]
                            loc_logits_ = loc_logits_.astype('int')

                            # show the query image with bbox
                            # show the image as RGB, gt_loc is green, pred_loc is red
                            img = cv2.cvtColor(query_, cv2.COLOR_BGR2RGB)
                            img = cv2.rectangle(img, 
                                                (gt_loc_[0] - gt_loc_[2]//2, gt_loc_[1] - gt_loc_[3]//2),
                                                (gt_loc_[0] + gt_loc_[2]//2, gt_loc_[1] + gt_loc_[3]//2),
                                                (0, 255, 0), 2)
                            img = cv2.rectangle(img,
                                                (loc_logits_[0] - loc_logits_[2]//2, loc_logits_[1] - loc_logits_[3]//2),
                                                (loc_logits_[0] + loc_logits_[2]//2, loc_logits_[1] + loc_logits_[3]//2),
                                                (255, 0, 0), 2)
                            writer.add_images('val/query_loc', img, cur_step, dataformats='HWC')

                     
                    
                    # compute mAP using val_data
                    # get the descriptors and labels
                    queries_descriptors, queries_labels, queries_blur_levels, database_descriptors, database_labels, database_blur_levels, q_img_paths, db_img_paths = get_db_q_descriptors(model, val_database_query_loader, device, args.train_take_only_sharp)
                    
                    nom_db = torch.norm(database_descriptors, p=2, dim=1)
                    nom_q = torch.norm(queries_descriptors, p=2, dim=1)
                    tol = 1e-4
                    if torch.any(torch.abs(nom_db - torch.ones(database_descriptors.shape[0]).to(device)) > tol):
                        print_str = "!!! WARNING !!! Val database_descriptors not normalized with tolerance " + str(tol) + ", max diff: " + str(torch.max(torch.abs(nom_db - torch.ones(database_descriptors.shape[0]).to(device))))
                        train_logger.info(print_str)
                        print(print_str)
                        # database_descriptors = F.normalize(database_descriptors, p=2, dim=1)
                    if torch.any(torch.abs(nom_q - torch.ones(queries_descriptors.shape[0]).to(device)) > tol):
                        print_str = "!!! WARNING !!! Val queries_descriptors not normalized with tolerance " + str(tol) + ", max diff: " + str(torch.max(torch.abs(nom_q - torch.ones(queries_descriptors.shape[0]).to(device))))
                        train_logger.info(print_str)
                        print(print_str)
                        # queries_descriptors = F.normalize(queries_descriptors, p=2, dim=1)
                    
                    mAP,mAP_blur_levels = compute_mAP(database_descriptors, queries_descriptors, 
                                                    database_labels, queries_labels, 
                                                    database_blur_levels, queries_blur_levels,
                                                    top_ks=args.top_k, batch_compute=args.mAP_batch_compute,
                                                    results_path=None,
                                                    query_image_paths=q_img_paths, database_image_paths=db_img_paths,
                                                    save_images=False)

                    # writter
                    writer.add_scalar('val/mAP', mAP, cur_step)
                    print_str = 'Val mAP: {}'.format(mAP)

                    if mAP_blur_levels is not None:
                        # mAP_blur_levels: [num_blur_levels], list
                        for i_ in range(len(mAP_blur_levels)):
                            writer.add_scalar('val/mAP_blur_level_{}'.format(i_), mAP_blur_levels[i_], cur_step)
                        
                        print_str = 'Val mAP: {}, mAP_blur_levels: {}'.format(mAP, mAP_blur_levels)

                    train_logger.info(print_str)
                    print(print_str)


                    # ----------------------------SAVE MODEL----------------------------
                    # save best model according to mAP
                    if mAP > best_val_mAP:
                        best_val_mAP = mAP
                        save_path = os.path.join(args.train_log_dir, 'best_mAP_model_val.pkl')
                        torch.save({
                            'epoch': epoch,
                            'cur_step': cur_step,
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),

                            'best_val_mAP': best_val_mAP,
                            'val_mAP_blur_levels': mAP_blur_levels,
                        }, save_path)
                        print_str = 'Save the model with best val mAP {}'.format(best_val_mAP)
                        train_logger.info(print_str)
                        print(print_str)
                                       
            
            # if total_loss_epoch is not empty
            if total_loss_epoch>0:
                total_loss_epoch = total_loss_epoch / len(train_set)
                loss_cls_epoch = loss_cls_epoch / len(train_set)
                loss_blur_estimation_epoch = loss_blur_estimation_epoch / len(train_set)
                loss_contrastive_epoch = loss_contrastive_epoch / len(train_set)
                loss_loc_epoch = loss_loc_epoch / len(train_set)

                writer.add_scalar('train_epoch/total_loss', total_loss_epoch, epoch)
                if args.pred_cls:
                    writer.add_scalar('train_epoch/loss_cls', loss_cls_epoch, epoch)
                if args.pred_blur_level:
                    writer.add_scalar('train_epoch/loss_blur_estimation', loss_blur_estimation_epoch, epoch)
                if args.pred_descriptor:
                    writer.add_scalar('train_epoch/loss_contrastive', loss_contrastive_epoch, epoch)
                if args.pred_loc:
                    writer.add_scalar('train_epoch/loss_loc', loss_loc_epoch, epoch)

                t_end = time.time()      
                print_str = '[Epoch {:d}/{:d}], Average training loss of this epoch: {}, time: {:.2f}s'.format(epoch+1, num_epochs, total_loss_epoch, t_end-t_start)
            else:
                print_str = '[Epoch {:d}/{:d}] is skipped'.format(epoch+1, num_epochs)       

            train_logger.info(print_str)
            print(print_str)

            if args.save_train_model_per_epoch > 0 and (epoch+1) % args.save_train_model_per_epoch == 0:
                save_path = os.path.join(args.train_log_dir, 'train_model_epoch{}.pkl'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'cur_step': cur_step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'train_contrastive_loss': total_loss_epoch,
                    'best_val_mAP': best_val_mAP
                }, save_path)
                print_str = 'Save the model with training contrastive loss {}'.format(total_loss_epoch)
                train_logger.info(print_str)
                print(print_str)
            
            # update the learning rate scheduler if it is not None
            if scheduler is not None:
                scheduler.step()
                # log the learning rate
                writer.add_scalar('train_epoch/lr', scheduler.get_last_lr()[0], epoch)
                train_logger.info('Current Learning Rate: {}'.format(scheduler.get_last_lr()[0]))
        
        if args.save_last_model:
            save_path = os.path.join(args.train_log_dir, 'last_model.pkl')
            torch.save({
                'epoch': epoch,
                'cur_step': cur_step,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_contrastive_loss': total_loss_epoch,
            }, save_path)

            print_str = 'Save the last model with training contrastive loss {}'.format(total_loss_epoch)
            train_logger.info(print_str)
            print(print_str)
            

        train_logger.info('Training is finished.')
        print('Training is finished.')
    train_logger.handlers.clear()
    writer.close()


def get_db_q_descriptors(model, db_q_loader, device, train_take_only_sharp):
    model.to(device)
    model.eval()

    sharp_mixed = 's' if train_take_only_sharp else 'm'
    # create the database and query set descriptors
    q_img_paths=[]
    db_img_paths=[]
    with torch.no_grad():      
        db_q_loader.dataset.set_dataset_type(('q', sharp_mixed))
        for i, (q_images, q_labels, q_img_path, q_blur_level) in tqdm(enumerate(db_q_loader), total=len(db_q_loader)):
            q_images = q_images.to(device) # [B, 3, H, W]
            q_descriptors = model(q_images, only_descriptor=True) # [B, descriptor_size]
            q_labels = q_labels.to(device) # [B, num_instances_in_testset]
            q_blur_level = None if q_blur_level[0] == -1 else q_blur_level.to(device) # [B]
            q_img_paths.extend(q_img_path)
            if i == 0:
                queries_descriptors = q_descriptors
                queries_labels = q_labels
                queries_blur_levels = q_blur_level
            else:
                queries_descriptors = torch.cat((queries_descriptors, q_descriptors), 0) # [N, descriptor_size]
                queries_labels = torch.cat((queries_labels, q_labels), 0) # [N, num_instances_in_testset]
                if q_blur_level is not None:
                    queries_blur_levels = torch.cat((queries_blur_levels, q_blur_level), 0) # [N]

        db_q_loader.dataset.set_dataset_type(('db', sharp_mixed))
        for i, (db_images, db_labels, db_img_path, db_blur_level) in tqdm(enumerate(db_q_loader), total=len(db_q_loader)):
            db_images = db_images.to(device) # [B, 3, H, W]
            db_descriptors = model(db_images, only_descriptor=True) # [B, descriptor_size]
            db_labels = db_labels.to(device) # [B, num_instances_in_testset]
            db_blur_level = None if db_blur_level[0] == -1 else db_blur_level.to(device) # [B]
            db_img_paths.extend(db_img_path)
            if i == 0:
                database_descriptors = db_descriptors
                database_labels = db_labels
                database_blur_levels = db_blur_level
            else:
                database_descriptors = torch.cat((database_descriptors, db_descriptors), 0) # [N, descriptor_size]
                database_labels = torch.cat((database_labels, db_labels), 0) # [N, num_instances_in_testset]
                if db_blur_level is not None:
                    database_blur_levels = torch.cat((database_blur_levels, db_blur_level), 0) # [N]
    
    return queries_descriptors, queries_labels, queries_blur_levels, database_descriptors, database_labels, database_blur_levels, q_img_paths, db_img_paths

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -----------------------------PARSER-----------------------------
    parser = argparse.ArgumentParser(description='Blur Retrieval Training')

    # -----------------------------TRAIN DATA-----------------------------
    parser.add_argument('--data_dir', type=str, default='./overfit_test', help='path to dataset')
    
    parser.add_argument('--train_folder', type=str, default='train', help='train folder name')
    parser.add_argument('--val_folder', type=str, default='val', help='val folder name')
    parser.add_argument('--test_folder', type=str, default='test', help='test folder name')

    parser.add_argument('--train_ratio', type=float, default=0.8, help='train split')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='val split')
    parser.add_argument('--database_ratio', type=float, default=11/12, help='database in test data')

    parser.add_argument('--num_pos_per_tuple', type=int, default=1, help='number of positive samples per tuple')
    parser.add_argument('--num_negs_per_tuple', type=int, default=5, help='number of negative samples per tuple')
    parser.add_argument('--dataset_transforms', type=str, default=None, help='dataset transforms')

    parser.add_argument('--contrastive_bl_range', type=str, default='L', help='contrastive blur level range')
        
    parser.add_argument('--take_blur_levels', type=int, default=0, help='take which blur levels for the experiment')
    parser.add_argument('--train_take_only_sharp', type=bool, default=False, help='whether to take only sharp images')

    # -----------------------------OPTIMIZER-----------------------------
    parser.add_argument('--optim', type=str, default='adam', help='optimizer, adam or sgd')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='step', help='learning rate scheduler, step, exp or cosine')
    parser.add_argument('--scheduler_stepLR_step_size', type=int, default=None, help='step size for stepLR scheduler')
    parser.add_argument('--scheduler_stepLR_gamma', type=float, default=None, help='gamma for stepLR scheduler')
    parser.add_argument('--scheduler_exp_gamma', type=float, default=None, help='gamma for exp scheduler')
    parser.add_argument('--scheduler_cosine_T_max', type=int, default=None, help='T_max for cosine scheduler')
    parser.add_argument('--scheduler_cosine_eta_min', type=float, default=None, help='eta_min for cosine scheduler')

    # -----------------------------MODEL-----------------------------
    parser.add_argument('--pred_cls', type=bool, default=False, help='whether to predict classes')
    parser.add_argument('--pred_blur_level', type=bool, default=False, help='whether to predict blur levels')
    parser.add_argument('--pred_descriptor', type=bool, default=True, help='whether to predict descriptors')
    parser.add_argument('--pred_loc', type=bool, default=False, help='whether to predict object bounding box')
    parser.add_argument('--pred_blur_level_type', type=str, default=None, help='blur level type, discrete or continuous_erosion or continuous_original')
    parser.add_argument('--localization_method', type=str, default=None, help='localization method, bbox or mask')

    parser.add_argument('--descriptor_size', type=int, default=128, help='descriptor size')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--num_blur_levels', type=int, default=6, help='number of blur levels')
    parser.add_argument('--image_height', type=int, default=240, help='image height')
    parser.add_argument('--image_width', type=int, default=320, help='image width')
    
    parser.add_argument('--encoder_pretrained', type=bool, default=True, help='whether to use pretrained model')
    parser.add_argument('--encoder_norm_type', type=str, default=None, help='encoder normalization type, None, BatchNorm, InstanceNorm, LayerNorm')

    # -----------------------------LOSS-----------------------------
    parser.add_argument('--cls_loss_weight', type=float, default=1.0, help='weight for classification loss')
    parser.add_argument('--blur_estimation_loss_weight', type=float, default=1.0, help='weight for blur level estimation loss')
    parser.add_argument('--contrastive_loss_weight', type=float, default=1.0, help='weight for contrastive loss')
    parser.add_argument('--contrastive_margin', type=float, default=0.7, help='margin for contrastive loss')
    parser.add_argument('--obj_loc_loss_weight', type=float, default=10.0, help='weight for object bounding box loss')
    
    # -----------------------------TRAINING-----------------------------
    parser.add_argument('--train', type=bool, default=True, help='whether to train')

    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataloader')

    parser.add_argument('--train_batch_size', type=int, default=1, help='training batch size')

    parser.add_argument('--resume_path', type=str, default=None, help='path to resume training') #'./results/overfit_test/o/best_contrastive_loss_model.pkl'
    parser.add_argument('--resume_save_loading_time', type=bool, default=False, help='whether to save loading time when resuming training')
    parser.add_argument('--resume_same_data_order', type=bool, default=False, help='whether to use the same data order when resuming training')

    parser.add_argument('--train_log_dir', type=str, default='./results/overfit_test/four_ins', help='path to save training logs')
    parser.add_argument('--log_step', type=int, default=11, help='step size for printing log info')

    parser.add_argument('--save_train_model_per_epoch ', type=int, default=0, help='whether to save training model per epoch')
    parser.add_argument('--save_last_model', type=bool, default=False, help='whether to save the last model')
    
    parser.add_argument('--pred_cls_label_type', type=str, default=None, help='pred cls label type, cls or ins')
    parser.add_argument('--arcface_s', type=float, default=30.0, help='arcface s')
    parser.add_argument('--arcface_m', type=float, default=0.15, help='arcface m')

    # -----------------------------VALIDATION-----------------------------
    parser.add_argument('--val', type=bool, default=True, help='whether to validate during training')
    parser.add_argument('--val_step', type=int, default=33, help='step size for validation')
    parser.add_argument('--val_batch_size', type=int, default=1, help='validation batch size')
    parser.add_argument('--top_k', type=int, default=100, help='test batch size')
    parser.add_argument('--mAP_batch_compute', type=bool, default=True, help='whether to compute mAP in batch')
    
    # -----------------------------OTHERS----------------------------- 
    parser.add_argument('--debug', type=bool, default=True, help='whether to run the model with debug mode')

    # !!!!!!-----------------------------EXPERIMENTS-----------------------------!!!!!!
    args = parser.parse_args()
    
    args.train = True
    args.val = True

    args.train_ratio = 0.7
    args.val_ratio = 0.15
    args.database_ratio = 10/12

    args.mAP_batch_compute = False
    args.top_k = 500

    args.train_batch_size = 32 
    args.val_batch_size = 32

    args.log_step=100 
    args.val_step=400
    
    args.lr = 1e-4
    args.lr_scheduler = None
    args.num_epochs = 30
    args.save_train_model_per_epoch = 1
    args.save_last_model = False
    
    args.contrastive_margin = 0.7
    args.contrastive_bl_range = 'L'
    args.num_pos_per_tuple = 1
    args.num_negs_per_tuple = 5

    args.encoder_norm_type = 'NN'

    args.train_take_only_sharp = False
    args.take_blur_levels = [0,1,2,3,4,5]
        
    args.pred_descriptor=True
    args.pred_cls=True
    args.pred_blur_level=True
    args.pred_loc=True
    args.pred_blur_level_type='continuous_erosion'
    
    train_name = ''
    
    if args.pred_descriptor:
        args.descriptor_size = 128
        args.contrastive_loss_weight = 1.0
        train_name += 'des_w' + str(args.contrastive_loss_weight)
        
    if args.pred_cls:
        args.pred_cls_label_type='ins'
        args.num_classes = 792
        args.arcface_s = 30.0
        args.arcface_m = 0.15
        args.cls_loss_weight = 0.1
        train_name += 'cls_w' + str(args.cls_loss_weight)

    if args.pred_loc:
        args.localization_method = 'bbox'
        args.obj_loc_loss_weight = 10.0
        train_name += 'bbox_w' + str(args.obj_loc_loss_weight)
    else:
        args.localization_method = None
    
    if args.pred_blur_level:
        args.blur_estimation_loss_weight = 1.0
        train_name += args.pred_blur_level_type + '_w' + str(args.blur_estimation_loss_weight)
    
    args.data_dir = '/cluster/project/infk/cvg/students/rzoran/synthetic_data' 
    args.train_log_dir= './results/{}/train_results'.format(train_name)
    
    if args.debug:
        args.data_dir = '/local/home/ronzou/euler/synthetic_data'
        args.train_log_dir= args.train_log_dir.replace('./results/', './debug/')

    assure_dir(args.train_log_dir)
    
              
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Train: {}".format(train_name))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")    

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(args)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    main(args, device=device)
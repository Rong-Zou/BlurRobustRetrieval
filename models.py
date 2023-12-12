import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import random
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from model_modules import EncoderCNN, pool_and_norm, Normalize, Arcface

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


class BlurRetrievalNet(nn.Module):
    def __init__(self, num_classes=None, num_blur_levels=None, descriptor_size=128, image_size=[None,None], 
                 pred_loc=False,
                 pretrained=True, norm_type=None, pred_blur_level_type = 'discrete', arcface_s=30.0, arcface_m=0.15):
        super(BlurRetrievalNet, self).__init__()
        
        # image_size is [H, W]
        self.blur_encoder = EncoderCNN(version=2, pretrained=pretrained, norm_type=norm_type, image_size=image_size)
        
        # classifier head is pool_and_norm followed by MLP, pass the fc layer of resnet50 to MLP
        fc = models.resnet50(pretrained=pretrained).fc
        # input_dim is resnet50 last conv layer channel number
        input_dim = fc.in_features

        self.descriptor_size = descriptor_size
        self.gem_descriptor = pool_and_norm()               
        self.whitening_p1 = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU())
        in_feat_dim = input_dim
        self.num_blur_levels = num_blur_levels
        self.pred_loc = pred_loc
        
        if self.num_blur_levels is not None:
            in_feat_dim += 1024
        if self.pred_loc:
            in_feat_dim += 1024
        self.whitening_p2 = nn.Sequential(nn.Linear(in_feat_dim, self.descriptor_size), Normalize(power=2))
        
        self.num_classes = num_classes
        self.arcface_s = arcface_s
        self.arcface_m = arcface_m
        if self.num_classes is not None:
            self.classifier = Arcface(self.descriptor_size, self.num_classes, self.arcface_s, self.arcface_m)
                
        
        if self.num_blur_levels is not None:
            if pred_blur_level_type == 'discrete':
                self.blur_level_estimation_p1 = nn.Sequential(pool_and_norm(), nn.Linear(input_dim, 1024), nn.ReLU())
                self.blur_level_estimation_p2 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
                self.blur_level_estimation_p3 = nn.Sequential(nn.Linear(256, self.num_blur_levels))
            elif 'continuous' in pred_blur_level_type:
                self.blur_level_estimation_p1 = nn.Sequential(pool_and_norm(), nn.Linear(input_dim, 1024), nn.ReLU())
                self.blur_level_estimation_p2 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
                self.blur_level_estimation_p3 = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid()) # output range [0,1]
            else:
                raise NotImplementedError
                 
        
        if self.pred_loc:
            self.loc_predictor_p1 = nn.Sequential(pool_and_norm(), nn.Linear(input_dim, 1024), nn.ReLU())
            self.loc_predictor_p2 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
            self.loc_predictor_p3 = nn.Sequential(nn.Linear(256, 4), nn.Sigmoid()) # output range [0,1]
            
    
        self.image_size = image_size

        
    def forward(self, img, only_descriptor=False, cls_target=None):
        # blur encoder
        features_list = self.blur_encoder(img, return_all_feature_maps = False)

        blur_features = features_list[0]
        # print(blur_features.shape) # torch.Size([4, 2048, 20, 15])
        
        
        # blur level estimation head
        blur_estimation_logits = None
        if self.num_blur_levels is not None:
            blur_estimation_logits_1024 = self.blur_level_estimation_p1(blur_features)
            blur_estimation_logits_256 = self.blur_level_estimation_p2(blur_estimation_logits_1024)
            blur_estimation_logits = self.blur_level_estimation_p3(blur_estimation_logits_256)
        # print(blur_estimation_logits.shape) # torch.Size([4, 7])
        
        
        loc_logits = None
        if self.pred_loc:
            loc_logits_1024 = self.loc_predictor_p1(blur_features)
            loc_logits_256 = self.loc_predictor_p2(loc_logits_1024)
            loc_logits = self.loc_predictor_p3(loc_logits_256)
        
        
        # contrastive learning head
        des = self.gem_descriptor(blur_features)
        # print(descriptor.shape) # torch.Size([4, 512])
        des = self.whitening_p1(des)
        # concatenate descriptor with blur_estimation_logits_1024 and loc_logits_1024

        if self.num_blur_levels is not None:
            des = torch.cat((des, blur_estimation_logits_1024), dim=1)
        if self.pred_loc:
            des = torch.cat((des, loc_logits_1024), dim=1)

        # print(des.shape)
        descriptor = self.whitening_p2(des)
        # print(descriptor.shape)
            
        if only_descriptor:
            return descriptor
      
        
        # classifier head
        blur_classes_logits = None
        if self.num_classes is not None:
            assert cls_target is not None
            blur_classes_logits = self.classifier(descriptor, cls_target)
        # print(blur_classes_logits.shape) # torch.Size([4, 40])

        return blur_classes_logits, blur_estimation_logits, descriptor, loc_logits
    

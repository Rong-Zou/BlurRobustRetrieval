# compute different losses
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

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

class BlurRetrievalLoss(nn.Module):
    def __init__(self, loss_weights=None, contrastive_margin=0.7):
        super(BlurRetrievalLoss, self).__init__()
        self.loss_weights = loss_weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.contrastive_margin = contrastive_margin
        if self.loss_weights is None:
            self.loss_weights = self.get_default_loss_weights()
    
    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights
    
    def get_default_loss_weights(self):
        loss_weights = {}
        loss_weights['cls_loss_weight'] = 1.0
        loss_weights['blur_estimation_loss_weight'] = 1.0
        loss_weights['contrastive_loss_weight'] = 1.0
        loss_weights['loc_loss_weight'] = 1.0
        return loss_weights
    
    def forward(self, pred, gt):
        # combine all losses
        loss = torch.tensor(0.0).to(self.device)
        loss_cls = torch.tensor(0.0).to(self.device)
        loss_blur_level = torch.tensor(0.0).to(self.device)
        loss_contrastive = torch.tensor(0.0).to(self.device)
        loss_loc = torch.tensor(0.0).to(self.device)

        if self.loss_weights['cls_loss_weight'] > 0 and pred[0] is not None and gt[0] is not None:
            loss_cls = classification_loss(pred[0], gt[0]) 
            loss += self.loss_weights['cls_loss_weight'] * loss_cls
            # print('loss_cls: {:.4f}'.format(loss_cls))
        
        if self.loss_weights['blur_estimation_loss_weight'] > 0 and pred[1] is not None and gt[1] is not None:
            loss_blur_level = blur_estimation_loss(pred[1], gt[1])
            loss += self.loss_weights['blur_estimation_loss_weight'] * loss_blur_level
            # print('loss_blur_level: {:.4f}'.format(loss_blur_level))

        if self.loss_weights['contrastive_loss_weight'] > 0 and pred[2] is not None and gt[2] is not None:
            loss_contrastive = contrastive_loss(pred[2], gt[2], margin = self.contrastive_margin)
            loss += self.loss_weights['contrastive_loss_weight'] * loss_contrastive
            # print('loss_contrastive: {:.4f}'.format(loss_contrastive))
        
        if self.loss_weights['loc_loss_weight'] > 0 and pred[3] is not None and gt[3] is not None:
            loss_loc = loc_loss(pred[3], gt[3])
            loss += self.loss_weights['loc_loss_weight'] * loss_loc
            # print('loss_loc: {:.4f}'.format(loss_loc))
        
        return loss, loss_cls, loss_blur_level, loss_contrastive, loss_loc


def blur_estimation_loss(blur_level_estimation, blur_level_gt, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    blur_level_estimation = blur_level_estimation.to(device)
    blur_level_gt = blur_level_gt.to(device)
    
    loss = F.l1_loss(blur_level_estimation.squeeze(), blur_level_gt.squeeze())

    return loss


def classification_loss(cls_pred, cls_gt, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cls_pred = cls_pred.to(device)
    cls_gt = cls_gt.to(device)
    
    # compute the cross entropy loss
    loss = F.cross_entropy(cls_pred, cls_gt)
    return loss

def contrastive_loss(descriptors, labels, margin=0.7, eps=1e-9, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    descriptors = descriptors.to(device)
    labels = labels.to(device)

    # descciptors: [q1, p, n1, n2, n3, ..., nn; q2, p, n1, n2, n3, ..., nn; ...], multiple queries
    # labels: [-1, 1, 0, 0, 0, ..., 0; -1, 1, 0, 0, 0, ..., 0; ...]
    # q: query, p: positive, n: negative

    num_queries = torch.sum(labels.data==-1)
    num_imgs_per_query = descriptors.shape[1] // num_queries
    num_compares = num_imgs_per_query - 1
    queries = descriptors[:, ::num_imgs_per_query].repeat_interleave(num_compares, dim=1)
    # compares = descriptors[those elements in labels that are not -1]
    compare_indices = torch.nonzero(labels.data!=-1).squeeze()
    compares = descriptors[:, compare_indices]
    assert queries.shape == compares.shape

    labels_ = labels[labels.data != -1]
    assert queries.shape[1] == labels_.shape[0]

    # compute the euclidean distance between queries and compares
    dist = torch.pow(queries-compares + eps, 2).sum(dim=0).sqrt()

    # compute the loss
    loss = 0.5 * labels_ * torch.pow(dist, 2) + 0.5 * (1 - labels_) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    
    loss = torch.sum(loss)

    return loss

def loc_loss(pred, gt, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pred = pred.to(device)
    gt = gt.to(device)

    # compute the L1 loss
    loss = F.l1_loss(pred, gt)

    return loss
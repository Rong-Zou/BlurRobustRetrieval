import os
import sys
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from tqdm import tqdm
# from tensorboardX import SummaryWriter
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import *
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

def compute_mAP(database_descriptors, queries_descriptors, 
                database_labels, queries_labels, 
                database_blur_levels=None, queries_blur_levels=None,
                top_ks=None, batch_compute=True,
                results_path=None, query_image_paths=None, database_image_paths=None,
                save_images=True, save_num_imges_per_blur_level=10,
                consider_only_db_blur_level=None,
                ):
    """
    database_descriptors: tensor of shape (num_database_imgs, descriptor_dim)
    queries_descriptors: tensor of shape (num_queries_imgs, descriptor_dim)
    database_labels: tensor of shape (num_database_imgs, num_instances)
    queries_labels: tensor of shape (num_queries_imgs, num_instances)
    database_blur_levels: tensor of shape (num_database_imgs), ground truth blur levels of database images
    queries_blur_levels: tensor of shape (num_queries_imgs), ground truth blur levels of query images
    top_ks: int or list
    """
    device = database_descriptors.device
    num_queries = queries_descriptors.shape[0]
    num_database = database_descriptors.shape[0]     
        

    if database_blur_levels is not None and consider_only_db_blur_level is not None:
        if isinstance(consider_only_db_blur_level, int):
            # filter database, keep only blur level db_blur_level
            database_descriptors = database_descriptors[database_blur_levels == consider_only_db_blur_level, :]
            database_labels = database_labels[database_blur_levels == consider_only_db_blur_level, :]
                
            num_database = database_descriptors.shape[0]
            # also change the database_image_paths
            database_image_paths = [database_image_paths[id] for id in torch.nonzero(database_blur_levels == consider_only_db_blur_level, as_tuple=False).squeeze()]

            database_blur_levels = database_blur_levels[database_blur_levels == consider_only_db_blur_level]

        elif isinstance(consider_only_db_blur_level, list):
            # filter database, keep only blur levels in db_blur_levels
            database_descriptors = database_descriptors[torch.any(database_blur_levels == torch.tensor(consider_only_db_blur_level).to(device).unsqueeze(1), axis=0), :]
            database_labels = database_labels[torch.any(database_blur_levels == torch.tensor(consider_only_db_blur_level).to(device).unsqueeze(1), axis=0), :]
                  
            num_database = database_descriptors.shape[0]
            # also change the database_image_paths
            database_image_paths = [database_image_paths[id] for id in torch.nonzero(torch.any(database_blur_levels == torch.tensor(consider_only_db_blur_level).to(device).unsqueeze(1), axis=0), as_tuple=False).squeeze()]
            database_blur_levels = database_blur_levels[torch.any(database_blur_levels == torch.tensor(consider_only_db_blur_level).to(device).unsqueeze(1), axis=0)]


    nom_db = torch.norm(database_descriptors, p=2, dim=1)
    nom_q = torch.norm(queries_descriptors, p=2, dim=1)
    tol = 1e-5
    if torch.any(torch.abs(nom_db - torch.ones(database_descriptors.shape[0]).to(device)) > tol):
        print_str = "!!! WARNING !!! database_descriptors not normalized with tolerance " + str(tol) + ", max diff: " + str(torch.max(torch.abs(nom_db - torch.ones(database_descriptors.shape[0]).to(device))))
        print(print_str)
        # database_descriptors = F.normalize(database_descriptors, p=2, dim=1)
    if torch.any(torch.abs(nom_q - torch.ones(queries_descriptors.shape[0]).to(device)) > tol):
        print_str = "!!! WARNING !!! queries_descriptors not normalized with tolerance " + str(tol) + ", max diff: " + str(torch.max(torch.abs(nom_q - torch.ones(queries_descriptors.shape[0]).to(device))))
        print(print_str)
        # queries_descriptors = F.normalize(queries_descriptors, p=2, dim=1)

    if top_ks is None:
        top_ks = [None]
    elif isinstance(top_ks, int):
        top_ks = [top_ks]

    mAP = None
    mAP_blur_levels = None
    # top_k is a list, for each element in top_k, compute the mAP
    for top_k in top_ks:
        print("top_k: {}".format(top_k))
    
        if top_k is None:
            print("!!! WARNING !!! top_k is not specified, set it to the number of database images ({}).".format(num_database))
            top_k = num_database
        
        if top_k > num_database:
            print("!!! WARNING !!! top_k ({}) is larger than the number of database images ({}), set top_k to the number of database images".format(top_k, num_database))
            top_k = num_database
            
        if batch_compute is None:
            batch_compute = True if num_queries < 3000 else False

        if batch_compute:

            gt_matching_matrix = torch.matmul(queries_labels, database_labels.T).float()
            # gt_matching_matrix[i][j] is 1 if the i-th query image matches the j-th database image, 0 otherwise
            
            similarity_matrix = torch.matmul(queries_descriptors, database_descriptors.T)
            # similarity_matrix[i][j] is the similarity score between the i-th query image and the j-th database image

            indexes_sorted_similarity_matrix = torch.argsort(similarity_matrix, axis=1)
            # flip the matrix to sort in descending order
            indexes_sorted_similarity_matrix = torch.flip(indexes_sorted_similarity_matrix, [1])
            
            top_k_indexes_sorted_similarity_matrix = indexes_sorted_similarity_matrix[:, :top_k]
            top_k_retrieval_right_wrong_matrix = torch.gather(gt_matching_matrix, 1, top_k_indexes_sorted_similarity_matrix)
            
            # compute the average precision (AP) for each query image
            # first compute the total number of matches retrieved, for each query image
            retrieved_matches_at_k = torch.cumsum(top_k_retrieval_right_wrong_matrix, axis=1)
            retrieved_matches_each_query = torch.sum(top_k_retrieval_right_wrong_matrix, axis=1)
            # retrieved_matches_each_query[i][j] is the number of matches retrieved for the i-th query image, after j-th retrieval
            # compute the precision for each query image
            precision_at_k = retrieved_matches_at_k / torch.arange(1, top_k+1).float().to(device)
            # precision_each_query[i][j] is the precision for the i-th query image, after j-th retrieval
            # compute the average precision (AP) for each query image
            AP = torch.sum(precision_at_k * top_k_retrieval_right_wrong_matrix, axis=1) / retrieved_matches_each_query
            # replace nan values with 0
            AP = torch.nan_to_num(AP) # size (num_queries)
            # AP[i] is the average precision for the i-th query image
            # compute the mean average precision (mAP) for the whole query set
            mAP = torch.mean(AP)

            

            if database_blur_levels is not None and queries_blur_levels is not None:
                # get query_indexes_of_each_blur_level from queries_blur_levels
                # for the query set, compute the mAP for each blur level

                min_blur_level = torch.min(torch.min(database_blur_levels), torch.min(queries_blur_levels))
                max_blur_level = torch.max(torch.max(database_blur_levels), torch.max(queries_blur_levels))
                blur_levels = range(min_blur_level, max_blur_level+1)
                query_indexes_of_each_blur_level = [[] for _ in range(len(blur_levels))]
                mAP_blur_levels = []

                for i in range(len(blur_levels)):
                    query_indexes_of_each_blur_level[i] = torch.nonzero(queries_blur_levels == blur_levels[i], as_tuple=False).squeeze()
                    # mAPs for each blur level
                    mAP_blur_levels.append(torch.mean(AP[query_indexes_of_each_blur_level[i]]))
                    
                # mAP for each blur level
                mAP_blur_levels = torch.tensor(mAP_blur_levels).to(device)

                if results_path is not None:
                    save_results(results_path = results_path, 
                                save_images=save_images,
                                save_num_imges_per_blur_level=save_num_imges_per_blur_level,
                                query_image_paths=query_image_paths, 
                                database_image_paths=database_image_paths, 
                                query_blur_levels=queries_blur_levels,
                                database_blur_levels=database_blur_levels,
                                blur_levels = blur_levels,
                                top_k_indexes_sorted_similarity_matrix=top_k_indexes_sorted_similarity_matrix, 
                                top_k_retrieval_right_wrong_matrix=top_k_retrieval_right_wrong_matrix, 
                                mAP=mAP, mAP_blur_level=mAP_blur_levels, 
                                top_k=top_k,)
                
            else:
                if results_path is not None:
                    save_results(results_path = results_path, 
                                save_images=save_images,
                                save_num_imges_per_blur_level=save_num_imges_per_blur_level,
                                query_image_paths=query_image_paths, 
                                database_image_paths=database_image_paths, 
                                top_k_indexes_sorted_similarity_matrix=top_k_indexes_sorted_similarity_matrix, 
                                top_k_retrieval_right_wrong_matrix=top_k_retrieval_right_wrong_matrix, 
                                mAP=mAP, mAP_blur_level=None, top_k=top_k)
                
        else:
            AP = torch.zeros(num_queries).to(device)
            
            top_k_indexes_sorted_similarity_matrix_all = torch.zeros((num_queries, top_k)).to(device)
            top_k_retrieval_right_wrong_matrix_all = torch.zeros((num_queries, top_k)).to(device)
            
            
            if database_blur_levels is not None and queries_blur_levels is not None:
                # Do the same thing as above, but for each query image
                min_blur_level = torch.min(torch.min(database_blur_levels), torch.min(queries_blur_levels))
                max_blur_level = torch.max(torch.max(database_blur_levels), torch.max(queries_blur_levels))
                blur_levels = range(min_blur_level, max_blur_level+1)
                query_indexes_of_each_blur_level = [[] for _ in range(len(blur_levels))]
                mAP_blur_levels = []

                for q in range(num_queries):
                    gt_matching_matrix = torch.matmul(queries_labels[q, :], database_labels.T).float() # size(num_database)
                    similarity_matrix = torch.matmul(queries_descriptors[q, :], database_descriptors.T) # size(num_database)
                    indexes_sorted_similarity_matrix = torch.argsort(similarity_matrix, axis=0) # size(num_database)
                    # flip the matrix to sort in descending order
                    indexes_sorted_similarity_matrix = torch.flip(indexes_sorted_similarity_matrix, [0])
                    top_k_indexes_sorted_similarity_matrix = indexes_sorted_similarity_matrix[:top_k]
                    top_k_retrieval_right_wrong_matrix = gt_matching_matrix[top_k_indexes_sorted_similarity_matrix]

                    if save_images:
                        # similarity_matrix_all[q, :] = similarity_matrix
                        top_k_indexes_sorted_similarity_matrix_all[q, :] = top_k_indexes_sorted_similarity_matrix
                        top_k_retrieval_right_wrong_matrix_all[q, :] = top_k_retrieval_right_wrong_matrix
                    
                    retrieved_matches_at_k = torch.cumsum(top_k_retrieval_right_wrong_matrix, axis=0)
                    retrieved_matches = torch.sum(top_k_retrieval_right_wrong_matrix)

                    if retrieved_matches != 0:               
                        score = torch.linspace(1, retrieved_matches, int(retrieved_matches.item())).to(device)
                        index = (torch.nonzero(top_k_retrieval_right_wrong_matrix == 1, as_tuple=False).squeeze() + 1.0).float()
                        precision = score / index
                        AP[q] = torch.mean(precision)
                        
                for i in range(len(blur_levels)):
                    query_indexes_of_each_blur_level[i] = torch.nonzero(queries_blur_levels == blur_levels[i], as_tuple=False).squeeze()
                    
                    # mAPs for each blur level
                    mAP_blur_levels.append(torch.mean(AP[query_indexes_of_each_blur_level[i]]))
                
                mAP = torch.mean(AP)
                # mAP for each blur level
                mAP_blur_levels = torch.tensor(mAP_blur_levels).to(device)
                
                if results_path is not None:
                    save_results(results_path=results_path,
                                save_images=save_images,
                                save_num_imges_per_blur_level=save_num_imges_per_blur_level,
                                query_image_paths=query_image_paths,
                                database_image_paths=database_image_paths,
                                query_blur_levels=queries_blur_levels,
                                database_blur_levels=database_blur_levels,
                                
                                blur_levels = blur_levels,  
                                
                                top_k_indexes_sorted_similarity_matrix=top_k_indexes_sorted_similarity_matrix_all,
                                top_k_retrieval_right_wrong_matrix=top_k_retrieval_right_wrong_matrix_all,
                                mAP=mAP, mAP_blur_level=mAP_blur_levels, 
                                top_k=top_k)
                
            else:
                for q in range(num_queries):
                    gt_matching_matrix = torch.matmul(queries_labels[q, :], database_labels.T).float() # size(num_database)
                    similarity_matrix = torch.matmul(queries_descriptors[q, :], database_descriptors.T) # size(num_database)
                    indexes_sorted_similarity_matrix = torch.argsort(similarity_matrix, axis=0) # size(num_database)
                    # flip the matrix to sort in descending order
                    indexes_sorted_similarity_matrix = torch.flip(indexes_sorted_similarity_matrix, [0])
                    top_k_indexes_sorted_similarity_matrix = indexes_sorted_similarity_matrix[:top_k]
                    top_k_retrieval_right_wrong_matrix = gt_matching_matrix[top_k_indexes_sorted_similarity_matrix]

                    if save_images:
                        top_k_indexes_sorted_similarity_matrix_all[q, :] = top_k_indexes_sorted_similarity_matrix
                        top_k_retrieval_right_wrong_matrix_all[q, :] = top_k_retrieval_right_wrong_matrix
                    
                    retrieved_matches_at_k = torch.cumsum(top_k_retrieval_right_wrong_matrix, axis=0)
                    retrieved_matches = torch.sum(top_k_retrieval_right_wrong_matrix)

                    if retrieved_matches == 0:
                        continue
                    
                    score = torch.linspace(1, retrieved_matches, int(retrieved_matches.item())).to(device)
                    index = (torch.nonzero(top_k_retrieval_right_wrong_matrix == 1, as_tuple=False).squeeze() + 1.0).float()
                    precision = score / index
                    AP[i] = torch.mean(precision)
                mAP = torch.mean(AP)
                if results_path is not None:
                    save_results(results_path=results_path,
                                save_images=save_images,
                                save_num_imges_per_blur_level=save_num_imges_per_blur_level,
                                query_image_paths=query_image_paths,
                                database_image_paths=database_image_paths,
                                top_k_indexes_sorted_similarity_matrix=top_k_indexes_sorted_similarity_matrix_all,
                                top_k_retrieval_right_wrong_matrix=top_k_retrieval_right_wrong_matrix_all,
                                mAP=mAP, mAP_blur_level=None, top_k=top_k
                                )
                                
    return mAP, mAP_blur_levels

def save_results(results_path, 
                 save_images=True,
                 save_num_imges_per_blur_level=10,
                 query_image_paths=None, 
                 database_image_paths=None,
                 query_blur_levels=None,
                 database_blur_levels=None,  
                 blur_levels = None,
                 top_k_indexes_sorted_similarity_matrix = None,
                 top_k_retrieval_right_wrong_matrix = None,
                 mAP=0, 
                 mAP_blur_level=None, 
                 top_k=0,
                 retrival_per_row=10):


    if isinstance(top_k, torch.Tensor):
        top_k = top_k.cpu().numpy()
    if mAP_blur_level is not None and isinstance(mAP_blur_level, torch.Tensor):
        mAP_blur_level = mAP_blur_level.cpu().numpy()
    if isinstance(mAP, torch.Tensor):
        mAP = mAP.cpu().numpy()

    with open(os.path.join(results_path, "results{}.txt".format(top_k)), "w") as f:
        f.write("top_k: " + "\n")
        f.write(str(top_k) + "\n")
        f.write("mAP: "+ "\n")
        f.write(str(mAP) + "\n")
        if mAP_blur_level is not None:
            f.write("mAP blur level: " + "\n")
            f.write(str(mAP_blur_level) + "\n")
            
    if save_images:
        if query_blur_levels is not None and isinstance(query_blur_levels, torch.Tensor):
            query_blur_levels = query_blur_levels.cpu().numpy()
        if database_blur_levels is not None and isinstance(database_blur_levels, torch.Tensor):
            database_blur_levels = database_blur_levels.cpu().numpy()
        
        if blur_levels is not None and isinstance(blur_levels, torch.Tensor):
            blur_levels = blur_levels.cpu().numpy()

        if top_k_indexes_sorted_similarity_matrix is not None and isinstance(top_k_indexes_sorted_similarity_matrix, torch.Tensor):
            top_k_indexes_sorted_similarity_matrix = top_k_indexes_sorted_similarity_matrix.cpu().numpy().astype(int)
            # make type int
        if top_k_retrieval_right_wrong_matrix is not None and isinstance(top_k_retrieval_right_wrong_matrix, torch.Tensor):
            top_k_retrieval_right_wrong_matrix = top_k_retrieval_right_wrong_matrix.cpu().numpy()
        
        # get unique blur levels
        if mAP_blur_level is not None:
            unique_query_blur_levels = np.unique(query_blur_levels)
            # make dir for each unique blur level
            for blur_level in unique_query_blur_levels:
                if not os.path.exists(os.path.join(results_path, "blur_level_"+str(blur_level))):
                    os.makedirs(os.path.join(results_path, "blur_level_"+str(blur_level)))
        else:
            if not os.path.exists(os.path.join(results_path, "results_all_imgs")):
                os.makedirs(os.path.join(results_path, "results_all_imgs"))

        top_k_draw_im = 100 if top_k > 100 else top_k

        # get the number of images in each blur level, and the indexes of each blur level
        
        if mAP_blur_level is not None:
            each_blur_level_idxes = []
            num_imgs_each_blur_level = []
            for blur_level in unique_query_blur_levels:
                each_blur_level_idxes.append(np.where(query_blur_levels == blur_level)[0])
                num_imgs_each_blur_level.append(len(each_blur_level_idxes[-1]))

            
            selected_query_image_indexes = []
            
            # for each blur level, randomly select Q images without replacement
            for i in range(len(unique_query_blur_levels)):
                
                if num_imgs_each_blur_level[i] < save_num_imges_per_blur_level:
                    selected_query_image_indexes.extend(each_blur_level_idxes[i])
                else:
                    selected_query_image_indexes.extend(np.random.choice(each_blur_level_idxes[i], save_num_imges_per_blur_level, replace=False).tolist())          
        
        if mAP_blur_level is not None:
            idxes = selected_query_image_indexes
        else:
            idxes = range(len(query_image_paths))
            
        # for each query image, save the query image and the top k retrieved images, and the top k retrieved images for each blur level

        for i in tqdm(idxes):

            
            if mAP_blur_level is not None:
                blur_level = query_blur_levels[i]
                path_ = os.path.join(results_path, "blur_level_"+str(blur_level))
            else:
                path_ = os.path.join(results_path, "results_all_imgs")
            # save the results in the folder of this blur level

            # save the query image without showing it
            try:
                Image.open(query_image_paths[i]).save(os.path.join(path_, "query_image_"+str(i)+".png"))
            except:
                Image.open(query_image_paths[i].replace('/cluster/project/infk/cvg/students/rzoran', '/local/home/ronzou/euler')).save(os.path.join(path_, "query_image_"+str(i)+".png"))

            
            import matplotlib
            matplotlib.use('Agg',force=True)
            import matplotlib.pyplot as plt

            plt.figure(2, figsize=(20, 10))
            plt.clf()
            for j in range(top_k_draw_im):
                plt.subplot(top_k_draw_im // retrival_per_row, retrival_per_row, j+1)
                try:
                    plt.imshow(Image.open(database_image_paths[top_k_indexes_sorted_similarity_matrix[i][j]]))
                except:
                    plt.imshow(Image.open(database_image_paths[top_k_indexes_sorted_similarity_matrix[i][j]].replace('/cluster/project/infk/cvg/students/rzoran', '/local/home/ronzou/euler')))
                    

                if top_k_retrieval_right_wrong_matrix[i][j] == 1:
                    plt.title("Rank: " + str(j+1) + " BL: " + str(int(database_blur_levels[top_k_indexes_sorted_similarity_matrix[i][j]])), color='g')
                else:
                    plt.title("Rank: " + str(j+1) + " BL: " + str(int(database_blur_levels[top_k_indexes_sorted_similarity_matrix[i][j]])), color='r')

                plt.axis('off')

            plt.tight_layout()
            

            plt.savefig(os.path.join(path_, "retrieved_images_"+str(i)+".png"))
            plt.close()
            
            
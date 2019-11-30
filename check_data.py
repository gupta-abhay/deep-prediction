from data_new import Argoverse_MultiLane_Data
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import get_nt_distance,get_xy_from_nt_seq
import argparse
import numpy as np
from utils import get_xy_from_nt_seq
import glob
import pdb
import pickle
# import glob
from torch.utils.data import Dataset, DataLoader
import torch
import math
import numpy as np
from random import shuffle
import os
import pandas as pd
from data_new import collate_traj_multilane
argoverse_map=ArgoverseMap()
print("Loaded map")
dataset_train=Argoverse_MultiLane_Data('data/train/data/',avm=argoverse_map,train_seq_size=20,mode="train")
dataset_val=Argoverse_MultiLane_Data('data/val/data/',avm=argoverse_map,train_seq_size=20,mode="validate")

dataloader_train=DataLoader(dataset_train,batch_size=256,
                        shuffle=False, num_workers=8,collate_fn=collate_traj_multilane)
dataloader_val=DataLoader(dataset_val,batch_size=64,
                        shuffle=False, num_workers=8,collate_fn=collate_traj_multilane)
all_correct_seq_path=[]
# selected=0
total=0
for i,traj_dict in enumerate(dataloader_train):
    gt_traj=traj_dict['gt_traj']
    gt_unnorm_traj=traj_dict['gt_unnorm_traj'].numpy()
    all_centerlines=[ helper["ORACLE_CENTERLINE"] for helper in traj_dict['helpers']]
    pred_unnorm_traj=get_xy_from_nt_seq(gt_traj.numpy(),all_centerlines)
    norm=np.linalg.norm(pred_unnorm_traj-gt_unnorm_traj,axis=(1,2))
    index=norm<0.05
    seq_paths=traj_dict['seq_path']
    
    all_correct_seq_path.extend([seq_paths[i] for i in range(len(seq_paths)) if index[i]==True])
    total+=len(seq_paths)
    print(f"{len(all_correct_seq_path)}/{total} selected for train",end="\r")
print()
with open("train.pkl",'wb') as f:
    pickle.dump(all_correct_seq_path,f)
    

# for i,traj_dict in enumerate(dataloader_val):
#     gt_traj=traj_dict['gt_traj']
#     gt_unnorm_traj=traj_dict['gt_unnorm_traj'].numpy()
#     all_centerlines=[ helper["ORACLE_CENTERLINE"] for helper in traj_dict['helpers']]
#     pred_unnorm_traj=get_xy_from_nt_seq(gt_traj.numpy(),all_centerlines)
#     norm=np.linalg.norm(pred_unnorm_traj-gt_unnorm_traj,axis=(1,2))
#     index=norm<0.05
#     seq_paths=traj_dict['seq_path']
    
#     all_correct_seq_path.extend([seq_paths[i] for i in range(len(seq_paths)) if index[i]==True])
#     total+=len(seq_paths)
#     print(f"{len(all_correct_seq_path)}/{total} selected with small error",end="\r")
# print()
# with open("val.pkl",'wb') as f:
#     pickle.dump(all_correct_seq_path,f)



# for i,traj_dict in enumerate(dataloader_train):
#     helpers['CENTERLINE']=np.expand_dims(map_feature_helpers['ORACLE_CENTERLINE'],axis=0)
#     input_, output=get_abs_traj(input_=np.expand_dims(map_features[0:20,:],axis=0),output=np.expand_dims(map_features[20:,:],axis=0),args=args,helpers=helpers) 
#     raw_recon_data=np.vstack((input_[0],output[0]))
#     error=np.linalg.norm(raw_data-raw_recon_data)
#     errors.append(error)
#     print(f"Error iteration {i}: {error:.7f}, Max error: {max(errors):.7f}",end="\r")

#     if i%1000==0:
#         pdb.set_trace()
#     # all_centerline_train_traj=get_nt_distance(raw_data,helpers['CENTERLINE'][0])
#     # recon_prev=get_xy_from_nt_seq(nt_seq=np.expand_dims(all_centerline_train_traj,axis=0),centerlines=[helpers['CENTERLINE'][0]])
#     # recon_part_prev=get_xy_from_nt_seq(nt_seq=np.expand_dims(map_features,axis=0),centerlines=[helpers['CENTERLINE'][0]])
#     # print(f"Error iteration other way at iteration {i}: {np.linalg.norm(raw_data-recon_prev):.7f}")
#     # print(f"Error iteration other way at iteration {i}: {np.linalg.norm(raw_data-recon_part_prev):.7f}")
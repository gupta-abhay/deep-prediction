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
from data_new import collate_traj_multilane,Argoverse_Social_Centerline_Data,Argoverse_MultiLane_Data
argoverse_map=ArgoverseMap()
# print("Loaded map")
import pdb
def collate_traj_random(list_data):
    return None

# argoverse_train=Argoverse_Social_Data('data/train/data/',avm=argoverse_map,train_seq_size=20,mode="train",load_saved=False)
# for dict_i in argoverse_train:
#     pdb.set_trace()
# exit()
# dataset_train=Argoverse_Social_Centerline_Data('data/train/data/',avm=argoverse_map,train_seq_size=20,mode="train",save=False,load_saved=False)
# dataloader_train=DataLoader(dataset_train,batch_size=10,
#                         shuffle=False, num_workers=2,collate_fn=collate_traj_multilane)
# dataset_val=Argoverse_MultiLane_Data('data/val/data/',avm=argoverse_map,train_seq_size=20,mode="validate",save=True)

# for dict_i in dataset_train:
#     print("hi")
#     pdb.set_trace()
# exit()
# dataset_train=Argoverse_MultiLane_Data('data/train/data/',train_seq_size=20,mode="train",save=True,load_saved=False)
# dataloader_train=DataLoader(dataset_train,batch_size=256,
#                         shuffle=False, num_workers=16,collate_fn=collate_traj_random)
# num_batches=len(dataloader_train.batch_sampler)
# for i_batch,dict_i in enumerate(dataloader_train):
#     print(f"Done {i_batch}/{num_batches}")

dataset_val=Argoverse_MultiLane_Data('data/val/data/',avm=argoverse_map,train_seq_size=20,mode="validate",save=True)
dataloader_val=DataLoader(dataset_val,batch_size=256,
                        shuffle=False, num_workers=8,collate_fn=collate_traj_random)
num_batches=len(dataloader_val.batch_sampler)
for i_batch,dict_i in enumerate(dataloader_val):
    print(f"Done {i_batch}/{num_batches}")





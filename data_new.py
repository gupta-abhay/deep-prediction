from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.utils.centerline_utils import get_xy_from_nt_seq,get_nt_distance
# from argoverse.utils.centerline_utils import get_nt_distance,get_oracle_from_candidate_centerlines,get_xy_from_nt_seq
import glob
from torch.utils.data import Dataset, DataLoader
import torch
import math
import numpy as np
from random import shuffle
import os
import pickle
import pandas as pd
import pdb
# from shapely.geometry import LineString, Point
from argoverse.utils.centerline_utils import get_nt_distance,get_oracle_from_candidate_centerlines,get_xy_from_nt_seq
from shapely.ops import nearest_points
from argoverse.utils.map_feature_utils import MapFeaturesUtils
from argoverse.utils.social_features_utils import SocialFeaturesUtils
from typing import Any, Dict, List, Tuple
_FEATURES_SMALL_SIZE = 100
RAW_DATA_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}

def compute_features(
        seq_path: str,
        map_features_utils_instance: MapFeaturesUtils,
        social_features_utils_instance: SocialFeaturesUtils,avm,mode
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute social and map features for the sequence.

    Args:
        seq_path (str): file path for the sequence whose features are to be computed.
        map_features_utils_instance: MapFeaturesUtils instance.
        social_features_utils_instance: SocialFeaturesUtils instance.
    Returns:
        merged_features (numpy array): SEQ_LEN x NUM_FEATURES
        map_feature_helpers (dict): Dictionary containing helpers for map features

    """
    # args = parse_arguments()
    df = pd.read_csv(seq_path, dtype={"TIMESTAMP": str})

    # Get social and map features for the agent
    agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values

    # Social features are computed using only the observed trajectory
    social_features = social_features_utils_instance.compute_social_features(
        df, agent_track, 20, 50,
        RAW_DATA_FORMAT)
    # social_features=None

    # agent_track will be used to compute n-t distances for future trajectory,
    # using centerlines obtained from observed trajectory
    # print("In compute features")
    map_features, map_feature_helpers = map_features_utils_instance.compute_map_features(
        agent_track,
        20,
        50,
        RAW_DATA_FORMAT,
        mode,
        avm
    )

    # Combine social and map features

    # If track is of OBS_LEN (i.e., if it's in test mode), use agent_track of full SEQ_LEN,
    # But keep (OBS_LEN+1) to (SEQ_LEN) indexes having None values
    return social_features,map_features,map_feature_helpers
    # if agent_track.shape[0] == args.obs_len:
    #     agent_track_seq = np.full(
    #         (args.obs_len + args.pred_len, agent_track.shape[1]), None)
    #     agent_track_seq[:args.obs_len] = agent_track
    #     merged_features = np.concatenate(
    #         (agent_track_seq, social_features, map_features), axis=1)
    # else:
    #     merged_features = np.concatenate(
    #         (agent_track, social_features, map_features), axis=1)

    # return merged_features, map_feature_helpers


def collate_traj_multilane(list_data):
    dict_collate={}
    dict_input=list_data[0]
    list_data_ref=[data for data in list_data if data["norm"]<1.0]
    if len(list_data_ref)==0:
        list_data=[list_data[0]]
    else:
        list_data=list_data_ref

    for key in dict_input.keys():
        v=[]
        for data in list_data:
            v.append(data[key])
        if ((key == 'helpers') or (key == 'seq_path') or (key == 'city') or (key == 'social_features')):
            dict_collate[key]=v
        else:
            dict_collate[key]=torch.Tensor(v).float()
    return dict_collate

def collate_traj_xy(list_data):
    dict_collate={}
    dict_input=list_data[0]
    for key in dict_input.keys():
        v=[]
        for data in list_data:
            v.append(data[key])
        if ((key == 'helpers') or (key == 'seq_path') or (key == 'city') or (key == 'neighbours')):
            dict_collate[key]=v
        else:
            dict_collate[key]=torch.Tensor(v).float()
    return dict_collate

def collate_traj_social_centerline(list_data):
    dict_collate={}
    dict_input=list_data[0]
    list_data_ref=[data for data in list_data if data["norm"]<1.0]
    if len(list_data_ref)==0:
        list_data=[list_data[0]]
    else:
        list_data=list_data_ref
    for key in dict_input.keys():
        v=[]
        for data in list_data:
            v.append(data[key])
        if ((key == 'helpers') or (key == 'seq_path') or (key == 'city') or (key == 'neighbours') or (key == 'social_features')):
            dict_collate[key]=v
        else:
            dict_collate[key]=torch.Tensor(v).float()
    return dict_collate

class Argoverse_Data(Dataset):
    def __init__(self,root_dir='argoverse-data/forecasting_sample/data',train_seq_size=20):
        super(Argoverse_Data,self).__init__()
        self.root_dir=root_dir
        self.afl = ArgoverseForecastingLoader(self.root_dir)
        self.seq_paths=glob.glob(f"{self.root_dir}/*.csv")
        self.train_seq_size=train_seq_size
    def __len__(self):
        return 200
        # return len(self.seq_paths)
        # return 500
        # return len(self.seq_paths)//3


class Argoverse_MultiLane_Data(Argoverse_Data):
    def __init__(self,root_dir='argoverse-data//data',avm=None,train_seq_size=20,mode="train",save=False,load_saved=False):
        super(Argoverse_MultiLane_Data,self).__init__(root_dir,train_seq_size)
        if avm is None:
            self.avm=ArgoverseMap()
        else:
            self.avm=avm
        # if mode=="train":
        #     with open('train.pkl', 'rb') as f:
        #         self.seq_paths=pickle.load(f)
        # elif mode=="validate":
        #     with open('val.pkl', 'rb') as f:
        #         self.seq_paths=pickle.load(f)
        self.map_features_utils_instance=MapFeaturesUtils()
        self.social_features_utils_instance=SocialFeaturesUtils()
        self.mode=mode
        self.save=save
        self.load_saved=load_saved
    def compute_features_old(self,seq_path,map_instance,social_feature_instance,avm,mode="train"):
        check1=True
        if check1:
            if mode=="train" or mode=="validate":
                current_loader = self.afl.get(seq_path)
                agent_traj=current_loader.agent_traj
                # df = pd.read_csv(seq_path, dtype={"TIMESTAMP": str})
                # agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values
                candidate_centerlines = self.avm.get_candidate_centerlines_for_traj(agent_traj, current_loader.city,viz=False)
                current_centerline=get_oracle_from_candidate_centerlines(candidate_centerlines,agent_traj)
                agent_traj_norm=get_nt_distance(agent_traj,current_centerline)
                return None,agent_traj_norm,{"ORACLE_CENTERLINE":current_centerline}
            elif mode=="validate_multiple":
                current_loader = self.afl.get(seq_path)
                agent_traj=current_loader.agent_traj
                candidate_centerlines = self.avm.get_candidate_centerlines_for_traj(agent_traj, current_loader.city,viz=False)
        else:
            map_features, map_feature_helpers = self.map_features_utils_instance.compute_map_features(agent_track,20,50,RAW_DATA_FORMAT,mode,avm)
            return None,map_features,map_feature_helpers
    def __getitem__(self,index): 
        if self.mode=="train" or self.mode=="validate":
            # import pdb;pdb.set_trace()
            if self.load_saved and self.mode=="train":
                with open(f"/home/scratch/nitinsin/argoverse/train/{index}.pkl", 'rb') as f:
                    train_dict=pickle.load(f)
                return train_dict
            if self.load_saved and self.mode=="validate":
                with open(f"/home/scratch/nitinsin/argoverse/val/{index}.pkl", 'rb') as f:
                    val_dict=pickle.load(f)
                return val_dict
            current_loader = self.afl.get(self.seq_paths[index])
            agent_traj=current_loader.agent_traj
            social_features,map_features,map_feature_helpers = compute_features(
                self.seq_paths[index], self.map_features_utils_instance,self.social_features_utils_instance,self.avm,'train')
            # social_features,map_features,map_feature_helpers = self.compute_features_old(
            #     self.seq_paths[index], None,None,None,'train')
            unnorm_traj=get_xy_from_nt_seq(np.expand_dims(map_features,axis=0),[map_feature_helpers["ORACLE_CENTERLINE"]])
            norm=np.linalg.norm(unnorm_traj-agent_traj)
            # if norm>1.0:
            #     print(f"Norm at index {index}",norm)
            ref_t=map_features[self.train_seq_size-1,1]
            map_features[:,1]=map_features[:,1]-ref_t
            if self.mode=="train":
                return_dict= {'seq_path':self.seq_paths[index],'train_traj':map_features[:self.train_seq_size,:],
                        'gt_traj':map_features[self.train_seq_size:,:],'helpers':map_feature_helpers,
                        'norm':norm,'ref_t':ref_t,'social_features':social_features}
                if self.save:
                    with open(f"/home/scratch/nitinsin/argoverse/train/{index}.pkl",'wb') as f:
                        pickle.dump(return_dict,f)
            else:
                return_dict= {'seq_path':self.seq_paths[index],'train_traj':map_features[:self.train_seq_size,:],
                        'gt_unnorm_traj':agent_traj[self.train_seq_size:,:],'helpers':map_feature_helpers,
                        'norm':norm,'ref_t':ref_t,'social_features':social_features}
                if self.save:
                    with open(f"/home/scratch/nitinsin/argoverse/val/{index}.pkl",'wb') as f:
                        pickle.dump(return_dict,f)
            
            return return_dict
            # return {'seq_path':self.seq_paths[index],'train_unnorm_traj': agent_traj[:self.train_seq_size,:],
            #         'train_traj':map_features[:self.train_seq_size,:],'gt_traj':map_features[self.train_seq_size:,:],
            #         'gt_unnorm_traj':agent_traj[self.train_seq_size:,:],'helpers':map_feature_helpers,
            #         'norm_traj':map_features,'unnorm_traj':agent_traj}
        elif self.mode=="validate_multiple":
            current_loader = self.afl.get(self.seq_paths[index])
            agent_traj=current_loader.agent_traj
            social_features,map_features,map_feature_helpers = compute_features(
                self.seq_paths[index], self.map_features_utils_instance,self.social_features_utils_instance,self.avm,'test')
            return {'seq_path':self.seq_paths[index],'helpers':map_feature_helpers,'train_unnorm_traj':agent_traj[0:self.train_seq_size,:],
                    'gt_unnorm_traj':agent_traj[self.train_seq_size:,:],'city':current_loader.city,'norm':0.0}
        elif self.mode=="test":
            social_features,map_features,map_feature_helpers = compute_features(
                self.seq_paths[index], self.map_features_utils_instance,self.social_features_utils_instance,self.avm,'test')
            return {'seq_path':self.seq_paths[index],'helpers':map_feature_helpers}








class Argoverse_Social_Data(Argoverse_Data):
    def __init__(self,root_dir='argoverse-data/forecasting_sample/data',train_seq_size=20,mode="train",save=False,load_saved=False,avm=None):
        super(Argoverse_Social_Data,self).__init__(root_dir,train_seq_size)
        # self.agent_rel=agent_rel
        self.save=save
        self.mode=mode
        self.load_saved=load_saved

    def transform_social(self,agent_trajectory,neighbour_trajectories):
        def rotation_angle(x,y):
            angle=np.arctan(abs(y/x))
            direction= -1* np.sign(x*y)
            return direction*angle
        # pdb.set_trace()
        trajectory_mean=agent_trajectory[0]
        agent_trajectory=agent_trajectory-trajectory_mean
        trajectory_rotation=rotation_angle(agent_trajectory[19,0],agent_trajectory[19,1])
        c, s = np.cos(trajectory_rotation), np.sin(trajectory_rotation)
        R = np.array([[c,-s], [s, c]])
        agent_trajectory=np.transpose(agent_trajectory,[1,0])
        # agent_trajectory=torch.tensor(agent_trajectory)
        # agent_trajectory=agent_trajectory.permute(1,0)
        agent_trajectory=np.matmul(R,agent_trajectory)
        # agent_trajectory=torch.tensor(agent_trajectory)
        agent_trajectory=np.transpose(agent_trajectory,[1,0])

        normalized_neighbour_trajectories=[]
        # normalized_gt_neighbour_trajectories=[]
        for neighbour_trajectory in neighbour_trajectories:
            # trajectory=neighbour_trajectory
            neighbour_trajectory=neighbour_trajectory-trajectory_mean
            neighbour_trajectory=np.transpose(neighbour_trajectory,[1,0])
            neighbour_trajectory=np.matmul(R,neighbour_trajectory)
            neighbour_trajectory=np.transpose(neighbour_trajectory,[1,0])
            normalized_neighbour_trajectories.append(neighbour_trajectory)
        return agent_trajectory, np.array(normalized_neighbour_trajectories),trajectory_mean,R
            #if self.use_cuda:
            #    normalized_neighbour_trajectories=normalized_neighbou
        # if self.mode_test:
        #     return agent_trajectory[0:self.train_seq_size],normalized_neighbour_trajectories
        # else:
        #     return agent_trajectory[0:self.train_seq_size], agent_trajectory[self.train_seq_size:].float(),normalized_neighbour_trajectories 

    def __getitem__(self,index):
        # pdb.set_trace()
        if self.load_saved and self.mode=="train":
            with open(f"/home/scratch/nitinsin/argoverse_social_xy/train/{index}.pkl", 'rb') as f:
                train_dict=pickle.load(f)
            return train_dict
        if self.load_saved and self.mode=="validate":
            with open(f"/home/scratch/nitinsin/argoverse_social_xy/val/{index}.pkl", 'rb') as f:
                val_dict=pickle.load(f)
            return val_dict
        current_loader = self.afl.get(self.seq_paths[index])
        agent_traj=current_loader.agent_traj
        neighbours_traj=current_loader.neighbour_traj()
        agent_trajectory,normalized_neighbour_trajectories,mean,rotation=self.transform_social(agent_traj,neighbours_traj)
        # social_helper={}
        # pdb.set_trace()
        # social_helper['neighbours']=normalized_neighbour_trajectories
        if self.mode=="train":
            return_dict= {'seq_path':self.seq_paths[index],'train_traj':agent_trajectory[:self.train_seq_size,:],
                        'gt_traj':agent_trajectory[self.train_seq_size:,:],'neighbours':normalized_neighbour_trajectories,
                        'helpers':{'mean':mean, 'rotation':rotation}
                        }
            if self.save:
                with open(f"/home/scratch/nitinsin/argoverse_social_xy/train/{index}.pkl",'wb') as f:
                    pickle.dump(return_dict,f) 
        elif self.mode=="validate":
            return_dict= {'seq_path':self.seq_paths[index],'train_traj':agent_trajectory[:self.train_seq_size,:], 
                        'gt_traj':agent_trajectory[self.train_seq_size:,:],'gt_unnorm_traj':agent_traj[self.train_seq_size:,:],
                        'neighbours':normalized_neighbour_trajectories,'helpers':{'mean':mean, 'rotation':rotation}}
            # if self.save:
            #     with open(f"/home/scratch/nitinsin/argoverse_social_xy/val/{index}.pkl",'wb') as f:
            #         pickle.dump(return_dict,f)
        # return return_dict
        return return_dict



class Argoverse_Social_Centerline_Data(Argoverse_Data):
    def __init__(self,root_dir='argoverse-data/forecasting_sample/data',train_seq_size=20,mode="train",save=False,load_saved=False,avm=None):
        super(Argoverse_Social_Centerline_Data,self).__init__(root_dir,train_seq_size)
        # self.agent_rel=agent_rel
        if avm is None:
            self.avm=ArgoverseMap()
        else:
            self.avm=avm
        self.map_features_utils_instance=MapFeaturesUtils()
        self.social_features_utils_instance=SocialFeaturesUtils()
        self.save=save
        self.mode=mode
        self.load_saved=load_saved

    def convert_neighbour_centerline(self,neighbours_traj,centerline,ref_t):
        neighbour_centerline_frame=[]
        if len(neighbours_traj)==0:
            return np.array([])
        for neighbour_traj in neighbours_traj:
            temp=get_nt_distance(neighbour_traj,centerline)
            temp[:,1]=temp[:,1]-ref_t
            neighbour_centerline_frame.append(temp)
        return np.stack(neighbour_centerline_frame,axis=0)
    def __getitem__(self,index):
        # pdb.set_trace()
        if self.load_saved and self.mode=="train":
            with open(f"/home/scratch/nitinsin/argoverse_social_centerline/train/{index}.pkl", 'rb') as f:
                train_dict=pickle.load(f)
            return train_dict
        if self.load_saved and self.mode=="validate":
            with open(f"/home/scratch/nitinsin/argoverse_social_centerline/val/{index}.pkl", 'rb') as f:
                val_dict=pickle.load(f)
            return val_dict

        current_loader = self.afl.get(self.seq_paths[index])
        agent_traj=current_loader.agent_traj
        neighbours_traj=current_loader.neighbour_traj()
        social_features,map_features,map_feature_helpers = compute_features(
                self.seq_paths[index], self.map_features_utils_instance,self.social_features_utils_instance,self.avm,'train')
        
        unnorm_traj=get_xy_from_nt_seq(np.expand_dims(map_features,axis=0),[map_feature_helpers["ORACLE_CENTERLINE"]])
        norm=np.linalg.norm(unnorm_traj-agent_traj)
        ref_t=map_features[self.train_seq_size-1,1]
        map_features[:,1]=map_features[:,1]-ref_t
        neighbour_centerline_frame=self.convert_neighbour_centerline(neighbours_traj,map_feature_helpers["ORACLE_CENTERLINE"],ref_t)
        if self.mode=="train":
            return_dict= {'seq_path':self.seq_paths[index],'train_traj':map_features[:self.train_seq_size,:],
                        'gt_traj':map_features[self.train_seq_size:,:],'neighbours':neighbour_centerline_frame,
                        'helpers':map_feature_helpers, 'norm':norm,'ref_t':ref_t,'social_features':social_features
                        }   
        elif self.mode=="validate":
            return_dict= {'seq_path':self.seq_paths[index],'train_traj':map_features[:self.train_seq_size,:],
                        'gt_traj':map_features[self.train_seq_size:,:],'gt_unnorm_traj':agent_traj[self.train_seq_size:,:],
                        'neighbours':neighbour_centerline_frame,'helpers': map_feature_helpers,'norm':norm, 'ref_t':ref_t,
                        'social_features':social_features}
            if self.save:
                with open(f"/home/scratch/nitinsin/argoverse_social_centerline/val/{index}.pkl",'wb') as f:
                    pickle.dump(return_dict,f)
        return return_dict
       
    
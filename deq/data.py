from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.utils.centerline_utils import get_nt_distance,get_oracle_from_candidate_centerlines,get_xy_from_nt_seq
import glob
from torch.utils.data import Dataset, DataLoader
import torch
import math
import numpy as np
from random import shuffle
import os
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

''' Need better data representation as compared to just the trajectories.
Feed ground truth of major trajectories point.
Center lines.
'''
def collate_traj_lanecentre(list_data):
    train_agent=[]
    gt_agent=[]
    centerline=[]
    dict_collate={}
    dict_input=list_data[0]
    for key in dict_input.keys():
        v=[]
        # print("SOlving key", key)
        for data in list_data:
            # print(key,data[key].shape)
            v.append(data[key])
        if (key is 'centerline') or (key is 'city'):
            dict_collate[key]=v
        elif key is 'seq_index':
            dict_collate[key]=torch.Tensor(v)
        else:
            dict_collate[key]=torch.stack(v,dim=0)
    return dict_collate
    # return {'train_agent': torch.stack(train_agent,dim=0),'gt_agent': torch.stack(gt_agent) , 'neighbour':neighbour} 

def collate_traj_social(list_data):
    train_agent=[]
    gt_agent=[]
    neighbour=[]
    for data in list_data:
        train_agent.append(data['train_agent'])
        gt_agent.append(data['gt_agent'])
        neighbour.append(data['neighbour'])
    
    return {'train_agent': torch.stack(train_agent,dim=0),'gt_agent': torch.stack(gt_agent) , 'neighbour':neighbour} 

def collate_traj_social_test(list_data):
    seq_index=[]
    train_agent=[]
    neighbour=[]
    for data in list_data:
        train_agent.append(data['train_agent'])
        neighbour.append(data['neighbour'])
        seq_index.append(data['seq_index'])
    return {'seq_index': torch.stack(seq_index,dim=0), 'train_agent': torch.stack(train_agent,dim=0) , 'neighbour':neighbour} 
    

class Argoverse_Data(Dataset):
    def __init__(self,root_dir='argoverse-data/forecasting_sample/data',train_seq_size=20,cuda=False,test=False):
        super(Argoverse_Data,self).__init__()
        self.root_dir=root_dir
        self.afl = ArgoverseForecastingLoader(self.root_dir)
        self.seq_paths=glob.glob(f"{self.root_dir}/*.csv")
        self.train_seq_size=train_seq_size
        self.use_cuda=cuda
        self.mode_test=test

    def __len__(self):
        return len(self.seq_paths)

    def old_transform(self,trajectory):
        def rotation_angle(x,y):
            angle=np.arctan(abs(y/x))
            direction= -1* np.sign(x*y)
            return direction*angle
        translation=trajectory[0]
        trajectory=trajectory-trajectory[0]
        theta=rotation_angle(trajectory[19,0],trajectory[19,1])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c,-s], [s, c]])
        trajectory=torch.tensor(trajectory)
        trajectory=trajectory.permute(1,0)
        trajectory=np.matmul(R,trajectory)
        trajectory=torch.tensor(trajectory)
        trajectory=trajectory.permute(1,0)
        if self.mode_test:
            return trajectory[0:self.train_seq_size].float(),R,translation
        else:
            return trajectory[0:self.train_seq_size].float(),trajectory[self.train_seq_size:].float()

    def transform(self,trajectory):
        def rotation_angle(x,y):
            angle=np.arctan(abs(y/x))
            direction= -1* np.sign(x*y)
            return direction*angle
        
        if self.mode_test:
            translation=-trajectory[0]
            train_trajectory=trajectory+translation
            theta=rotation_angle(train_trajectory[19,0],train_trajectory[19,1])
            c, s = np.cos(theta), np.sin(theta)
            R = torch.Tensor([[c,-s], [s, c]]).float()
            train_trajectory=torch.tensor(train_trajectory).float()
            train_trajectory=torch.matmul(R,train_trajectory.permute(1,0)).permute(1,0)

            return train_trajectory,R,torch.Tensor(translation).float()
        else:
            old_trajectory=trajectory
            translation=-trajectory[0]
            transformed_trajectory=trajectory+translation
            theta=rotation_angle(transformed_trajectory[19,0],transformed_trajectory[19,1])
            c, s = np.cos(theta), np.sin(theta)
            R = torch.Tensor([[c,-s], [s, c]]).float()
            transformed_trajectory=torch.tensor(transformed_trajectory).float()
            transformed_trajectory=torch.matmul(R,transformed_trajectory.permute(1,0)).permute(1,0)
            train_trajectory=transformed_trajectory[:self.train_seq_size]
            gt_transformed_trajectory=transformed_trajectory[self.train_seq_size:]
            actual_gt_trajectory=torch.Tensor(trajectory[self.train_seq_size:]).float()
            return train_trajectory,gt_transformed_trajectory,actual_gt_trajectory,R,torch.Tensor(translation).float()


    def inverse_transform_one(self,trajectory,R,t):
        out=torch.matmul(R,trajectory.permute(1,0)).permute(1,0)
        return out+ t.reshape(1,2)


    def inverse_transform(self,trajectory,traj_dict):
        R=traj_dict['rotation']
        t=traj_dict['translation']
        if self.use_cuda:
            R=R.cuda()
            t=t.cuda()
        out=torch.matmul(R.permute(0,2,1),trajectory.permute(0,2,1)).permute(0,2,1)
        out= out - t.reshape(t.shape[0],1,2)
        return out


    def __getitem__(self,index):
        '''
        Obtain neighbour trajectories as well.
        Obtain map parameters at the trajectories
        Do it in the coordinates of the centerlines as well
        '''

        current_loader = self.afl.get(self.seq_paths[index])
        agent_traj=current_loader.agent_traj
        
        if self.mode_test:
            agent_train_traj,R,translation=self.transform(agent_traj)
            seq_index=int(os.path.basename(self.seq_paths[index]).split('.')[0])
            return {'seq_index': seq_index,'train_agent':agent_train_traj,'rotation':R,'translation':translation,'city':current_loader.city}
        else:
            agent_train_traj,agent_gt_traj,agent_unnorm_gt_traj,R,translation=self.transform(agent_traj)
            return {'seq_path':self.seq_paths[index],'train_agent':agent_train_traj, 'gt_agent':agent_gt_traj,'gt_unnorm_agent':agent_unnorm_gt_traj,'rotation':R,'translation':translation,'city':current_loader.city}
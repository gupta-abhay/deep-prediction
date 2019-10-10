from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import glob
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
''' Need better data representation as compared to just the trajectories.
Feed ground truth of major trajectories point.
Center lines.

'''
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
        # import pdb; pdb.set_trace()
        self.train_seq_size=train_seq_size
        self.use_cuda=cuda
        self.mode_test=test

    def __len__(self):
        return 100
        return len(self.seq_paths)
    
    def transform(self,trajectory):
        def rotation_angle(x,y):
            angle=np.arctan(abs(y/x))
            direction= -1* np.sign(x*y)
            return direction*angle
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
            return trajectory[0:self.train_seq_size].float()
        else:
            return trajectory[0:self.train_seq_size].float(),trajectory[self.train_seq_size:].float()

    def __getitem__(self,index):
        '''
        Obtain neighbour trajectories as well.
        Obtain map parameters at the trajectories
        Do it in the coordinates of the centerlines as well
        '''

        current_loader = self.afl.get(self.seq_paths[index])
        agent_traj=current_loader.agent_traj
        if self.mode_test:
            agent_train_traj=self.transform(agent_traj)
            # import pdb; pdb.set_trace()
            seq_index=int(os.path.basename(self.seq_paths[index]).split('.')[0])
            #if self.use_cuda:
            #    agent_train_traj=agent_train_traj.cuda()
            #    seq_index=seq_
            return {'seq_index': seq_index,'train_agent':agent_train_traj}
        else:
            agent_train_traj,agent_gt_traj=self.transform(agent_traj)
            #if self.use_cuda:
            #    agent_train_traj=agent_train_traj.cuda()
            #    agent_gt_traj=agent_gt_traj.cuda()
            return {'train_agent':agent_train_traj, 'gt_agent':agent_gt_traj}


class Argoverse_Social_Data(Argoverse_Data):
    def __init__(self,root_dir='argoverse-data/forecasting_sample/data',social=False,train_seq_size=20,cuda=False,test=False):
        super(Argoverse_Social_Data,self).__init__(root_dir,train_seq_size,cuda,test)

    def transform_social(self,agent_trajectory,neighbour_trajectories):
        def rotation_angle(x,y):
            angle=np.arctan(abs(y/x))
            direction= -1* np.sign(x*y)
            return direction*angle
        trajectory_mean=agent_trajectory[0]
        trajectory_rotation=rotation_angle(agent_trajectory[19,0],agent_trajectory[19,1])
        c, s = np.cos(trajectory_rotation), np.sin(trajectory_rotation)
        R = np.array([[c,-s], [s, c]])

        agent_trajectory=agent_trajectory-trajectory_mean
        agent_trajectory=torch.tensor(agent_trajectory)
        agent_trajectory=agent_trajectory.permute(1,0)
        agent_trajectory=np.matmul(R,agent_trajectory)
        agent_trajectory=torch.tensor(agent_trajectory)
        agent_trajectory=agent_trajectory.permute(1,0)
        
        normalized_neighbour_trajectories=[]
        # normalized_gt_neighbour_trajectories=[]
        for neighbour_trajectory in neighbour_trajectories:
            trajectory=neighbour_trajectory['trajectory']
            trajectory=trajectory-trajectory_mean
            trajectory=torch.tensor(trajectory)
            trajectory=trajectory.permute(1,0)
            trajectory=np.matmul(R,trajectory)
            trajectory=torch.tensor(trajectory)
            trajectory=trajectory.permute(1,0).float()
            normalized_neighbour_trajectories.append(trajectory)
        if len(normalized_neighbour_trajectories)!= 0:
            normalized_neighbour_trajectories=torch.stack(normalized_neighbour_trajectories,dim=0)
        else:
            normalized_neighbour_trajectories=torch.Tensor()
            #if self.use_cuda:
            #    normalized_neighbour_trajectories=normalized_neighbou
        if self.mode_test:
            return agent_trajectory[0:self.train_seq_size].float(),normalized_neighbour_trajectories
        else:
            return agent_trajectory[0:self.train_seq_size].float(), agent_trajectory[self.train_seq_size:].float(),normalized_neighbour_trajectories 

    def __getitem__(self,index):
        current_loader = self.afl.get(self.seq_paths[index])
        agent_traj=current_loader.agent_traj
        neighbours_traj=current_loader.neighbour_traj()
        if self.mode_test:
            agent_train_traj,neighbours_traj=self.transform_social(agent_traj,neighbours_traj)
            seq_index=int(os.path.basename(self.seq_paths[index]).split('.')[0])
            #if self.use_cuda:
            #    agent_train_traj=agent_train_traj.cuda()
            #    seq_index=seq_index.cuda()
            return {'seq_index': int(os.path.basename(self.seq_paths[index]).split('.')[0]),'train_agent':agent_train_traj, 'neighbour':neighbours_traj}
        else:
            agent_train_traj,agent_gt_traj,neighbours_traj=self.transform_social(agent_traj,neighbours_traj)
            return {'train_agent':agent_train_traj, 'gt_agent':agent_gt_traj, 'neighbour':neighbours_traj}

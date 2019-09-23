from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import glob
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
''' Need better data representation as compared to just the trajectories.
Feed ground truth of major trajectories point.
Center lines.

'''
def collate_traj(list_data):
    train_agent=[]
    gt_agent=[]
    neighbour=[]
    for data in list_data:
        train_agent.append(data['train_agent'])
        gt_agent.append(data['gt_agent'])
        neighbour.append(data['neighbour'])
    # train_agent=torch.stack(train_agent,dim=0)
    # gt_agent=torch.stack(gt_agent,dim=0)
    # return train_agent,gt_agent
    # return {'train_agent': torch.stack(train_agent,dim=0),'gt_agent': torch.stack(gt_agent,dim=0)}
    return {'train_agent': torch.stack(train_agent,dim=0),'gt_agent': torch.stack(gt_agent) , 'neighbour':neighbour} 
class Argoverse_Data(Dataset):
    def __init__(self,root_dir='argoverse-data/forecasting_sample/data',social=False,train_seq_size=20):
        super(Argoverse_Data,self).__init__()
        self.root_dir=root_dir
        self.afl = ArgoverseForecastingLoader(self.root_dir)
        
        self.seq_paths=glob.glob(f"{self.root_dir}/*.csv")
        self.social=social
        self.train_seq_size=train_seq_size
    

    def __len__(self):
        return len(self.seq_paths)
    

    def transform(self,trajectory):
        def rotation_angle(x,y):
            angle=np.arctan(abs(y/x))
            direction= -1* np.sign(x*y)
            return direction*angle
        trajectory=trajectory-trajectory[0]
        theta=rotation_angle(trajectory[20,0],trajectory[20,1])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c,-s], [s, c]])
        trajectory=torch.tensor(trajectory)
        trajectory=trajectory.permute(1,0)
        trajectory=np.matmul(R,trajectory)
        trajectory=torch.tensor(trajectory)
        trajectory=trajectory.permute(1,0)
        return trajectory[0:self.train_seq_size].float(),trajectory[self.train_seq_size:].float()

    
    def transform_social(self,agent_trajectory,neighbour_trajectories):
        def rotation_angle(x,y):
            angle=np.arctan(abs(y/x))
            direction= -1* np.sign(x*y)
            return direction*angle
        trajectory_mean=agent_trajectory[0]
        trajectory_rotation=rotation_angle(agent_trajectory[20,0],agent_trajectory[20,1])
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
            # if start_index<15:
            #     continue
            # print("The shape of neighbour trajectory is ",trajectory.shape)
            # print(f"Start index: {start_index}. End index: {end_index}")
            trajectory=trajectory-trajectory_mean
            trajectory=torch.tensor(trajectory)
            trajectory=trajectory.permute(1,0)
            trajectory=np.matmul(R,trajectory)
            trajectory=torch.tensor(trajectory)
            trajectory=trajectory.permute(1,0)
            normalized_neighbour_trajectories.append(trajectory)
        return agent_trajectory[0:self.train_seq_size], agent_trajectory[self.train_seq_size:],torch.stack(normalized_neighbour_trajectories,dim=0) 
        

    def __getitem__(self,index):
        '''
        Obtain neighbour trajectories as well.
        Obtain map parameters at the trajectories
        Do it in the coordinates of the centerlines as well
        '''

        current_loader = self.afl.get(self.seq_paths[index])
        agent_traj=current_loader.agent_traj
        if self.social:
            neighbours_traj=current_loader.neighbour_traj()
            agent_train_traj,agent_gt_traj,neighbours_traj=self.transform_social(agent_traj,neighbours_traj)
            return {'train_agent':agent_train_traj, 'gt_agent':agent_gt_traj, 'neighbour':neighbours_traj}
        else:
            agent_train_traj,agent_gt_traj=self.transform(agent_traj)
            return {'train_agent':agent_train_traj, 'gt_agent':agent_gt_traj}

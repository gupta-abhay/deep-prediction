import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from utils import get_xy_from_nt_seq
"""All models should predict 5 trajectories. We will use 1 from R,t prediction to evaluate.
If less repeat till you get 6.""" 

class SinglePrediction(nn.Module):
    """Predict single trajectory along each candidate centerlines"""
    def __init__(self):
        self.encoder_lstm=nn.LSTMCell(input_size=2,hidden_size=64)
        self.embedding_pos=nn.Linear(64,2)
        self.decoder_lstm=nn.LSTMCell(input_size=64,hidden_size=64)

    def forward(self,input_dict,mode="train"):
        if mode=="train" or mode=="validate":
            input_traj=input_dict['train_traj']
            self.h,self.c=(torch.zeros(input_traj.shape[0],64),torch.zeros(input_traj.shape[0],64))
            self.h=self.h.cuda()
            self.c=self.c.cuda()
            for i in range(20):
                self.h,self.c=self.encoder_lstm(input_traj[:,i,:],(self.h,self.c))
            for i in range(30):
                self.h,self.c=self.decoder_lstm(self.h,(self.h,self.c))
                out.append(self.embedding_pos(self.h))
            pred_traj=torch.stack(out,dim=1)
            if mode=="validate":
                all_centerlines=[helper_dict["ORACLE_CENTERLINE"] for helper_dict in input_dict['helpers']]
                pred_unnorm_traj=get_xy_from_nt_seq(pred_traj,all_centerlines)
                return pred_unnorm_traj
            else:
                return pred_traj
        elif mode=="validate_multiple" or mode=="test":
            # input_traj=input_dict['train_traj']
           
            input_traj=[]
            for helper_dict in input_dict["helpers"]:
                input_traj.extend(helper_dict['CANDIDATE_NT_DISTANCES'])
            input_traj=torch.Tensor(input_traj).float().cuda()
            self.h,self.c=(torch.zeros(input_traj.shape[0],64),torch.zeros(input_traj.shape[0],64))
            self.h=self.h.cuda()
            self.c=self.c.cuda()
            for i in range(20):
                self.h,self.c=self.encoder_lstm(input_traj[:,i,:],(self.h,self.c))
            for i in range(30):
                self.h,self.c=self.decoder_lstm(self.h,(self.h,self.c))
                out.append(self.embedding_pos(self.h))
            pred_traj=torch.stack(out,dim=1)
            i=0
            all_pred_unnorm_traj=[]
            for helper_dict in input_dict["helpers"]:
                pred_unnorm_traj=get_xy_from_nt_seq(pred_traj[i:i+len(helper_dict["CANDIDATE_NT_DISTANCES"])],helper_dict["CANDIDATE_CENTERLINES"])
                all_pred_unnorm_traj.append(pred_unnorm_traj)
            return all_pred_unnorm_traj

        else:
            print(f"Wrong mode {mode}. What are you doing")


class MultiPrediction(nn.Module):
    pass


class SinglePredictionImage(nn.Module):
    pass

class MultiPredictionImage(nn.Module):
    pass


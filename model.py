import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from TCN.tcn import TemporalConvNetwork, TimeDistributedLayer
# from TCN.trellisnet import TrellisNet
# from TCN.utils import WeightDrop

class LaneMultiModel(nn.Module):
    def __init__(self,cuda=False):
        super(LSTMModel,self).__init__()
        self.encoder_lstm=nn.LSTMCell(input_size=2,hidden_size=64)
        self.embedding_pos=nn.Linear(64,2)
        self.decoder_lstm=nn.LSTMCell(input_size=64,hidden_size=64)
        self.use_cuda=cuda

    def forward(self,input_dict):
        # import pdb; pdb.set_trace()
        list_lists=input_dict['train_agent'] ## this will be a list conver to batch and reshape

        input_traj=torch.Tensor([traj for one_list in list_lists for traj in one_list])

        # if len(input_dict[''])
        self.h,self.c=(torch.zeros(input_traj.shape[0],64),torch.zeros(input_traj.shape[0],64))
        if self.use_cuda:
            input_traj=input_traj.cuda()
            self.h=self.h.cuda()
            self.c=self.c.cuda()
        for i in range(20):
            self.h,self.c=self.encoder_lstm(input_traj[:,i,:],(self.h,self.c))
        out=[]
        for i in range(30):
            self.h,self.c=self.decoder_lstm(self.h,(self.h,self.c))
            out.append(self.embedding_pos(self.h))

        pred_traj=torch.stack(out,dim=1)
        pred_list_traj=[]
        for one_list in list_lists:
            pred_list_traj.append(pred_traj[i:i+len(one_list)])
            i=i+len(one_list)
        if 'gt_unnorm_agent' in input_traj.keys():
            pass
        else:
            return pred_list_traj

class LSTMModel(nn.Module):
    def __init__(self,cuda=False):
        super(LSTMModel,self).__init__()
        self.encoder_lstm=nn.LSTMCell(input_size=2,hidden_size=64)
        self.embedding_pos=nn.Linear(64,2)
        self.decoder_lstm=nn.LSTMCell(input_size=64,hidden_size=64)
        self.use_cuda=cuda

    def forward(self,input_dict):
        # import pdb; pdb.set_trace()
        input_traj=input_dict['train_agent']
        self.h,self.c=(torch.zeros(input_traj.shape[0],64),torch.zeros(input_traj.shape[0],64))
        if self.use_cuda:
            input_traj=input_traj.cuda()
            self.h=self.h.cuda()
            self.c=self.c.cuda()
        for i in range(20):
            self.h,self.c=self.encoder_lstm(input_traj[:,i,:],(self.h,self.c))
        out=[]
        for i in range(30):
            self.h,self.c=self.decoder_lstm(self.h,(self.h,self.c))
            out.append(self.embedding_pos(self.h))
        pred_traj=torch.stack(out,dim=1)
        return pred_traj
    # def inverse_transform(self,pred_traj,inv_R,inv_t):
    #     shape_tensor=pred_traj.shape
    #     out1=torch.matmul(inv_R,pred_traj.reshape(-1,2).transpose(1,0)).transpose(1,0).reshape(shape_tensor[0],shape_tensor[1],shape_tensor[2])
    #     out2= out1 + inv_t.reshape(1,1,2)
    #     return out2
        # torch.matmul(R,pred_traj.reshape(-1,2).transpose(1,0)).transpose(1,0).reshape(32,30,-1).shape
        # pass
        # pred_traj=np.matmul(inv_R,pred


# TCN Model
class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, embedding_size=64, use_cuda=True):
        super(TCNModel, self).__init__()
        self.input_embedding = nn.Linear(2, embedding_size)
        self.tcn = TemporalConvNetwork(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.output_embedding = nn.Linear(embedding_size, 2)
        self.tdst_output = TimeDistributedLayer(nn.Linear(20, 30), batch_first=True)
        self.use_cuda = use_cuda

    
    def forward(self, input_dict):
        input_traj = input_dict['train_agent']
        if self.use_cuda:
            input_traj = input_traj.cuda()

        x = self.input_embedding(input_traj)
        x = self.tcn(x)
        x = self.tdst_output(x)
        x = x.permute(0,2,1)
        out = self.output_embedding(x)
        return out

class Social_Model(nn.Module):
    def __init__(self,cuda=False):
        super(Social_Model,self).__init__()
        self.agent_encoder=nn.LSTM(input_size=2,hidden_size=64,batch_first=True)
        self.neighbour_encoder=nn.LSTM(input_size=2,hidden_size=64,batch_first=True)
        self.decoder_lstm=nn.LSTMCell(input_size=128,hidden_size=128)
        self.embedding_pos=nn.Linear(128,2)
        self.use_cuda=cuda
    def forward(self,input_dict):
        # import pdb; pdb.set_trace()
        agent_traj=input_dict['train_agent']
        neighbour_traj=input_dict['neighbour']
        if self.use_cuda:
            agent_traj=agent_traj.cuda()
        agent_embedding,_=self.agent_encoder(agent_traj)
        agent_embedding=agent_embedding[:,-1,:]
        # import pdb; pdb.set_trace()
        neighbour_embedding=[]
        pred_traj=[]
        for batch_index in range(len(neighbour_traj)):
            curr_neighbours_traj=neighbour_traj[batch_index]
            if self.use_cuda:
                curr_neighbours_traj=curr_neighbours_traj.cuda()
            if curr_neighbours_traj.shape[0]!=0:
                out=self.neighbour_encoder(curr_neighbours_traj)[0][:,-1,:]
                out,_=torch.max(out,dim=0)
                neighbour_embedding.append(out)
                # import pdb; pdb.set_trace()
            else:
                if self.use_cuda:
                    out=torch.zeros(64).cuda()
                else:
                    out=torch.zeros(64)
                neighbour_embedding.append(out)
        neighbour_embedding=torch.stack(neighbour_embedding,dim=0)
        self.h=torch.cat([agent_embedding,neighbour_embedding],dim=1)
        if self.use_cuda:
            self.c=torch.zeros(self.h.shape).cuda()
        else:
            self.c=torch.zeros(self.h.shape)
        # import pdb; pdb.set_trace()
        for _ in range(30):
            self.h,self.c=self.decoder_lstm(self.h,(self.h,self.c))
            pred_traj.append(self.embedding_pos(self.h))
        pred_traj=torch.stack(pred_traj,dim=1)
        return pred_traj

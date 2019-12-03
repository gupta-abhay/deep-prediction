import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from utils import get_xy_from_nt_seq
import pdb
import numpy as np
from shapely.geometry import LineString
"""All models should predict 5 trajectories. We will use 1 from R,t prediction to evaluate.
If less repeat till you get 6.""" 

class LSTMModel(nn.Module):
    """Predict single trajectory along each candidate centerlines"""
    def __init__(self):
        super(LSTMModel,self).__init__()
        self.encoder_lstm=nn.LSTMCell(input_size=2,hidden_size=64)
        self.embedding_pos=nn.Linear(64,2)
        self.decoder_lstm=nn.LSTMCell(input_size=64,hidden_size=64)

    def forward(self,input_dict,mode="train"):
        if mode=="train" or mode=="validate":
            input_traj=input_dict['train_traj']
            # gt_traj=input_dict['gt_traj']
            # all_centerlines=[helper_dict["ORACLE_CENTERLINE"] for helper_dict in input_dict['helpers']]
            # pred_unnorm_traj=get_xy_from_nt_seq(gt_traj,all_centerlines)
            # print("Reconstruction error of gt in this batch",np.linalg.norm(pred_unnorm_traj-input_dict['gt_unnorm_traj'].numpy()))
            # pred_unnorm_traj=get_xy_from_nt_seq(input_traj,all_centerlines)
            # print("Reconstruction error of train in this batch",np.linalg.norm(pred_unnorm_traj-input_dict['train_unnorm_traj'].numpy()))
            input_traj=input_traj.cuda()
            self.h,self.c=(torch.zeros(input_traj.shape[0],64),torch.zeros(input_traj.shape[0],64))

            self.h=self.h.cuda()
            self.c=self.c.cuda()
            out=[]
            # pdb.set_trace()
            # x_ref=input_traj[:,19,1]
            # input_traj[:,:,1]=input_traj[:,:,1]-x_ref.unsqueeze(1)
            for i in range(20):
                self.h,self.c=self.encoder_lstm(input_traj[:,i,:],(self.h,self.c))
            for i in range(30):
                self.h,self.c=self.decoder_lstm(self.h,(self.h,self.c))
                out.append(self.embedding_pos(self.h))
            pred_traj=torch.stack(out,dim=1)
            # pred_traj[:,:,1]=pred_traj[:,:,1]+x_ref.unsqueeze(1)
            if mode=="train":
                return pred_traj
            elif mode=="validate":
                # pred_traj=pred_traj.detach().cpu().numpy()
                pred_traj[:,:,1]=pred_traj[:,:,1] + input_dict['ref_t'].unsqueeze(1).cuda()
                # return pred_traj
                all_centerlines=[helper_dict["ORACLE_CENTERLINE"] for helper_dict in input_dict['helpers']]
                pred_unnorm_traj=get_xy_from_nt_seq(pred_traj.detach().cpu().numpy(),all_centerlines)
                pred_unnorm_traj=torch.Tensor(pred_unnorm_traj).float().cuda()
                return pred_unnorm_traj
            else:
                print("what are you doing here")
        elif mode=="validate_multiple" or mode=="test":
            input_traj=[]
            # pdb.set_trace()
            for helper_dict in input_dict["helpers"]:
                input_traj.extend([traj[0:20,:] for traj in helper_dict["CANDIDATE_NT_DISTANCES"]])
            input_traj=torch.Tensor(input_traj).float()
            ref_t=input_traj[:,19,1].clone()
            input_traj[:,:,1]=input_traj[:,:,1] - ref_t.unsqueeze(1)
            input_traj=input_traj.cuda()
            self.h,self.c=(torch.zeros(input_traj.shape[0],64),torch.zeros(input_traj.shape[0],64))
            self.h=self.h.cuda()
            self.c=self.c.cuda()
            for i in range(20):
                self.h,self.c=self.encoder_lstm(input_traj[:,i,:],(self.h,self.c))
            out=[]
            for i in range(30):
                self.h,self.c=self.decoder_lstm(self.h,(self.h,self.c))
                out.append(self.embedding_pos(self.h))
            # pdb.set_trace()
            pred_traj=torch.stack(out,dim=1)
            pred_traj=pred_traj.detach().cpu()
            pred_traj[:,:,1] = pred_traj[:,:,1]+ ref_t.unsqueeze(1)
            pred_traj=pred_traj.numpy()
            
            index_curr_pred=0

            all_pred_unnorm_traj=[]
            for i_pred,helper_dict in enumerate(input_dict["helpers"]):
                size_curr_pred=len(helper_dict["CANDIDATE_NT_DISTANCES"])
                partial_pred_traj=pred_traj[index_curr_pred:index_curr_pred+size_curr_pred,:,:]
                pred_unnorm_traj=get_xy_from_nt_seq(partial_pred_traj,helper_dict["CANDIDATE_CENTERLINES"])
                all_pred_unnorm_traj.append(pred_unnorm_traj)
                index_curr_pred+=size_curr_pred
            return all_pred_unnorm_traj

        else:
            print(f"Wrong mode {mode}. What are you doing")

class LSTMModel_CenterlineEmbed(nn.Module):
    """Predict single trajectory along each candidate centerlines"""
    def __init__(self):
        super(LSTMModel_CenterlineEmbed,self).__init__()
        self.encoder_lstm=nn.LSTMCell(input_size=2,hidden_size=64)
        self.embedding_pos=nn.Linear(96,2)
        self.decoder_lstm=nn.LSTMCell(input_size=96,hidden_size=96)
        self.centerline_embed=nn.LSTMCell(input_size=2,hidden_size=32)
        self.use_centerline_rotation=True
    def forward(self,input_dict,mode="train"):
        if mode=="train" or mode=="validate":
            input_traj=input_dict['train_traj']
            # pdb.set_trace()
            # gt_traj=input_dict['gt_traj']
            # all_centerlines=[helper_dict["ORACLE_CENTERLINE"] for helper_dict in input_dict['helpers']]
            # pred_unnorm_traj=get_xy_from_nt_seq(gt_traj,all_centerlines)
            # print("Reconstruction error of gt in this batch",np.linalg.norm(pred_unnorm_traj-input_dict['gt_unnorm_traj'].numpy()))
            # pred_unnorm_traj=get_xy_from_nt_seq(input_traj,all_centerlines)
            # print("Reconstruction error of train in this batch",np.linalg.norm(pred_unnorm_traj-input_dict['train_unnorm_traj'].numpy()))
            len_list=[]
            centerline_norm_list=[]
            for index,helper_dict in enumerate(input_dict['helpers']):
                centerline=helper_dict['ORACLE_CENTERLINE']
                ls=LineString(centerline)
                start_coords=np.array(ls.interpolate(input_dict['ref_t'][index].numpy()).coords[:])
                centerline_norm=np.linalg.norm(centerline-start_coords,axis=1)
                min_index=np.argmin(centerline_norm)
                centerline_norm=(centerline-start_coords)[min_index:,:]
                # if self.use_centerline_rotation:
                #     def rotation_angle(x,y):

                #     start_pred_coord=np.array(ls.interpolate(input_dict['ref_t'][index].numpy()+input_traj[20,:].numpy()).coords[:])
                #     centerline_pred_norm=np.linalg.norm(centerline-start_pred_coord,axis=1)
                #     min_index=np.argmin(centerline_pred_norm)
                    
                centerline_norm_list.append(centerline_norm)
                len_list.append(centerline_norm.shape[0])

            # pdb.set_trace()
            max_len=max(len_list)
            all_centerlines_array=np.zeros((len(len_list),max_len,2))
            for index,centerline_array in enumerate(centerline_norm_list):
                all_centerlines_array[index,-len_list[index]:,:]=centerline_array
            all_centerlines_tensor=torch.Tensor(all_centerlines_array).cuda()
            # pdb.set_trace()
            self.cent_h,self.cent_c=(torch.zeros(all_centerlines_tensor.shape[0],32).cuda(),torch.zeros(all_centerlines_tensor.shape[0],32).cuda())
            for i in range(max_len):
                self.cent_h,self.cent_c=self.centerline_embed(all_centerlines_tensor[:,i,:],(self.cent_h,self.cent_c))
        
            # pdb.set_trace()
            input_traj=input_traj.cuda()
            self.h,self.c=(torch.zeros(input_traj.shape[0],64),torch.zeros(input_traj.shape[0],64))

            self.h=self.h.cuda()
            self.c=self.c.cuda()
            out=[]
            for i in range(20):
                self.h,self.c=self.encoder_lstm(input_traj[:,i,:],(self.h,self.c))
            # pdb.set_trace()
            self.h=torch.cat((self.h,self.cent_h),dim=1)
            self.c=torch.cat((self.c,self.cent_c),dim=1)
            for i in range(30):
                self.h,self.c=self.decoder_lstm(self.h,(self.h,self.c))
                out.append(self.embedding_pos(self.h))
            pred_traj=torch.stack(out,dim=1)
            # pdb.set_trace()
            # pred_traj[:,:,1]=pred_traj[:,:,1]+x_ref.unsqueeze(1)
            if mode=="train":
                return pred_traj
            elif mode=="validate":
                # pred_traj=pred_traj.detach().cpu().numpy()
                pred_traj[:,:,1]=pred_traj[:,:,1] + input_dict['ref_t'].unsqueeze(1).cuda()
                # return pred_traj
                all_centerlines=[helper_dict["ORACLE_CENTERLINE"] for helper_dict in input_dict['helpers']]
                pred_unnorm_traj=get_xy_from_nt_seq(pred_traj.detach().cpu().numpy(),all_centerlines)
                pred_unnorm_traj=torch.Tensor(pred_unnorm_traj).float().cuda()
                return pred_unnorm_traj
            else:
                print("what are you doing here")
        elif mode=="validate_multiple" or mode=="test":
            input_traj=[]
            # pdb.set_trace()
            for helper_dict in input_dict["helpers"]:
                input_traj.extend([traj[0:20,:] for traj in helper_dict["CANDIDATE_NT_DISTANCES"]])
            input_traj=torch.Tensor(input_traj).float()
            ref_t=input_traj[:,19,1].clone()
            input_traj[:,:,1]=input_traj[:,:,1] - ref_t.unsqueeze(1)
            input_traj=input_traj.cuda()
            self.h,self.c=(torch.zeros(input_traj.shape[0],64),torch.zeros(input_traj.shape[0],64))
            self.h=self.h.cuda()
            self.c=self.c.cuda()
            for i in range(20):
                self.h,self.c=self.encoder_lstm(input_traj[:,i,:],(self.h,self.c))
            out=[]
            for i in range(30):
                self.h,self.c=self.decoder_lstm(self.h,(self.h,self.c))
                out.append(self.embedding_pos(self.h))
            # pdb.set_trace()
            pred_traj=torch.stack(out,dim=1)
            pred_traj=pred_traj.detach().cpu()
            pred_traj[:,:,1] = pred_traj[:,:,1]+ ref_t.unsqueeze(1)
            pred_traj=pred_traj.numpy()
            
            index_curr_pred=0

            all_pred_unnorm_traj=[]
            for i_pred,helper_dict in enumerate(input_dict["helpers"]):
                size_curr_pred=len(helper_dict["CANDIDATE_NT_DISTANCES"])
                partial_pred_traj=pred_traj[index_curr_pred:index_curr_pred+size_curr_pred,:,:]
                pred_unnorm_traj=get_xy_from_nt_seq(partial_pred_traj,helper_dict["CANDIDATE_CENTERLINES"])
                all_pred_unnorm_traj.append(pred_unnorm_traj)
                index_curr_pred+=size_curr_pred
            return all_pred_unnorm_traj

        else:
            print(f"Wrong mode {mode}. What are you doing")
class Social_Model(nn.Module):
    def __init__(self,cuda=False):
        super(Social_Model,self).__init__()
        self.agent_encoder=nn.LSTM(input_size=2,hidden_size=64,batch_first=True)
        self.neighbour_encoder=nn.LSTM(input_size=2,hidden_size=64,batch_first=True)
        self.decoder_lstm=nn.LSTMCell(input_size=64,hidden_size=64)
        self.embedding_pos=nn.Linear(64,2)
        self.rel_dis_embedding=nn.Linear(2,64)
        self.multilayer_perceptron=nn.Sequential(nn.Linear(128,512),nn.ReLU(),nn.BatchNorm1d(512),
                                                nn.Linear(512,1024),nn.ReLU(),nn.BatchNorm1d(1024))
        self.multilayer_perceptron_inv=nn.Sequential(nn.Linear(1024+64,1024),nn.ReLU(),nn.BatchNorm1d(1024),
                                                nn.Linear(1024,64),nn.ReLU(),nn.BatchNorm1d(64))
        # self.use_cuda=cuda
    def forward(self,input_dict,mode):
        # import pdb; pdb.set_trace()
        if mode=="train" or mode=="validate":
            agent_traj=input_dict['train_traj']
            # pdb.set_trace()
            batch_neighbours=input_dict['neighbours']
            batch_relative_distance=[]
            for batch_index,curr_neighbours in enumerate(batch_neighbours):
                for neighbour in curr_neighbours:
                    batch_relative_distance.append(neighbour[-1,:]-agent_traj[batch_index,-1,:].numpy())
            # batch_relative_distance=[neighbour[-1,:]-agent_traj[-1,:].numpy() for curr_neighbours in batch_neighbours for neighbour in curr_neighbours]
            all_neighbours_batch=[neighbour for curr_neighbours in batch_neighbours for neighbour in curr_neighbours]
            all_neighbours_tensor=torch.Tensor(all_neighbours_batch).float().cuda()
            batch_relative_distance_tensor=torch.Tensor(batch_relative_distance).float().cuda()
            # batch_relative_distance=torch.Tensor(batch_relative_distance)
            agent_traj=agent_traj.cuda()
            agent_embedding,(agent_h,agent_c)=self.agent_encoder(agent_traj)
            agent_embedding=agent_embedding[:,-1,:]
            neighbour_embedding,_=self.neighbour_encoder(all_neighbours_tensor)
            neighbour_embedding=neighbour_embedding[:,-1,:]
            relative_distance_embedd=self.rel_dis_embedding(batch_relative_distance_tensor)

            perceptron_out=self.multilayer_perceptron(torch.cat((neighbour_embedding,relative_distance_embedd),axis=1))


            count=0
            maxed_embeeding=[]
            for curr_neighbours in batch_neighbours:
                if len(curr_neighbours)!=0:
                    maxed_embeeding.append(torch.max(perceptron_out[count:count+len(curr_neighbours)],axis=0)[0])
                else:
                    maxed_embeeding.append(torch.zeros(1024).cuda())
                count+=len(curr_neighbours)
            
            decoder_embedding=torch.stack(maxed_embeeding,axis=0)
            decoder_embedding=torch.cat((decoder_embedding,agent_embedding),axis=1)
            decoder_embedding=self.multilayer_perceptron_inv(decoder_embedding)

            self.h=decoder_embedding
            self.c=agent_c[0]
            out=[]
            for i in range(30):
                self.h,self.c=self.decoder_lstm(self.h,(self.h,self.c))
                out.append(self.embedding_pos(self.h))
            pred_traj=torch.stack(out,dim=1)
            if mode=="validate":
                # pdb.set_trace()
                R=torch.Tensor([helper['rotation'] for helper in input_dict['helpers']]).float().cuda()
                t=torch.Tensor([helper['mean'] for helper in input_dict['helpers']]).float().cuda()
                pred_traj=torch.matmul(R.permute(0,2,1),pred_traj.permute(0,2,1)).permute(0,2,1)
                pred_traj= pred_traj + t.reshape(t.shape[0],1,2)
                return pred_traj
            else:
                return pred_traj


class Social_Model_Centerline(nn.Module):
    def __init__(self,cuda=False):
        super(Social_Model_Centerline,self).__init__()
        self.agent_encoder=nn.LSTM(input_size=2,hidden_size=64,batch_first=True)
        self.neighbour_encoder=nn.LSTM(input_size=2,hidden_size=64,batch_first=True)
        self.decoder_lstm=nn.LSTMCell(input_size=96,hidden_size=96)
        self.embedding_pos=nn.Linear(96,2)
        self.rel_dis_embedding=nn.Linear(2,64)
        self.multilayer_perceptron=nn.Sequential(nn.Linear(128,512),nn.ReLU(),nn.BatchNorm1d(512),
                                                nn.Linear(512,1024),nn.ReLU(),nn.BatchNorm1d(1024))
        self.multilayer_perceptron_inv=nn.Sequential(nn.Linear(1024+64,1024),nn.ReLU(),nn.BatchNorm1d(1024),
                                                nn.Linear(1024,64),nn.ReLU(),nn.BatchNorm1d(64))
        self.centerline_embed=nn.LSTM(input_size=2,hidden_size=32,batch_first=True)
        # self.use_cuda=cuda
    def forward(self,input_dict,mode):
        # import pdb; pdb.set_trace()
        if mode=="train" or mode=="validate":
            agent_traj=input_dict['train_traj']
            # pdb.set_trace()
            batch_neighbours=input_dict['neighbours']

            len_list=[]
            centerline_norm_list=[]
            for index,helper_dict in enumerate(input_dict['helpers']):
                centerline=helper_dict['ORACLE_CENTERLINE']
                ls=LineString(centerline)
                start_coords=np.array(ls.interpolate(input_dict['ref_t'][index].numpy()+agent_traj[index,0,1].numpy()).coords[:])
                centerline_norm=np.linalg.norm(centerline-start_coords,axis=1)
                min_index=np.argmin(centerline_norm)
                centerline_norm=(centerline-start_coords)[min_index:,:]
                # if self.use_centerline_rotation:
                #     centerline_norm=np.array(centerline_norm)
                #     def rotation_angle(x,y):
                #         angle=np.arctan(abs(y/x))
                #         direction= -1* np.sign(x*y)
                #         return direction*angle
                #     c, s = np.cos(trajectory_rotation), np.sin(trajectory_rotation)
                #     R = np.array([[c,-s], [s, c]])
                #     agent_trajectory=np.transpose(agent_trajectory,[1,0])
                # #     start_pred_coord=np.array(ls.interpolate(input_dict['ref_t'][index].numpy()+input_traj[20,:].numpy()).coords[:])
                # #     centerline_pred_norm=np.linalg.norm(centerline-start_pred_coord,axis=1)
                # #     min_index=np.argmin(centerline_pred_norm)
                    
                centerline_norm_list.append(centerline_norm)
                len_list.append(centerline_norm.shape[0])
            max_len=max(len_list)
            all_centerlines_array=np.zeros((len(len_list),max_len,2))
            for index,centerline_array in enumerate(centerline_norm_list):
                all_centerlines_array[index,-len_list[index]:,:]=centerline_array
            all_centerlines_tensor=torch.Tensor(all_centerlines_array).cuda()
            # pdb.set_trace()
            # self.cent_h,self.cent_c=(torch.zeros(all_centerlines_tensor.shape[0],32).cuda(),torch.zeros(all_centerlines_tensor.shape[0],32).cuda())
            # import pdb; pdb.set_trace()
            _,(self.cent_h,self.cent_c)=self.centerline_embed(all_centerlines_tensor)
            self.cent_h=self.cent_h[0]
            self.cent_c=self.cent_c[0]
            # for i in range(max_len):
            #     self.cent_h,self.cent_c=self.centerline_embed(all_centerlines_tensor[:,i,:],(self.cent_h,self.cent_c))
            



            batch_relative_distance=[]
            for batch_index,curr_neighbours in enumerate(batch_neighbours):
                for neighbour in curr_neighbours:
                    batch_relative_distance.append(neighbour[-1,:]-agent_traj[batch_index,-1,:].numpy())
            # batch_relative_distance=[neighbour[-1,:]-agent_traj[-1,:].numpy() for curr_neighbours in batch_neighbours for neighbour in curr_neighbours]
            all_neighbours_batch=[neighbour for curr_neighbours in batch_neighbours for neighbour in curr_neighbours]
            all_neighbours_tensor=torch.Tensor(all_neighbours_batch).float().cuda()
            batch_relative_distance_tensor=torch.Tensor(batch_relative_distance).float().cuda()
            # batch_relative_distance=torch.Tensor(batch_relative_distance)
            agent_traj=agent_traj.cuda()
            agent_embedding,(agent_h,agent_c)=self.agent_encoder(agent_traj)
            agent_embedding=agent_embedding[:,-1,:]
            neighbour_embedding,_=self.neighbour_encoder(all_neighbours_tensor)
            neighbour_embedding=neighbour_embedding[:,-1,:]
            relative_distance_embedd=self.rel_dis_embedding(batch_relative_distance_tensor)

            perceptron_out=self.multilayer_perceptron(torch.cat((neighbour_embedding,relative_distance_embedd),axis=1))


            count=0
            maxed_embeeding=[]
            for curr_neighbours in batch_neighbours:
                if len(curr_neighbours)!=0:
                    maxed_embeeding.append(torch.max(perceptron_out[count:count+len(curr_neighbours)],axis=0)[0])
                else:
                    maxed_embeeding.append(torch.zeros(1024).cuda())
                count+=len(curr_neighbours)
            
            decoder_embedding=torch.stack(maxed_embeeding,axis=0)
            decoder_embedding=torch.cat((decoder_embedding,agent_embedding),axis=1)
            decoder_embedding=self.multilayer_perceptron_inv(decoder_embedding)

            self.h=decoder_embedding
            self.c=agent_c[0]

            self.h=torch.cat((self.h,self.cent_h),dim=1)
            self.c=torch.cat((self.c,self.cent_c),dim=1)

            out=[]

            for i in range(30):
                self.h,self.c=self.decoder_lstm(self.h,(self.h,self.c))
                out.append(self.embedding_pos(self.h))

            pred_traj=torch.stack(out,dim=1)
            if mode=="validate":
                pred_traj[:,:,1]=pred_traj[:,:,1] + input_dict['ref_t'].unsqueeze(1).cuda()
                all_centerlines=[helper_dict["ORACLE_CENTERLINE"] for helper_dict in input_dict['helpers']]
                pred_unnorm_traj=get_xy_from_nt_seq(pred_traj.detach().cpu().numpy(),all_centerlines)
                pred_unnorm_traj=torch.Tensor(pred_unnorm_traj).float().cuda()
                return pred_unnorm_traj
            else:
                return pred_traj

class MultiPrediction(nn.Module):
    pass


class SinglePredictionImage(nn.Module):
    pass

class MultiPredictionImage(nn.Module):
    pass
class SocialGan(nn.Module):
    pass


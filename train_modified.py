from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
from statistics import mean
import glob
import torch
import pandas as pd
from argoverse.visualization.visualize_sequences import viz_sequence
# torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn

from data import Argoverse_Data,Argoverse_Social_Data,Argoverse_LaneCentre_Data,\
                collate_traj_social,collate_traj_social_test,collate_traj_lanecentre
from model import LSTMModel, TCNModel, Social_Model
from vrae_model import VRAE
from argoverse.evaluation.eval_forecasting import get_ade, get_fde
from argoverse.evaluation.competition_util import generate_forecasting_h5
import matplotlib.pyplot as plt
import argparse
import warnings
from time import localtime, strftime
# from logtger import TensorLogger
import numpy as np
from visualize import viz_predictions
import os
import threading
import copy
import resource
import pdb
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

class Trainer():
    def __init__(self,model,use_cuda,parallel,optimizer,train_loader,\
        val_loader,test_loader,loss_fn,num_epochs,writer,args,modeldir,max_grad_norm=5,clip=False,model_type=None):
        self.model=model
        #self.test_model=copy.deepcopy(model)
        self.use_cuda=use_cuda
        self.parallel = parallel
        if self.use_cuda:
            self.model=self.model.cuda()
            #self.test_model=self.test_model.cuda()
        if self.parallel:
            self.model = nn.DataParallel(self.model)
        
        self.optimizer=optimizer
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.test_loader=test_loader
        self.loss_fn=loss_fn
        self.num_epochs=num_epochs
        
        self.best_1_ade = np.inf
        self.best_1_fde = np.inf
        self.best_3_ade = np.inf
        self.best_3_fde = np.inf

        self.writer = writer
        self.args = args
        self.model_dir = modeldir
        self.model_type = model_type

        self.clip = clip
        self.max_grad_norm = max_grad_norm

    def check_normalization(self):
        for i_batch,traj_dict in enumerate(self.val_loader):
            pred_traj=traj_dict['gt_agent']
            pred_traj=self.train_loader.dataset.inverse_transform(pred_traj,traj_dict)
            gt_traj=traj_dict['gt_unnorm_agent']
            if self.use_cuda:
                pred_traj=pred_traj.cuda()
                gt_traj=gt_traj.cuda()
            loss=self.loss_fn(pred_traj,gt_traj)
            print(f"Batch: {i_batch}, Loss: {loss.data}")

    def train_epoch(self):
        total_loss=0
        num_batches=len(self.train_loader.batch_sampler)
        self.model.train()
        no_samples=0
        for i_batch,traj_dict in enumerate(self.train_loader):
            gt_traj=traj_dict['gt_agent']
            if self.use_cuda:
                gt_traj=gt_traj.cuda()
            
            if self.model_type == 'VRAE':
                pred_traj, latent_traj, latent_mean, latent_logvar = self.model(traj_dict)
                kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
                mse_loss=self.loss_fn(pred_traj,gt_traj)
                loss = kl_loss + mse_loss
            else:
                pred_traj=self.model(traj_dict)
                loss=self.loss_fn(pred_traj,gt_traj)

            total_loss=total_loss+loss.data
            avg_loss = float(total_loss)/(i_batch+1)
            print(f"Training Iter {i_batch+1}/{num_batches} Avg Loss {avg_loss:.4f}",end="\r")
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm = self.max_grad_norm)
            self.optimizer.step()
        
        return total_loss/(num_batches)

    def val_epoch(self, epoch):
        total_loss=0
        num_batches=len(self.val_loader.batch_sampler)
        self.model.eval()
        
        ade_one_sec,fde_one_sec,ade_three_sec,fde_three_sec=(0,0,0,0)
        ade_one_sec_avg, fde_one_sec_avg ,ade_three_sec_avg, fde_three_sec_avg = (0,0,0,0)
        no_samples=0
        
        for i_batch,traj_dict in enumerate(self.val_loader):
            gt_traj = traj_dict['gt_unnorm_agent']
            if self.use_cuda:
                gt_traj=gt_traj.cuda()
            
            if self.model_type == 'VRAE':
                pred_traj, latent_traj, latent_mean, latent_logvar = self.model(traj_dict)
                pred_traj = self.val_loader.dataset.inverse_transform(pred_traj,traj_dict)
                kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
                mse_loss=self.loss_fn(pred_traj,gt_traj)
                loss = kl_loss + mse_loss
            else:
                pred_traj=self.model(traj_dict)
                pred_traj=self.val_loader.dataset.inverse_transform(pred_traj,traj_dict)
                loss=self.loss_fn(pred_traj,gt_traj)

            total_loss=total_loss+loss.data
            batch_samples=gt_traj.shape[0]           
            
            ade_one_sec+=sum([get_ade(pred_traj[i,:10,:],gt_traj[i,:10,:]) for i in range(batch_samples)])
            fde_one_sec+=sum([get_fde(pred_traj[i,:10,:],gt_traj[i,:10,:]) for i in range(batch_samples)])
            ade_three_sec+=sum([get_ade(pred_traj[i,:,:],gt_traj[i,:,:]) for i in range(batch_samples)])
            fde_three_sec+=sum([get_fde(pred_traj[i,:,:],gt_traj[i,:,:]) for i in range(batch_samples)])
            
            no_samples+=batch_samples
            ade_one_sec_avg = float(ade_one_sec)/no_samples
            ade_three_sec_avg = float(ade_three_sec)/no_samples
            fde_one_sec_avg = float(fde_one_sec)/no_samples
            fde_three_sec_avg = float(fde_three_sec)/no_samples

            # if (i_batch+1) % self.args.val_log_interval == 0:
            print(f"Validation Iter {i_batch+1}/{num_batches} Avg Loss {total_loss/(i_batch+1):.4f} \
            One sec:- ADE:{ade_one_sec/(no_samples):.4f} FDE: {fde_one_sec/(no_samples):.4f}\
            Three sec:- ADE:{ade_three_sec/(no_samples):.4f} FDE: {fde_three_sec/(no_samples):.4f}",end="\r")

            _filename = self.model_dir + 'best-model.pt'

            if ade_three_sec_avg < self.best_3_ade and fde_three_sec_avg < self.best_3_fde:    
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'loss': total_loss/(i_batch+1)
                }, _filename)

                self.best_1_ade = ade_one_sec_avg
                self.best_1_fde = fde_one_sec_avg
                self.best_3_ade = ade_three_sec_avg
                self.best_3_fde = fde_three_sec_avg
                self.best_model_updated=True
        print()
        return total_loss/(num_batches), ade_one_sec/no_samples,fde_one_sec/no_samples,ade_three_sec/no_samples,fde_three_sec/no_samples
    
    # def test_epoch(self):
    #     num_batches=len(self.test_loader.batch_sampler)
    #     batch_size=self.test_loader.batch_size
    #     if model_dir is None:
    #         self.test_model.load_state_dict(torch.load(self.model_dir+'best-model.pt')['model_state_dict'])
    #     else:
    #         self.test_model.load_state_dict(torch.load(model_dir+'best-model.pt')['model_state_dict'])
    #     self.test_model.eval()
    #     no_samples=0
    #     output_all = {}
    #     for i_batch,traj_dict in enumerate(self.test_loader):
    #         seq_index=traj_dict['seq_index']
    #         pred_traj=self.test_model(traj_dict)
    #         pred_traj=self.test_loader.dataset.inverse_transform(pred_traj,traj_dict)
    #         if self.use_cuda:
    #             pred_traj=pred_traj.cpu()
    #         output_all.update({seq_index[index]:pred_traj[index].detach().repeat(9,1,1) for index in range(pred_traj.shape[0])})
    #         print(f"Test Iter {i_batch+1}/{num_batches}",end="\r")
    #     print()
    #     print("Saving the test data results in dir",self.test_path)
    #     self.save_trajectory(output_all)
    #     self.save_top_errors_accuracy()

    def save_trajectory(self,output_dict,save_path):
        generate_forecasting_h5(output_dict, save_path)
        print("done")

    def validate_model(self, model_path):
        # total_loss=0
        # num_batches=len(self.val_loader.batch_sampler)
        # self.model.load_state_dict(torch.load(model_path+'best-model.pt')['model_state_dict'])
        # self.model.eval()
        # ade_one_sec,fde_one_sec,ade_three_sec,fde_three_sec=(0,0,0,0)
        # ade_one_sec_avg, fde_one_sec_avg ,ade_three_sec_avg, fde_three_sec_avg = (0,0,0,0)
        # no_samples=0
        
        # for i_batch,traj_dict in enumerate(self.val_loader):
        #     gt_traj=traj_dict['gt_unnorm_agent']
        #     if self.use_cuda:
        #         gt_traj=gt_traj.cuda()
            
        #     if self.model_type == 'VRAE':
        #         pred_traj, latent_traj, latent_mean, latent_logvar = self.model(traj_dict)
        #         pred_traj = self.val_loader.dataset.inverse_transform(pred_traj,traj_dict)
        #         kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        #         mse_loss=self.loss_fn(pred_traj,gt_traj)
        #         loss = kl_loss + mse_loss
        #     else:
        #         pred_traj=self.model(traj_dict)
        #         pred_traj=self.val_loader.dataset.inverse_transform(pred_traj,traj_dict)
        #         loss=self.loss_fn(pred_traj,gt_traj)

            
        #     total_loss=total_loss+loss.data
        #     batch_samples=gt_traj.shape[0]           
            
        #     ade_one_sec+=sum([get_ade(pred_traj[i,:10,:],gt_traj[i,:10,:]) for i in range(batch_samples)])
        #     fde_one_sec+=sum([get_fde(pred_traj[i,:10,:],gt_traj[i,:10,:]) for i in range(batch_samples)])
        #     ade_three_sec+=sum([get_ade(pred_traj[i,:,:],gt_traj[i,:,:]) for i in range(batch_samples)])
        #     fde_three_sec+=sum([get_fde(pred_traj[i,:,:],gt_traj[i,:,:]) for i in range(batch_samples)])
            
        #     no_samples+=batch_samples
        #     ade_one_sec_avg = float(ade_one_sec)/no_samples
        #     ade_three_sec_avg = float(ade_three_sec)/no_samples
        #     fde_one_sec_avg = float(fde_one_sec)/no_samples
        #     fde_three_sec_avg = float(fde_three_sec)/no_samples

        #     print(f"Validation Iter {i_batch+1}/{num_batches} Avg Loss {total_loss/(i_batch+1):.4f} \
        #     One sec:- ADE:{ade_one_sec/(no_samples):.4f} FDE: {fde_one_sec/(no_samples):.4f}\
        #     Three sec:- ADE:{ade_three_sec/(no_samples):.4f} FDE: {fde_three_sec/(no_samples):.4f}",end="\r")

        # print()
        self.save_top_errors_accuracy(self.model_dir, model_path)
        print("Saved error plots")

    def test_model(self,model_dir):
        num_batches=len(self.test_loader.batch_sampler)
        batch_size=self.test_loader.batch_size
        self.model.load_state_dict(torch.load(model_dir+'best-model.pt')['model_state_dict'])
        self.model.eval()
        no_samples=0
        output_all = {}
        for i_batch,traj_dict in enumerate(self.test_loader):
            seq_index=traj_dict['seq_index']
            pred_traj=self.model(traj_dict)
            pred_traj=self.test_loader.dataset.inverse_transform(pred_traj,traj_dict)
            if self.use_cuda:
                pred_traj=pred_traj.cpu()
            output_all.update({seq_index[index]:pred_traj[index].detach().repeat(9,1,1) for index in range(pred_traj.shape[0])})
            print(f"Test Iter {i_batch+1}/{num_batches}",end="\r")
        print()
        print("Saving the test data results in dir",model_dir)
        self.save_trajectory(output_all,model_dir)
        self.save_top_errors_accuracy(model_dir)

    def save_top_errors_accuracy(self,model_dir, model_path):
        self.model.load_state_dict(torch.load(model_path+'best-model.pt')['model_state_dict'])
        self.model.eval()
        min_loss=np.inf
        max_loss=0
        num_images=10
        
        loss_list_max=[]
        input_max_list=[]
        pred_max_list=[]
        target_max_list=[]
        city_name_max=[]
        seq_path_list_max=[]
        
        loss_list_min=[]
        input_min_list=[]
        pred_min_list=[]
        target_min_list=[]
        city_name_min=[]
        seq_path_list_min=[]


        print ("here")
        num_batches=len(self.val_loader.batch_sampler)
        for i_batch,traj_dict in enumerate(self.val_loader):
            print(f"Running {i_batch}/{num_batches}",end="\r")
            gt_traj=traj_dict['gt_unnorm_agent']
            train_traj=traj_dict['train_agent']
            if self.use_cuda:
                train_traj=train_traj.cuda()
            input_=self.val_loader.dataset.inverse_transform(train_traj,traj_dict)
            output=self.val_loader.dataset.inverse_transform(self.model(traj_dict),traj_dict)
            if self.use_cuda:
                output=output.cpu()
                input_=input_.cpu()
            
            loss=torch.norm(output.reshape(output.shape[0],-1)-gt_traj.reshape(gt_traj.shape[0],-1),dim=1)
            min_loss,min_index=torch.min(loss,dim=0)
            max_loss,max_index=torch.max(loss,dim=0)

            
            input_min_list.append(input_[min_index])
            pred_min_list.append(output[min_index])
            target_min_list.append(gt_traj[min_index])

            input_max_list.append(input_[max_index])
            pred_max_list.append(output[max_index])
            target_max_list.append(gt_traj[max_index])
            
            city_name_min.append(traj_dict['city'][min_index])
            city_name_max.append(traj_dict['city'][max_index])

            seq_path_list_max.append(traj_dict['seq_path'][max_index])
            seq_path_list_min.append(traj_dict['seq_path'][min_index])

            loss_list_max.append(min_loss.data)
            loss_list_min.append(max_loss.data)
           
        
        loss_list_max_array=np.array(loss_list_max)
        loss_list_max=list(loss_list_max_array.argsort()[-num_images:][::-1])

        loss_list_min_array=np.array(loss_list_min)
        loss_list_min=list(loss_list_min_array.argsort()[:num_images])

        avm=ArgoverseMap()
        
        high_error_path=model_dir+"/visualization/high_errors/"
        low_error_path=model_dir+"/visualization/low_errors/"

        if not os.path.exists(high_error_path):
            os.makedirs(high_error_path)

        if not os.path.exists(low_error_path):
            os.makedirs(low_error_path)

        input_max=[]
        pred_max=[]
        target_max=[]
        city_max=[]

        centerlines_max=[]
        for i,index in enumerate(loss_list_max):
            print(f"Max: {i}")
            input_max.append(input_max_list[index].detach().numpy())
            pred_max.append([pred_max_list[index].detach().numpy()])
            target_max.append(target_max_list[index].detach().numpy())
            city_max.append(city_name_max[index])
            viz_sequence(df=pd.read_csv(seq_path_list_max[index]) ,save_path=f"{high_error_path}/dataframe_{i}.png",show=True,avm=avm)
            centerlines_max.append(avm.get_candidate_centerlines_for_traj(input_max[-1], city_max[-1],viz=False))
        print("Created max array")
        
        input_min=[]
        pred_min=[]
        target_min=[]
        city_min=[]
        centerlines_min=[]
        for i,index in enumerate(loss_list_min):
            print(f"Min: {i}")
            input_min.append(input_min_list[index].detach().numpy())
            pred_min.append([pred_min_list[index].detach().numpy()])
            target_min.append(target_min_list[index].detach().numpy())
            city_min.append(city_name_min[index])
            # seq_path_min.append(seq_path_list_min[index])
            viz_sequence(df=pd.read_csv(seq_path_list_min[index]) ,save_path=f"{low_error_path}/dataframe_{i}.png",show=True,avm=avm)
            centerlines_min.append(avm.get_candidate_centerlines_for_traj(input_min[-1], city_min[-1],viz=False))
        print("Created min array")

        print(f"Saving max visualizations at {high_error_path}")
        viz_predictions(input_=np.array(input_max), output=pred_max,target=np.array(target_max),centerlines=centerlines_max,city_names=np.array(city_max),avm=avm,save_path=high_error_path)
        
        print(f"Saving min visualizations at {low_error_path}")
        viz_predictions(input_=np.array(input_min), output=pred_min,target=np.array(target_min),centerlines=centerlines_min,city_names=np.array(city_min),avm=avm,save_path=low_error_path)


    def run(self):
        if args.mode=="train":
            for epoch in range(self.num_epochs):
                print(f"\nEpoch {epoch}: ")
                avg_loss_train=self.train_epoch()
                avg_loss_val,ade_one_sec,fde_one_sec,ade_three_sec,fde_three_sec = self.val_epoch(epoch)
                # if (epoch+1==self.num_epochs):
                    # self.test_epoch()
                    #self.test_model(self.model_dir)
                # self.writer.scalar_summary('Val/1ADE_Epoch', ade_one_sec, epoch)
                # self.writer.scalar_summary('Val/3ADE_Epoch', ade_three_sec, epoch)
                # self.writer.scalar_summary('Val/1FDE_Epoch', fde_one_sec, epoch)
                # self.writer.scalar_summary('Val/3FDE_Epoch', fde_three_sec, epoch)
        elif args.mode=="validate":
            self.validate_model(self.model_dir)
        # elif args.mode=="test":
            #self.test_model(self.model_dir)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='Sequence Modeling - Argoverse Forecasting Task')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA (default: True)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for optimizer (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit (default: 10)')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='model type to execute (default: LSTM Baseline)')
    parser.add_argument('--data', type=str, default='XY',
                        help='type of data to use for training (default: XY, options: XY,LaneCentre,')
    parser.add_argument('--mode',type=str,default='train',help='mode: train, test ,validate')
    parser.add_argument('--model_dir',type=str,default=None,help='model path for test or validate')

    # parser.add_argument('--social',action='store_true',help='use neighbour data as well. default: False')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (default: 0.2)')
   
    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip, -1 means no clip (default: -1)')
    
    parser.add_argument('--ksize', type=int, default=3,
                        help='kernel size (default: 3)')
    parser.add_argument('--levels', type=int, default=10,
                        help='# of levels (default: 8)')
    parser.add_argument('--nhid', type=int, default=20,
                        help='number of hidden units per layer (default: 20)')
    parser.add_argument('--opsize', type=int, default=30,
                        help='number of output units for model (default: 30)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default: 1111)')
    
    
    args = parser.parse_args()

    curr_time = strftime("%Y%m%d%H%M%S", localtime())

    # initialize model and params
    
    if args.model == 'LSTM':
        model = LSTMModel(cuda=args.cuda)
    elif args.model == 'TCN':
        channel_sizes = [args.nhid] * args.levels
        model = TCNModel(args.nhid, args.opsize, channel_sizes, args.ksize, args.dropout, 128)
    elif args.model == 'SOCIAL':
        model = Social_Model(cuda=args.cuda)
    elif args.model == 'VRAE':
        model = VRAE(sequence_length=20, number_of_features=2)
    
    if args.mode is 'train':
        logger_dir = './runs/' + args.model + '/' + curr_time + '/'
        model_dir = './models/' + args.model + '/' + curr_time + '/'
        os.makedirs(model_dir)
    else:
        logger_dir=None
        model_dir=args.model_dir

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    print("CUDA is ",args.cuda)
    print("Model is ",args.model)
    print("Data is", args.data)
    print("Model dir is",model_dir)
    print(f"Training for {args.epochs} epochs")
    # Load data module
    # torch.multiprocessing.set_sharing_strategy('file_system')
    if args.data=="social":
        argoverse_train=Argoverse_Social_Data('data/train/data/',cuda=args.cuda)
        argoverse_val=Argoverse_Social_Data('data/val/data',cuda=args.cuda)
        argoverse_test = Argoverse_Social_Data('data/test_obs/data',cuda=args.cuda,test=True)
        train_loader = DataLoader(argoverse_train, batch_size=args.batch_size,
                        shuffle=True, num_workers=1,collate_fn=collate_traj_social)
        val_loader = DataLoader(argoverse_val, batch_size=args.batch_size,
                        shuffle=True, num_workers=1,collate_fn=collate_traj_social)
        test_loader = DataLoader(argoverse_test, batch_size=args.batch_size,
                        shuffle=True, num_workers=1,collate_fn=collate_traj_social_test)
    elif args.data=="LaneCentre":
        argoverse_map=ArgoverseMap()
        argoverse_train=Argoverse_LaneCentre_Data('data/train/data/',cuda=args.cuda,avm=argoverse_map)
        argoverse_val=Argoverse_LaneCentre_Data('data/val/data',cuda=args.cuda,avm=argoverse_map)
        argoverse_test = Argoverse_LaneCentre_Data('data/test_obs/data',cuda=args.cuda,test=True,avm=argoverse_map)
        train_loader = DataLoader(argoverse_train, batch_size=args.batch_size,
                        shuffle=True, num_workers=10,collate_fn=collate_traj_lanecentre)
        val_loader = DataLoader(argoverse_val, batch_size=args.batch_size,
                        shuffle=True, num_workers=10,collate_fn=collate_traj_lanecentre)
        test_loader = DataLoader(argoverse_test, batch_size=args.batch_size,
                        shuffle=True, num_workers=10,collate_fn=collate_traj_lanecentre)
    elif args.data=="XY":
        argoverse_train=Argoverse_Data('data/train/data/',cuda=args.cuda)
        argoverse_val=Argoverse_Data('data/val/data',cuda=args.cuda)
        argoverse_test = Argoverse_Data('data/test_obs/data',cuda=args.cuda,test=True)
        train_loader = DataLoader(argoverse_train, batch_size=args.batch_size,
                            shuffle=True, num_workers=10)
        val_loader = DataLoader(argoverse_val, batch_size=args.batch_size,
                            shuffle=True, num_workers=10)
        val_metric_loader = DataLoader(argoverse_val, batch_size=1,
                            shuffle=True, num_workers=1)
        test_loader = DataLoader(argoverse_test, batch_size=args.batch_size,
                            shuffle=False, num_workers=10)
    else:
        # raise ValueError('A very specific bad thing happened')
        raise ValueError(f"Dataset: {args.data} not present")

    loss_fn=nn.MSELoss()
    count=0
    total=0

    print("Argoverse train dataloder is of size", len(train_loader.batch_sampler))
    print("Argoverse val dataloader is of size", len(val_loader.batch_sampler))
    print("Argoverse test dataloader is of size", len(test_loader.batch_sampler))

    _parallel = False
    trainer=Trainer(model=model,use_cuda=args.cuda,parallel=_parallel,optimizer=optimizer,\
        train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,loss_fn=loss_fn,\
            num_epochs=args.epochs,writer=None,args=args,modeldir=model_dir)
    trainer.run()

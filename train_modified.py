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
import pickle

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

class Trainer():
    def __init__(self,model,use_cuda,parallel,optimizer,train_loader,\
        val_loader,test_loader,loss_fn,num_epochs,writer,args,modeldir,max_grad_norm=5,clip=False):
        self.model=model
        self.use_cuda=use_cuda
        self.parallel = parallel
        if self.use_cuda:
            self.model=self.model.cuda()
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
        self.model_type = args.model_type

        self.clip = clip
        self.max_grad_norm = max_grad_norm

    def check_normalization(self):
        all_loss=[]
        for i_batch,traj_dict in enumerate(self.val_loader):
            pred_traj=traj_dict['gt_agent']
            pred_traj=self.train_loader.dataset.inverse_transform(pred_traj,traj_dict)
            gt_traj=traj_dict['gt_unnorm_agent']
            if self.use_cuda:
                pred_traj=pred_traj.cuda()
                gt_traj=gt_traj.cuda()
            
            loss=torch.norm(pred_traj.reshape(pred_traj.shape[0],-1)-gt_traj.reshape(gt_traj.shape[0],-1),dim=1)
            min_loss,min_index=torch.min(loss,dim=0)
            max_loss,max_index=torch.max(loss,dim=0)
            all_loss.append(max_loss)
            print(f"Batch: {i_batch}, Min Loss: {min_loss:.5f}, Max Loss:{max_loss:.5f}")
        print(f"Max batch loss: {max(all_loss)}")
    
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

            print(f"Validation Iter {i_batch+1}/{num_batches} Avg Loss {total_loss/(i_batch+1):.4f} \
            One sec:- ADE:{ade_one_sec/(no_samples):.4f} FDE: {fde_one_sec/(no_samples):.4f}\
            Three sec:- ADE:{ade_three_sec/(no_samples):.4f} FDE: {fde_three_sec/(no_samples):.4f}",end="\r")
        
        _filename = self.model_dir + 'best-model.pt'

        if ade_three_sec_avg < self.best_3_ade and fde_three_sec_avg < self.best_3_fde:    
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'opt_state_dict': self.optimizer.state_dict(),
                'loss': total_loss/(i_batch+1)
            }, _filename)

            self.best_1_ade = ade_one_sec_avg
            self.best_1_fde = fde_one_sec_avg
            self.best_3_ade = ade_three_sec_avg
            self.best_3_fde = fde_three_sec_avg
            self.best_model_updated=True
        
        print()
        return total_loss/(num_batches), ade_one_sec/no_samples,fde_one_sec/no_samples,ade_three_sec/no_samples,fde_three_sec/no_samples
    
    def test_epoch(self):
        num_batches=len(self.test_loader.batch_sampler)
        batch_size=self.test_loader.batch_size
        
        if model_dir is None:
            self.test_model.load_state_dict(torch.load(self.model_dir+'best-model.pt')['model_state_dict'])
        else:
            self.test_model.load_state_dict(torch.load(model_dir+'best-model.pt')['model_state_dict'])
        self.test_model.eval()
        no_samples=0
        output_all = {}
        for i_batch,traj_dict in enumerate(self.test_loader):
            seq_index=traj_dict['seq_index']
            pred_traj=self.test_model(traj_dict)
            pred_traj=self.test_loader.dataset.inverse_transform(pred_traj,traj_dict)
            if self.use_cuda:
                pred_traj=pred_traj.cpu()
            output_all.update({seq_index[index]:pred_traj[index].detach().repeat(9,1,1) for index in range(pred_traj.shape[0])})
            print(f"Test Iter {i_batch+1}/{num_batches}",end="\r")
        print()
        print("Saving the test data results in dir",self.test_path)
        self.save_trajectory(output_all)
        self.save_top_errors_accuracy()

    def save_trajectory(self,output_dict,save_path):
        generate_forecasting_h5(output_dict, save_path)
        print("done")

    def validate_model(self, model_path):
        total_loss=0
        num_batches=len(self.val_loader.batch_sampler)
        self.model.load_state_dict(torch.load(model_path+'best-model.pt')['model_state_dict'])
        self.model.eval()
        ade_one_sec,fde_one_sec,ade_three_sec,fde_three_sec=(0,0,0,0)
        ade_one_sec_avg, fde_one_sec_avg ,ade_three_sec_avg, fde_three_sec_avg = (0,0,0,0)
        no_samples=0
        
        for i_batch,traj_dict in enumerate(self.val_loader):
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

            print(f"Validation Iter {i_batch+1}/{num_batches} Avg Loss {total_loss/(i_batch+1):.4f} \
            One sec:- ADE:{ade_one_sec/(no_samples):.4f} FDE: {fde_one_sec/(no_samples):.4f}\
            Three sec:- ADE:{ade_three_sec/(no_samples):.4f} FDE: {fde_three_sec/(no_samples):.4f}",end="\r")

        print()
        self.save_results_single_pred()

    def test_model_fn(self,model_dir):
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


    def save_results_single_pred(self):
        print("running save results")
        afl=ArgoverseForecastingLoader("data/val/data/")
        checkpoint = torch.load(self.model_dir+'best-model.pt', map_location=lambda storage, loc: storage)
        # self.model.load_state_dict(torch.load(self.model_dir+'best-model.pt')['model_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        save_results_path=self.model_dir+"/results/"
        # pdb.set_trace()
        if not os.path.exists(save_results_path):
            os.mkdir(save_results_path)
        num_batches=len(self.val_loader.batch_sampler)
        
        for i_batch,traj_dict in enumerate(self.val_loader):
            print(f"Running {i_batch}/{num_batches}",end="\r")
            gt_traj=traj_dict['gt_unnorm_agent'].numpy()
            # output=self.model(traj_dict,mode='validate')
            output=self.model(traj_dict)
            output=self.val_loader.dataset.inverse_transform(output,traj_dict)
            
            output=output.detach().cpu().numpy()
            seq_paths=traj_dict['seq_path']
            
            for index,seq_path in enumerate(seq_paths):
                loader=afl.get(seq_path)
                input_array=loader.agent_traj[0:20,:]
                city=loader.city
                del loader
                seq_index=int(os.path.basename(seq_path).split('.')[0])

                output_dict={'seq_path':seq_path,'seq_index':seq_index,'input':input_array,
                            'output':output[index],'target':gt_traj[index],'city':city}
                with open(f"{save_results_path}/{seq_index}.pkl", 'wb') as f:
                    pickle.dump(output_dict,f) 

    def run(self):
        if args.mode=="train":
            for epoch in range(self.num_epochs):
                print(f"\nEpoch {epoch}: ")
                avg_loss_train=self.train_epoch()
                avg_loss_val,ade_one_sec,fde_one_sec,ade_three_sec,fde_three_sec = self.val_epoch(epoch)
        elif args.mode=="validate":
            self.validate_model(self.model_dir)
        elif args.mode=="test":
            self.test_model(self.model_dir)

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
    parser.add_argument('--model_type', type=str, default='none',
                        help='model type to execute (default: none, pass VRAE for executing)')
    
    
    args = parser.parse_args()

    curr_time = strftime("%Y%m%d%H%M%S", localtime())
    # args.cuda = torch.cuda.is_available()

    # initialize model and params
    
    if args.model == 'LSTM':
        model = LSTMModel(cuda=args.cuda)
    elif args.model == 'TCN':
        channel_sizes = [args.nhid] * args.levels
        model = TCNModel(args.nhid, args.opsize, channel_sizes, args.ksize, args.dropout, 128, use_cuda=args.cuda)
    elif args.model == 'SOCIAL':
        model = Social_Model(cuda=args.cuda)
    elif args.model == 'VRAE':
        model = VRAE(sequence_length=30, number_of_features=2, block='GRU')
    
    if args.mode == 'train':
        logger_dir = './runs/' + args.model + '/' + curr_time + '/'
        model_dir = './models/' + args.model + '/' + curr_time + '/'
        os.makedirs(model_dir)
    else:
        logger_dir=None
        model_dir=args.model_dir

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    print("CUDA is ",args.cuda)
    print("Mode is ",args.mode)
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
                        shuffle=False, num_workers=10,collate_fn=collate_traj_lanecentre)
        test_loader = DataLoader(argoverse_test, batch_size=args.batch_size,
                        shuffle=False, num_workers=10,collate_fn=collate_traj_lanecentre)
    elif args.data=="XY":
        argoverse_train=Argoverse_Data('data/train/data/',cuda=args.cuda)
        argoverse_val=Argoverse_Data('data/val/data',cuda=args.cuda)
        argoverse_test = Argoverse_Data('data/test_obs/data',cuda=args.cuda,test=True)
        train_loader = DataLoader(argoverse_train, batch_size=args.batch_size,
                            shuffle=True, num_workers=10, drop_last=True)
        val_loader = DataLoader(argoverse_val, batch_size=args.batch_size,
                            shuffle=True, num_workers=10, drop_last=True)
        val_metric_loader = DataLoader(argoverse_val, batch_size=1,
                            shuffle=True, num_workers=1)
        test_loader = DataLoader(argoverse_test, batch_size=args.batch_size,
                            shuffle=False, num_workers=10, drop_last=True)
    else:
        # raise ValueError('A very specific bad thing happened')
        raise ValueError(f"Dataset: {args.data} not present")

    loss_fn=nn.MSELoss()
    count=0
    total=0

    print("Argoverse train dataloder is of size", len(train_loader.batch_sampler))
    print("Argoverse val dataloader is of size", len(val_loader.batch_sampler))
    print("Argoverse test dataloader is of size", len(test_loader.batch_sampler))
    # import pdb;pdb.set_trace();
    _parallel = False
    trainer=Trainer(model=model,use_cuda=args.cuda,parallel=_parallel,optimizer=optimizer,\
        train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,loss_fn=loss_fn,\
            num_epochs=args.epochs,writer=None,args=args,modeldir=model_dir)
    trainer.run()

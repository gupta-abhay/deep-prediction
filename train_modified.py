from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
from statistics import mean
import glob
import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn

from data import Argoverse_Data,Argoverse_Social_Data,Argoverse_LaneCentre_Data,\
                collate_traj_social,collate_traj_social_test,collate_traj_lanecentre
from model import LSTMModel, TCNModel, Social_Model
from argoverse.evaluation.eval_forecasting import get_ade, get_fde
from argoverse.evaluation.competition_util import generate_forecasting_h5
import matplotlib.pyplot as plt
import argparse
import warnings
from time import localtime, strftime
# from logtger import TensorLogger
import numpy as np
import os
import threading
import copy
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

class Trainer():
    def __init__(self,model,use_cuda,parallel,optimizer,train_loader,\
        val_loader,test_loader,loss_fn,num_epochs,writer,args,modeldir,testdir):
        self.model=model
        self.test_model=copy.deepcopy(model)
        self.test_path = testdir
        
        self.use_cuda=use_cuda
        self.parallel = parallel
        if self.use_cuda:
            self.model=self.model.cuda()
            self.test_model=self.test_model.cuda()
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
        # self.model_dir="./models/LSTM/20191104171311/"
        # self.test_path="./test/LSTM/20191104171311/"
    # def inverse_transform_rotation_translation(self,trajectory,R,t):
    #     out=torch.matmul(R.permute(0,2,1),trajectory.permute(0,2,1)).permute(0,2,1)
    #     out=out-t.reshape(t.shape[0],1,2)
    #     return out
    def check_normalization(self):
        for i_batch,traj_dict in enumerate(self.train_loader):
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
            pred_traj=self.model(traj_dict)
            gt_traj=traj_dict['gt_agent']
            if self.use_cuda:
                gt_traj=gt_traj.cuda()
            loss=self.loss_fn(pred_traj,gt_traj)
            total_loss=total_loss+loss.data
            avg_loss = float(total_loss)/(i_batch+1)
            print(f"Training Iter {i_batch+1}/{num_batches} Avg Loss {avg_loss:.4f}",end="\r")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print()
        return total_loss/(num_batches)

    def val_epoch(self, epoch):
        total_loss=0
        num_batches=len(self.val_loader.batch_sampler)
        self.model.eval()
        
        ade_one_sec,fde_one_sec,ade_three_sec,fde_three_sec=(0,0,0,0)
        ade_one_sec_avg, fde_one_sec_avg ,ade_three_sec_avg, fde_three_sec_avg = (0,0,0,0)
        no_samples=0
        
        for i_batch,traj_dict in enumerate(self.val_loader):
            pred_traj=self.model(traj_dict)
            pred_traj=self.val_loader.dataset.inverse_transform(pred_traj,traj_dict)
            gt_traj=traj_dict['gt_unnorm_agent']
            # R=traj_dict['rotation']
            # t=traj_dict['translation']
            if self.use_cuda:
                gt_traj=gt_traj.cuda()
                # R=R.cuda()
                # t=t.cuda()

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

            # self.writer.scalar_summary('Val/AvgLoss', float(total_loss)/(i_batch+1), i_batch+1)
            # self.writer.scalar_summary('Val/1ADE', ade_one_sec_avg, i_batch+1)
            # self.writer.scalar_summary('Val/3ADE', ade_three_sec_avg, i_batch+1)
            # self.writer.scalar_summary('Val/1FDE', fde_one_sec_avg, i_batch+1)
            # self.writer.scalar_summary('Val/3FDE', fde_three_sec_avg, i_batch+1)

            # if (i_batch+1) % self.args.val_log_interval == 0:
            print(f"Validation Iter {i_batch+1}/{num_batches} Avg Loss {total_loss/(i_batch+1):.4f} \
            One sec:- ADE:{ade_one_sec/(no_samples):.4f} FDE: {fde_one_sec/(no_samples):.4f}\
            Three sec:- ADE:{ade_three_sec/(no_samples):.4f} FDE: {fde_three_sec/(no_samples):.4f}",end="\r")

            _filename = self.model_dir + 'best-model.pt'

            # if ade_one_sec_avg < self.best_1_ade and ade_three_sec_avg < self.best_3_ade and fde_one_sec_avg < self.best_1_fde and fde_three_sec_avg < self.best_3_fde:
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
    
    def save_trajectory(self,output_dict):
        generate_forecasting_h5(output_dict, self.test_path)
        print("done")
    def test_epoch(self):
        num_batches=len(self.test_loader.batch_sampler)
        batch_size=self.test_loader.batch_size
        self.test_model.load_state_dict(torch.load(self.model_dir+'best-model.pt')['model_state_dict'])
        self.test_model.eval()
        no_samples=0
        output_all = {}
        for i_batch,traj_dict in enumerate(self.test_loader):
            seq_index=traj_dict['seq_index']
            # R=traj_dict['rotation']
            # t=traj_dict['translation']
            # if self.use_cuda:
            #     R=R.cuda()
            #     t=t.cuda()
            pred_traj=self.test_model(traj_dict)
            pred_traj=self.test_loader.dataset.inverse_transform(pred_traj,traj_dict)
            if self.use_cuda:
                pred_traj=pred_traj.cpu()
            output_all.update({seq_index[index]:pred_traj[index].detach().repeat(9,1,1) for index in range(pred_traj.shape[0])})
            print(f"Test Iter {i_batch+1}/{num_batches}",end="\r")
        print()
        print("Saving the test data results in dir",self.test_path)
        self.save_trajectory(output_all)
        
    def save_top_errors_accuracy(self):
        self.test_model.load_state_dict(torch.load(self.model_dir+'best-model.pt')['model_state_dict'])
        min_loss=np.inf
        max_loss=0
        max_traj_ditc
        for i_batch,traj_dict in enumerate(self.val_metric_loader):
            pred_traj=self.model(traj_dict)
            pred_traj=self.val_metric_loader.dataset.inverse_transform(pred_traj,traj_dict)
            gt_traj=traj_dict['gt_unnorm_agent']
            if self.use_cuda:
                gt_traj=gt_traj.cuda()
            loss=self.loss_fn(pred_traj,gt_traj)
            if loss<min_loss:
                min_traj_dict=traj_dict
                min_loss=loss
            if loss>max_loss:
                max_traj_dict=traj_dict
                max_loss=loss
        
        input_=self.val_metric_loader.dataset.inverse_transform(min_traj_dict['train_agent'],min_traj_dict)
        output=self.val_metric_loader.dataset.inverse_transform(self.model(min_traj_dict),min_traj_dict)
        target=min_traj_dict['gt_unnorm_agent']
        avm=ArgoverseMap()
        
        centerlines=avm.get_candidate_centerlines_for_traj(input_, current_loader.city,viz=False)
        if self.use_cuda:
            input_=self.val_metric_loader.dataset.inverse_transform(traj_dict['train_agent'],traj_dict).cpu()
        viz_predictions(input_=, output= pred_traj.unsqueeze(), target=gt_traj ,centerlines:,
                        city_names: np.ndarray,idx=None, show: bool = True,
            

    
    def train(self):
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch}: ")
            avg_loss_train=self.train_epoch()
            avg_loss_val,ade_one_sec,fde_one_sec,ade_three_sec,fde_three_sec = self.val_epoch(epoch)
            if (epoch+1==self.num_epochs):
                self.test_epoch()
                self.best_model_updated=False
            # self.writer.scalar_summary('Val/1ADE_Epoch', ade_one_sec, epoch)
            # self.writer.scalar_summary('Val/3ADE_Epoch', ade_three_sec, epoch)
            # self.writer.scalar_summary('Val/1FDE_Epoch', fde_one_sec, epoch)
            # self.writer.scalar_summary('Val/3FDE_Epoch', fde_three_sec, epoch)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='Sequence Modeling - Argoverse Forecasting Task')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA (default: True)')
    # parser.add_argument('--social',action='store_true',help='use neighbour data as well. default: False')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for optimizer (default: 0.001)')
    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip, -1 means no clip (default: -1)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='upper epoch limit (default: 10)')
    parser.add_argument('--ksize', type=int, default=7,
                        help='kernel size (default: 7)')
    parser.add_argument('--levels', type=int, default=10,
                        help='# of levels (default: 8)')
    parser.add_argument('--nhid', type=int, default=20,
                        help='number of hidden units per layer (default: 20)')
    parser.add_argument('--opsize', type=int, default=30,
                        help='number of output units for model (default: 30)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default: 1111)')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='model type to execute (default: LSTM Baseline)')
    parser.add_argument('--data', type=str, default='XY',
                        help='type of data to use for training (default: XY, options: XY,LaneCentre,')
    parser.add_argument('--train-log-interval', type=int, default=100,
                        help='number of intervals after which to print train stats (default: 100)')
    parser.add_argument('--val-log-interval', type=int, default=500,
                        help='number of intervals after which to print val stats (default: 500)')
    parser.add_argument('--test-log-interval', type=int, default=500,
                        help='number of intervals after which to print test stats (default: 500)')
    
    args = parser.parse_args()

    curr_time = strftime("%Y%m%d%H%M%S", localtime())

    # initialize model and params
    if args.model == 'LSTM':
        logger_dir = './runs/' + args.model + '/' + curr_time + '/'
        model_dir = './models/' + args.model + '/' + curr_time + '/'
        model = LSTMModel(cuda=args.cuda)
    elif args.model == 'TCN':
        logger_dir = './runs/' + args.model + '/' + curr_time + '/'
        model_dir = './models/' + args.model + '/' + curr_time + '/'
        channel_sizes = [args.nhid] * args.levels
        model = TCNModel(args.nhid, args.opsize, channel_sizes, args.ksize, args.dropout, 128)
    elif args.model == 'SOCIAL':
        logger_dir = './runs/' + args.model + '/' + curr_time + '/'
        model_dir = './models/' + args.model + '/' + curr_time + '/'
        model = Social_Model(cuda=args.cuda)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    test_dir='./test/'+args.model +'/' + curr_time + '/'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # tbLogger = TensorLogger(_logdir=logger_dir)
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
    # for centerlines in argoverse_train:
    #     if centerlines is None:
    #         count+=1
    #     total+=1
    #     print(f"{count}/{total} centerlines are None",end="\r")

    _parallel = False
    trainer=Trainer(model=model,use_cuda=args.cuda,parallel=_parallel,optimizer=optimizer,\
        train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,loss_fn=loss_fn,\
            num_epochs=args.epochs,writer=None,args=args,modeldir=model_dir,testdir=test_dir)
    trainer.train()

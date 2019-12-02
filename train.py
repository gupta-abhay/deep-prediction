from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
from statistics import mean
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn

from data import Argoverse_Data,collate_traj
from model import LSTMModel, TCNModel, Social_Model
from argoverse.evaluation.eval_forecasting import get_ade, get_fde
import matplotlib.pyplot as plt
import argparse
import warnings
from time import localtime, strftime
from logger import TensorLogger
import numpy as np
import os

class Trainer():
    def __init__(self,model,cuda,parallel,optimizer,train_loader,val_loader,test_loader,loss_fn,num_epochs,writer,args,modeldir):
        self.model=model
        self.cuda=cuda
        self.parallel = parallel
        if self.cuda:
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


    def train_epoch(self):
        total_loss=0
        num_batches=len(self.train_loader.batch_sampler)
        batch_size=self.train_loader.batch_size
        self.model.train()
        no_samples=0

        for i_batch,traj_dict in enumerate(self.train_loader):
            input_traj=traj_dict['train_agent']
            gr_traj=traj_dict['gt_agent']
            if self.args.social:
                neighbour_traj=traj_dict['neighbour']
            if self.cuda:
                input_traj=input_traj.cuda()
                gr_traj=gr_traj.cuda()
            if self.args.social and self.cuda:
                neighbour_traj=[neighbour.cuda() for neighbour in neighbour_traj]            
            if self.args.social:
                pred_traj=self.model({'agent_traj':input_traj,'neighbour_traj':neighbour_traj})
            else:   
                pred_traj=self.model(input_traj)
            
            loss=self.loss_fn(pred_traj,gr_traj)
            total_loss=total_loss+loss.data
            batch_samples=input_traj.shape[0]
            no_samples+=batch_samples
            avg_loss = float(total_loss)/(i_batch+1)
            self.writer.scalar_summary('Train/AvgLoss', avg_loss, i_batch+1)
            if (i_batch+1) % self.args.train_log_interval == 0:
                print(f"Training Iter {i_batch+1}/{num_batches} Avg Loss {total_loss/(i_batch+1)}",end="\r")
                # print(f"Training Iter {i_batch+1}/{num_batches} Avg Loss {total_loss/(i_batch+1)}"
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print()
        return total_loss/(num_batches)
    
    
    def val_epoch(self, epoch):
        total_loss=0
        num_batches=len(self.val_loader.batch_sampler)
        batch_size=self.val_loader.batch_size
        self.model.eval()
        
        ade_one_sec,fde_one_sec,ade_three_sec,fde_three_sec=(0,0,0,0)
        ade_one_sec_avg, fde_one_sec_avg ,ade_three_sec_avg, fde_three_sec_avg = (0,0,0,0)
        no_samples=0
        
        for i_batch,traj_dict in enumerate(self.val_loader):
            input_traj=traj_dict['train_agent']
            gr_traj=traj_dict['gt_agent']
            if self.args.social:
                neighbour_traj=traj_dict['neighbour']
            if self.cuda:
                input_traj=input_traj.cuda()
                gr_traj=gr_traj.cuda()
            if self.args.social and self.cuda:
                neighbour_traj=[neighbour.cuda() for neighbour in neighbour_traj]
            if self.args.social:
                pred_traj=self.model({'agent_traj':input_traj,'neighbour_traj':neighbour_traj})
            else:   
                pred_traj=self.model(input_traj)
            loss=self.loss_fn(pred_traj,gr_traj)
            total_loss=total_loss+loss.data
            batch_samples=input_traj.shape[0]
            no_samples+=batch_samples
            ade_one_sec+=sum([get_ade(pred_traj[i,:10,:],gr_traj[i,:10,:]) for i in range(batch_samples)])
            fde_one_sec+=sum([get_fde(pred_traj[i,:10,:],gr_traj[i,:10,:]) for i in range(batch_samples)])
            ade_three_sec+=sum([get_ade(pred_traj[i,:,:],gr_traj[i,:,:]) for i in range(batch_samples)])
            fde_three_sec+=sum([get_fde(pred_traj[i,:,:],gr_traj[i,:,:]) for i in range(batch_samples)])

            ade_one_sec_avg = float(ade_one_sec)/no_samples
            ade_three_sec_avg = float(ade_three_sec)/no_samples
            fde_one_sec_avg = float(fde_one_sec)/no_samples
            fde_three_sec_avg = float(fde_three_sec)/no_samples

            self.writer.scalar_summary('Val/AvgLoss', float(total_loss)/(i_batch+1), i_batch+1)

            if (i_batch+1) % self.args.val_log_interval == 0:
                print(f"Validation Iter {i_batch+1}/{num_batches} Avg Loss {total_loss/(i_batch+1):.4f} \
                One sec:- ADE:{ade_one_sec/(no_samples):.4f} FDE: {fde_one_sec/(no_samples):.4f}\
                Three sec:- ADE:{ade_three_sec/(no_samples):.4f} FDE: {fde_three_sec/(no_samples):.4f}",end="\r")

            _filename = self.model_dir + 'best-model.pt'

            if ade_one_sec_avg < self.best_1_ade and ade_three_sec_avg < self.best_3_ade and fde_one_sec_avg < self.best_1_fde and fde_three_sec_avg < self.best_3_fde:
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
        print()
        return total_loss/(num_batches), ade_one_sec/no_samples,fde_one_sec/no_samples,ade_three_sec/no_samples,fde_three_sec/no_samples
    

    def test_epoch(self):
        num_batches=len(self.test_loader.batch_sampler)
        batch_size=self.test_loader.batch_size
        self.model.eval()
        no_samples=0
        ade_one_sec,fde_one_sec,ade_three_sec,fde_three_sec=(0,0,0,0)

        for i_batch,traj_dict in enumerate(self.test_loader):
            input_traj=traj_dict['train_agent']
            gr_traj=traj_dict['gt_agent']
            if self.args.social:
                neighbour_traj=traj_dict['neighbour']
            if self.cuda:
                input_traj=input_traj.cuda()
                gr_traj=gr_traj.cuda()
            
            if self.args.social:
                neighbour_traj=traj_dict['neighbour']
                pred_traj=self.model({'agent_traj':input_traj,'neighbour_traj':neighbour_traj})
            else:   
                pred_traj=self.model(input_traj)
            batch_samples=input_traj.shape[0]
            no_samples+=batch_samples

            ade_one_sec+=sum([get_ade(pred_traj[i,:10,:],gr_traj[i,:10,:]) for i in range(batch_samples)])
            fde_one_sec+=sum([get_fde(pred_traj[i,:10,:],gr_traj[i,:10,:]) for i in range(batch_samples)])
            ade_three_sec+=sum([get_ade(pred_traj[i,:,:],gr_traj[i,:,:]) for i in range(batch_samples)])
            fde_three_sec+=sum([get_fde(pred_traj[i,:,:],gr_traj[i,:,:]) for i in range(batch_samples)])

            if (i_batch+1) % self.args.test_log_interval == 0:
                print(f"Test Iter {i_batch+1}/{num_batches} \
                    One sec:- ADE:{ade_one_sec/(no_samples):.4f} FDE: {fde_one_sec/(no_samples):.4f}\
                    Three sec:- ADE:{ade_three_sec/(no_samples):.4f} FDE: {fde_three_sec/(no_samples):.4f}",end="\r")

            ade_one_sec_avg = float(ade_one_sec)/no_samples
            ade_three_sec_avg = float(ade_three_sec)/no_samples
            fde_one_sec_avg = float(fde_one_sec)/no_samples
            fde_three_sec_avg = float(fde_three_sec)/no_samples

            self.writer.scalar_summary('Test/1ADE', ade_one_sec_avg, i_batch+1)
            self.writer.scalar_summary('Test/3ADE', ade_three_sec_avg, i_batch+1)
            self.writer.scalar_summary('Test/1FDE', fde_one_sec_avg, i_batch+1)
            self.writer.scalar_summary('Test/3FDE', fde_three_sec_avg, i_batch+1)
        print()
        return ade_one_sec/no_samples,fde_one_sec/no_samples,ade_three_sec/no_samples,fde_three_sec/no_samples
    
    
    def train(self):
        for epoch in range(self.num_epochs):
            print ("Starting epoch {}/{}".format(epoch+1, self.num_epochs))
            avg_loss_train = self.train_epoch()
            print ("Final Training Loss {}".format(avg_loss_train))

            avg_loss_val,ade_one_sec,fde_one_sec,ade_three_sec,fde_three_sec = self.val_epoch(epoch)
            print ("Validation Results for epoch {}/{}:    1sec. ADE/FDE: {}/{}    3sec. ADE/FDE: {}/{}".format(epoch+1, self.num_epochs, ade_one_sec, fde_one_sec, ade_three_sec, fde_three_sec))
            
            # Tensorboard summaries
            self.writer.scalar_summary('Val/1ADE_Epoch', ade_one_sec, epoch)
            self.writer.scalar_summary('Val/3ADE_Epoch', ade_three_sec, epoch)
            self.writer.scalar_summary('Val/1FDE_Epoch', fde_one_sec, epoch)
            self.writer.scalar_summary('Val/3FDE_Epoch', fde_three_sec, epoch)

            print ("Finishing epoch {}/{}".format(epoch+1, self.num_epochs))


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='Sequence Modeling - Argoverse Forecasting Task')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA (default: True)')
    parser.add_argument('--social',action='store_true',help='use neighbour data as well. default: False')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for optimizer (default: 0.001)')
    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip, -1 means no clip (default: -1)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit (default: 10)')
    parser.add_argument('--ksize', type=int, default=3,
                        help='kernel size (default: 3)')
    parser.add_argument('--levels', type=int, default=20,
                        help='# of levels (default: )')
    parser.add_argument('--nhid', type=int, default=20,
                        help='number of hidden units per layer (default: 20)')
    parser.add_argument('--opsize', type=int, default=30,
                        help='number of output units for model (default: 30)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default: 1111)')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='model type to execute (default: LSTM Baseline)')
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
        model = LSTMModel()
    elif args.model == 'TCN':
        logger_dir = './runs/' + args.model + '/' + curr_time + '/'
        model_dir = './models/' + args.model + '/' + curr_time + '/'
        channel_sizes = [args.nhid] * args.levels
        model = TCNModel(args.nhid, args.opsize, channel_sizes, args.ksize, args.dropout, 128)
    elif args.model == 'SOCIAL':
        logger_dir = './runs/' + args.model + '/' + curr_time + '/'
        model_dir = './models/' + args.model + '/' + curr_time + '/'
        model = Social_Model()
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    tbLogger = TensorLogger(_logdir=logger_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    print("CUDA is ",args.cuda)
    print("Model is ",args.model)
    print("Social is", args.social)
    # Load data module
    argoverse_train=Argoverse_Data('data/train/data/',social=args.social)
    argoverse_val=Argoverse_Data('data/val/data',social=args.social)
    argoverse_test = Argoverse_Data('data/test_obs/data',social=args.social)
    if args.social:
        train_loader = DataLoader(argoverse_train, batch_size=args.batch_size,
                        shuffle=True, num_workers=2,collate_fn=collate_traj)
        val_loader = DataLoader(argoverse_val, batch_size=args.batch_size,
                        shuffle=True, num_workers=2,collate_fn=collate_traj)
        test_loader = DataLoader(argoverse_test, batch_size=args.batch_size,
                        shuffle=True, num_workers=2,collate_fn=collate_traj)
    else:
        train_loader = DataLoader(argoverse_train, batch_size=args.batch_size,
                            shuffle=True, num_workers=2)
        val_loader = DataLoader(argoverse_val, batch_size=args.batch_size,
                            shuffle=True, num_workers=2)
        test_loader = DataLoader(argoverse_test, batch_size=args.batch_size,
                            shuffle=True, num_workers=2)

    # train model and losses
    loss_fn=nn.MSELoss()
    # _cuda = a
    
    _parallel = False
    trainer=Trainer(model=model,cuda=args.cuda,parallel=_parallel,optimizer=optimizer,train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,loss_fn=loss_fn,num_epochs=args.epochs,writer=tbLogger,args=args,modeldir=model_dir)
    trainer.train()

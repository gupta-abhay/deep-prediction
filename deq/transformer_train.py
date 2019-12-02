import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.evaluation.eval_forecasting import get_ade, get_fde

from statistics import mean
import glob
import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import warnings
from time import localtime, strftime
import numpy as np
import itertools
import os, sys
import threading
import copy

import pdb
import math

from visualize import viz_predictions
from data import Argoverse_Data
from seq_models.transformers.deq_transformer import DEQTransformerLM
from modules import radam
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass


class Trainer():
    def __init__(self,model,use_cuda,optimizer,train_loader,\
        val_loader,loss_fn,num_epochs,writer,args,modeldir):
        self.model = model
        self.use_cuda = use_cuda
        
        self.optimizer=optimizer
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.loss_fn=loss_fn
        self.num_epochs=num_epochs

        # self.params = params
        
        self.best_1_ade = np.inf
        self.best_1_fde = np.inf
        self.best_3_ade = np.inf
        self.best_3_fde = np.inf

        self.writer = writer
        self.args = args
        self.model_dir = modeldir

        self.train_step = args.start_train_steps

    def check_normalization(self):
        for i_batch,traj_dict in enumerate(self.val_loader):
            pred_traj=traj_dict['gt_agent']
            pred_traj=self.train_loader.dataset.inverse_transform(pred_traj,traj_dict)
            gt_traj=traj_dict['gt_unnorm_agent']
            if self.use_cuda:
                pred_traj=pred_traj.cuda()
                gt_traj=gt_traj.cuda()
            loss=self.loss_fn(pred_traj,gt_traj)
            print(f"Batch: {i_batch}, Loss: {loss.item()}")
    
    def train_epoch(self):
        self.model.train()
        subseq_len = args.subseq_len
        train_loss = 0
        mems = []
        num_batches=len(self.train_loader.batch_sampler)

        for i_batch, traj_dict in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if mems:
                mems[0] = mems[0].detach()

            data = traj_dict['train_agent']
            target = traj_dict['gt_agent']

            if self.use_cuda:
                data = data.cuda()
                target = target.cuda()
            
            pred_traj, mems = self.model(data, target, mems, train_step=self.train_step, f_thres=args.f_thres,
                                    b_thres=args.b_thres, subseq_len=subseq_len)

            loss = self.loss_fn(pred_traj, target)
            loss.backward()
            train_loss += loss.item()

            avg_loss = float(train_loss) / (i_batch+1)
            print(f"Training Iter {i_batch+1}/{num_batches} Avg Loss {avg_loss:.4f}", end="\r")

            torch.nn.utils.clip_grad_norm(self.model.parameters(), args.clip)
            self.optimizer.step()
            self.train_step += 1

            # Step-wise learning rate annealing according to some scheduling (we ignore 'constant' scheduling)
            if args.scheduler in ['cosine', 'constant', 'dev_perf']:
                # linear warmup stage
                if self.train_step < args.warmup_step:
                    curr_lr = args.lr * self.train_step / args.warmup_step
                    optimizer.param_groups[0]['lr'] = curr_lr
                else:
                    if args.scheduler == 'cosine':
                        scheduler.step(self.train_step)
            elif args.scheduler == 'inv_sqrt':
                scheduler.step(self.train_step)

        print()
        return train_loss/num_batches

    def val_epoch(self, epoch):
        subseq_len = args.subseq_len
        val_loss = 0
        mems = []
        num_batches=len(self.train_loader.batch_sampler)
        
        ade_one_sec,fde_one_sec,ade_three_sec,fde_three_sec=(0,0,0,0)
        ade_one_sec_avg, fde_one_sec_avg ,ade_three_sec_avg, fde_three_sec_avg = (0,0,0,0)
        no_samples=0
        
        for i_batch,traj_dict in enumerate(self.val_loader):
            if mems:
                mems[0] = mems[0].detach()
            
            data = traj_dict['train_agent']
            target = traj_dict['gt_agent']

            if self.use_cuda:
                data = data.cuda()
                target = target.cuda()
            
            pred_traj, mems = self.model(data, target, mems, train_step=self.train_step, f_thres=args.f_thres,
                                    b_thres=args.b_thres, subseq_len=subseq_len)

            loss = self.loss_fn(pred_traj, target)
            val_loss += loss.item()
            batch_samples = target.shape[0]
            
            ade_one_sec+=sum([get_ade(pred_traj[i,:10,:],target[i,:10,:]) for i in range(batch_samples)])
            fde_one_sec+=sum([get_fde(pred_traj[i,:10,:],target[i,:10,:]) for i in range(batch_samples)])
            ade_three_sec+=sum([get_ade(pred_traj[i,:,:],target[i,:,:]) for i in range(batch_samples)])
            fde_three_sec+=sum([get_fde(pred_traj[i,:,:],target[i,:,:]) for i in range(batch_samples)])
            
            no_samples+=batch_samples
            ade_one_sec_avg = float(ade_one_sec)/no_samples
            ade_three_sec_avg = float(ade_three_sec)/no_samples
            fde_one_sec_avg = float(fde_one_sec)/no_samples
            fde_three_sec_avg = float(fde_three_sec)/no_samples

            print(f"Validation Iter {i_batch+1}/{num_batches} Avg Loss {val_loss/(i_batch+1):.4f} \
            One sec:- ADE:{ade_one_sec/(no_samples):.4f} FDE: {fde_one_sec/(no_samples):.4f}\
            Three sec:- ADE:{ade_three_sec/(no_samples):.4f} FDE: {fde_three_sec/(no_samples):.4f}",end="\r")

            _filename = self.model_dir + 'best-model.pt'

            if ade_three_sec_avg < self.best_3_ade and fde_three_sec_avg < self.best_3_fde:    
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'loss': val_loss/(i_batch+1)
                }, _filename)

                self.best_1_ade = ade_one_sec_avg
                self.best_1_fde = fde_one_sec_avg
                self.best_3_ade = ade_three_sec_avg
                self.best_3_fde = fde_three_sec_avg
                self.best_model_updated=True
        
        print()
        return val_loss/(num_batches), ade_one_sec/no_samples,fde_one_sec/no_samples,ade_three_sec/no_samples,fde_three_sec/no_samples

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

    def save_top_errors_accuracy(self,model_dir, model_path):
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
        self.model.load_state_dict(torch.load(model_path+'best-model.pt')['model_state_dict'])
        self.model.eval()
        num_batches=len(self.val_loader.batch_sampler)

        for i_batch,traj_dict in enumerate(self.val_loader):
            print(f"Running {i_batch}/{num_batches}",end="\r")
            gt_traj=traj_dict['gt_unnorm_agent']
            train_traj=traj_dict['train_agent']
            if self.use_cuda:
                train_traj=train_traj.cuda()
            input_ = self.val_loader.dataset.inverse_transform(train_traj,traj_dict)
            output = self.model(traj_dict)
            output = self.val_loader.dataset.inverse_transform(output, traj_dict)
            
            if self.use_cuda:
                output=output.to('cpu')
                input_=input_.to('cpu')
            
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
                print(f"\nEpoch {epoch+1}/{self.num_epochs}: ")
                avg_loss_train=self.train_epoch()
                avg_loss_val,ade_one_sec,fde_one_sec,ade_three_sec,fde_three_sec = self.val_epoch(epoch)
        elif args.mode=="validate":
            self.validate_model(self.model_dir)
    
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv1d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('WeightShareSelfAttention') != -1:
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)

def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout

def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='PyTorch DEQ Sequence Model')
    parser.add_argument('--data', type=str, default='../data/wikitext-103',
                        help='location of the data corpus (default to the WT103 path)')
    parser.add_argument('--dataset', type=str, default='wt103',
                        choices=['wt103'],
                        help='dataset name')
    parser.add_argument('--n_layer', type=int, default=12,
                        help='number of total layers')
    parser.add_argument('--eval_n_layer', type=int, default=12,
                        help='number of total layers at evaluation')
    parser.add_argument('--n_head', type=int, default=10,
                        help='number of heads (default: 10)')
    parser.add_argument('--d_head', type=int, default=50,
                        help='head dimension (default: 50)')
    parser.add_argument('--d_embed', type=int, default=-1,
                        help='embedding dimension (default: match d_model)')
    parser.add_argument('--d_model', type=int, default=500,
                        help='model dimension (default: 500)')
    parser.add_argument('--d_inner', type=int, default=8000,
                        help='inner dimension in the position-wise feedforward block (default: 8000)')

    # Dropouts
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='global dropout rate (default: 0.05)')
    parser.add_argument('--dropatt', type=float, default=0.0,
                        help='attention map dropout rate (default: 0.0)')

    # Initializations
    # Note: Generally, to make sure the DEQ model is stable initially, we should constrain the range
    #       of initialization.
    parser.add_argument('--init', default='normal', type=str,
                        help='parameter initializer to use.')
    parser.add_argument('--emb_init', default='normal', type=str,
                        help='parameter initializer to use.')
    parser.add_argument('--init_range', type=float, default=0.05,
                        help='parameters initialized by U(-init_range, init_range)')
    parser.add_argument('--emb_init_range', type=float, default=0.01,
                        help='parameters initialized by U(-init_range, init_range)')
    parser.add_argument('--init_std', type=float, default=0.01,
                        help='parameters initialized by N(0, init_std)')
    parser.add_argument('--proj_init_std', type=float, default=0.01,
                        help='parameters initialized by N(0, init_std)')

    # Optimizers
    parser.add_argument('--optim', default='Adam', type=str,
                        choices=['Adam', 'SGD', 'Adagrad', 'RMSprop', 'RAdam'],
                        help='optimizer to use.')
    parser.add_argument('--lr', type=float, default=0.00025,
                        help='initial learning rate (0.00025|5 for adam|sgd)')
    parser.add_argument('--scheduler', default='cosine', type=str,
                        choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                        help='lr scheduler to use.')
    parser.add_argument('--warmup_step', type=int, default=0,
                        help='the number of steps to warm up the learning rate to its lr value')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay factor when ReduceLROnPlateau is used')
    parser.add_argument('--lr_min', type=float, default=0.0,
                        help='minimum learning rate during annealing')

    # Gradient updates
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--clip_nonemb', action='store_true',
                        help='only clip the gradient of non-embedding params')
    parser.add_argument('--max_step', type=int, default=200000,
                        help='upper epoch limit (at least 200K for WT103 or PTB)')
    parser.add_argument('--batch_size', type=int, default=60,
                        help='batch size')
    parser.add_argument('--batch_chunk', type=int, default=1,
                        help='split batch into chunks to save memory')

    # Sequence logistics
    parser.add_argument('--tgt_len', type=int, default=150,
                        help='number of tokens to predict')
    parser.add_argument('--eval_tgt_len', type=int, default=150,
                        help='number of tokens to predict for evaluation')
    parser.add_argument('--mem_len', type=int, default=150,
                        help='length of the retained previous heads')
    parser.add_argument('--subseq_len', type=int, default=100,
                        help='length of subsequence processed each time by DEQ')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='length of subsequence processed each time by DEQ')
    parser.add_argument('--local_size', type=int, default=0,
                        help='local horizon size')

    # Training techniques
    parser.add_argument('--not_tied', action='store_true',
                        help='do not tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--eval', action='store_true',
                        help='evaluation mode')
    parser.add_argument('--adaptive', action='store_true',
                        help='use adaptive softmax')
    parser.add_argument('--div_val', type=int, default=1,
                        help='divident value for adapative input and softmax')
    parser.add_argument('--pre_lnorm', action='store_true',
                        help='apply LayerNorm to the input instead of the output')
    parser.add_argument('--wnorm', action='store_true',
                        help='apply WeightNorm to the weights')
    parser.add_argument('--varlen', action='store_true',
                        help='use variable length')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='use multiple GPU')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='report interval')
    parser.add_argument('--eval-interval', type=int, default=4000,
                        help='evaluation interval')
    parser.add_argument('--f_thres', type=int, default=50,
                        help='forward pass Broyden threshold')
    parser.add_argument('--b_thres', type=int, default=80,
                        help='backward pass Broyden threshold')
    parser.add_argument('--work_dir', default='LM-TFM', type=str,
                        help='experiment directory.')
    parser.add_argument('--restart', action='store_true',
                        help='restart training from the saved checkpoint')
    parser.add_argument('--restart_dir', type=str, default='',
                        help='restart dir')
    parser.add_argument('--debug', action='store_true',
                        help='run in debug mode (do not create exp dir)')
    parser.add_argument('--same_length', action='store_true',
                        help='use the same attn length for all tokens')
    parser.add_argument('--attn_type', type=int, default=0,
                        help='attention type. 0 for ours, 1 for Shaw et al,'
                        '2 for Vaswani et al, 3 for Al Rfou et al. (Only 0 supported now)')
    parser.add_argument('--eta_min', type=float, default=0.0,
                        help='min learning rate for cosine scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--gpu0_bsz', type=int, default=-1,
                        help='batch size on gpu 0')
    parser.add_argument('--max_eval_steps', type=int, default=-1,
                        help='max eval steps')
    parser.add_argument('--pretrain_steps', type=int, default=0,
                        help='number of pretrain steps')
    parser.add_argument('--start_train_steps', type=int, default=0,
                        help='starting training step count (default to 0)')
    parser.add_argument('--patience', type=int, default=0,
                        help='patience')
    parser.add_argument('--load', type=str, default='',
                        help='path to load weight')
    parser.add_argument('--name', type=str, default='N/A',
                        help='name of the trial')
    parser.add_argument('--mode',type=str,default='train',help='mode: train, test ,validate')
    parser.add_argument('--model_dir',type=str,default='',help='path to saved model for validation')
    parser.add_argument('--epochs', type=int, default=25,
                        help='upper epoch limit (default: 25)')

        
    args = parser.parse_args()
    curr_time = strftime("%Y%m%d%H%M%S", localtime())
    if args.mode == 'train':
        logger_dir = './runs/transformers/' + curr_time + '/'
        model_dir = './models/transformers/' + curr_time + '/'
        os.makedirs(model_dir)
    else:
        logger_dir=None
        model_dir=args.model_dir

    args.tied = not args.not_tied
    args.pretrain_steps += args.start_train_steps
    assert args.seq_len > 0, "For now you must set seq_len > 0 when using deq"
    # args.work_dir += "deq"
    args.cuda = torch.cuda.is_available()
        
    if args.d_embed < 0:
        args.d_embed = args.nout

    assert args.batch_size % args.batch_chunk == 0

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    ntokens = 2
    if args.restart:
        with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
            model = torch.load(f)

        model = model.float()
        model.apply(update_dropout)
        model.apply(update_dropatt)
    else:
        model = DEQTransformerLM(n_token=ntokens, n_layer=args.n_layer,
                                eval_n_layer=args.eval_n_layer, n_head=args.n_head, d_model=args.d_model, d_head=args.d_model, d_inner=args.d_inner, dropout=args.dropout, dropatt=args.dropatt, mem_len=args.mem_len, tgt_len=args.tgt_len, tie_weights=True, d_embed=None)

        if len(args.load) == 0:
            model.apply(weights_init)
            model.word_emb.apply(weights_init)

    if args.multi_gpu:
        model = model.to(device)
        if args.gpu0_bsz >= 0:
            para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk, model, dim=1).to(device)   # Batch dim is dim 1
        else:
            para_model = nn.DataParallel(model, dim=1).to(device)
    else:
        para_model = model.to(device)

    loss_fn=nn.MSELoss()
    # params = list(model.parameters())
    lr = args.lr
    optimizer = getattr(optim if args.optim != 'RAdam' else radam, args.optim)(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_step, eta_min=args.eta_min)
    elif args.scheduler == 'inv_sqrt':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            if step == 0 and args.warmup_step == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > args.warmup_step else step / (args.warmup_step ** 1.5)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.scheduler == 'dev_perf':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)

    if args.restart:
        if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
            with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
                opt_state_dict = torch.load(f)
                optimizer.load_state_dict(opt_state_dict)
        else:
            print('Optimizer was not saved. Start from scratch.')


    print("CUDA is ",args.cuda)
    print("Model is TrellisNet-DEQ")
    print("Model dir is", model_dir)
    print(f"Training for {args.epochs} epochs")
    
    argoverse_train=Argoverse_Data('../data/train/data/',cuda=args.cuda)
    argoverse_val=Argoverse_Data('../data/val/data',cuda=args.cuda)
    train_loader = DataLoader(argoverse_train, batch_size=args.batch_size,
                        shuffle=True, num_workers=10, drop_last=True)
    val_loader = DataLoader(argoverse_val, batch_size=args.batch_size,
                        shuffle=True, num_workers=10, drop_last=True)

    print("Argoverse train dataloder is of size", len(train_loader.batch_sampler))
    print("Argoverse val dataloader is of size", len(val_loader.batch_sampler))

    _parallel = False
    trainer=Trainer(model=para_model,use_cuda=args.cuda,optimizer=optimizer,\
        train_loader=train_loader,val_loader=val_loader,loss_fn=loss_fn,\
            num_epochs=args.epochs,writer=None,args=args,modeldir=model_dir)
    trainer.run()
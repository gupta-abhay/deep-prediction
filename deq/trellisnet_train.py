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
import pickle

import pdb
import math

from visualize import viz_predictions
from data import Argoverse_Data
from seq_models.trellisnets.deq_trellisnet import DEQTrellisNetLM
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
        val_loader,loss_fn,num_epochs,writer,args,modeldir, params):
        self.model = model
        self.use_cuda = use_cuda
        
        self.optimizer=optimizer
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.loss_fn=loss_fn
        self.num_epochs=num_epochs

        self.params = params
        
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
            print(f"Batch: {i_batch}, Loss: {loss.data}")

    
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
            
            (_, _, pred_traj), mems = self.model(data, target, mems, train_step=self.train_step, f_thres=args.f_thres,
                                    b_thres=args.b_thres, subseq_len=subseq_len, decode=True)

            loss = self.loss_fn(pred_traj, target)
            loss.backward()
            train_loss += loss.item()

            avg_loss = float(train_loss) / (i_batch+1)
            print(f"Training Iter {i_batch+1}/{num_batches} Avg Loss {avg_loss:.4f}",end="\r")

            torch.nn.utils.clip_grad_norm(self.params, args.clip)
            self.optimizer.step()
            self.train_step += 1

        print()
        return train_loss/num_batches
        

    def val_epoch(self, epoch):
        subseq_len = args.subseq_len
        val_loss = 0
        mems = []
        num_batches=len(self.val_loader.batch_sampler)
        
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
            
            (_, _, pred_traj), mems = self.model(data, target, mems, train_step=self.train_step, f_thres=args.f_thres,
                                    b_thres=args.b_thres, subseq_len=subseq_len, decode=True)

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
            Three sec:- ADE:{ade_three_sec/(no_samples):.4f} FDE: {fde_three_sec/(no_samples):.4f}", end="\r")

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
        # subseq_len = args.subseq_len
        # val_loss = 0
        # mems = []
        # num_batches=len(self.val_loader.batch_sampler)

        # self.model.load_state_dict(torch.load(model_path+'trellis-model.pt')['model_state_dict'])
        # self.model.eval()
        
        # ade_one_sec,fde_one_sec,ade_three_sec,fde_three_sec=(0,0,0,0)
        # ade_one_sec_avg, fde_one_sec_avg ,ade_three_sec_avg, fde_three_sec_avg = (0,0,0,0)
        # no_samples=0
        
        # for i_batch,traj_dict in enumerate(self.val_loader):
        #     if mems:
        #         mems[0] = mems[0].detach()
            
        #     data = traj_dict['train_agent']
        #     target = traj_dict['gt_agent']

        #     if self.use_cuda:
        #         data = data.cuda()
        #         target = target.cuda()
            
        #     (_, _, pred_traj), mems = self.model(data, target, mems, train_step=self.train_step, f_thres=args.f_thres,
        #                             b_thres=args.b_thres, subseq_len=subseq_len, decode=True)

        #     loss = self.loss_fn(pred_traj, target)
        #     val_loss += loss.item()
        #     batch_samples = target.shape[0]
            
        #     ade_one_sec+=sum([get_ade(pred_traj[i,:10,:],target[i,:10,:]) for i in range(batch_samples)])
        #     fde_one_sec+=sum([get_fde(pred_traj[i,:10,:],target[i,:10,:]) for i in range(batch_samples)])
        #     ade_three_sec+=sum([get_ade(pred_traj[i,:,:],target[i,:,:]) for i in range(batch_samples)])
        #     fde_three_sec+=sum([get_fde(pred_traj[i,:,:],target[i,:,:]) for i in range(batch_samples)])
            
        #     no_samples+=batch_samples
        #     ade_one_sec_avg = float(ade_one_sec)/no_samples
        #     ade_three_sec_avg = float(ade_three_sec)/no_samples
        #     fde_one_sec_avg = float(fde_one_sec)/no_samples
        #     fde_three_sec_avg = float(fde_three_sec)/no_samples

        #     print(f"Validation Iter {i_batch+1}/{num_batches} Avg Loss {val_loss/(i_batch+1):.4f} \
        #     One sec:- ADE:{ade_one_sec/(no_samples):.4f} FDE: {fde_one_sec/(no_samples):.4f}\
        #     Three sec:- ADE:{ade_three_sec/(no_samples):.4f} FDE: {fde_three_sec/(no_samples):.4f}", end="\r")

        # print()
        # self.save_top_errors_accuracy(self.model_dir, model_path)
        # print("Saved error plots")

        self.save_results_single_pred()

 
    def save_results_single_pred(self):
        subseq_len = args.subseq_len
        print("running save results")
        afl=ArgoverseForecastingLoader("../data/val/data/")
        checkpoint = torch.load(self.model_dir+'trellis-model.pt', map_location=lambda storage, loc: storage)
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

            if mems:
                mems[0] = mems[0].detach()
            
            data = traj_dict['train_agent']
            target = traj_dict['gt_agent']

            if self.use_cuda:
                data = data.cuda()
                target = target.cuda()
            
            (_, _, output), mems = self.model(data, target, mems, train_step=self.train_step, f_thres=args.f_thres,
                                    b_thres=args.b_thres, subseq_len=subseq_len, decode=True)


            # output=self.model(traj_dict,mode='validate')
            # output=self.model(traj_dict)
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

            # input_tensor=np.array(input_tensor)

    def save_top_errors_accuracy(self,model_dir, model_path):
        subseq_len = args.subseq_len
        val_loss = 0
        mems = []
        num_batches=len(self.val_loader.batch_sampler)

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

        # self.model.load_state_dict(torch.load(model_path+'trellis-model.pt')['model_state_dict'])
        # self.model.eval()

        checkpoint = torch.load(model_path+'trellis-model.pt', map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        for i_batch,traj_dict in enumerate(self.val_loader):
            print(f"Running {i_batch}/{num_batches}",end="\r")

            if mems:
                mems[0] = mems[0].detach()

            gt_traj = traj_dict['gt_unnorm_agent']
            target = traj_dict['gt_agent']
            train_traj = traj_dict['train_agent']

            if self.use_cuda:
                train_traj=train_traj.cuda()
                target = target.cuda()
            
            input_ = self.val_loader.dataset.inverse_transform(train_traj,traj_dict)
            (_, _, output), mems = self.model(train_traj, target, mems, train_step=self.train_step, f_thres=args.f_thres,
                                    b_thres=args.b_thres, subseq_len=subseq_len, decode=True)
            output = self.val_loader.dataset.inverse_transform(output, traj_dict)
            
            # if self.use_cuda:
            output=output.cpu()
            input_=input_.cpu()
            target = target.cpu()
            
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
    

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='PyTorch DEQ Sequence Model')
    parser.add_argument('--n_layer', type=int, default=30,
                        help='number of total layers')
    parser.add_argument('--d_embed', type=int, default=128,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nout', type=int, default=128,
                        help='number of output units')
    parser.add_argument('--epochs', type=int, default=25,
                        help='upper epoch limit (default: 25)')

    # Optimizers
    parser.add_argument('--optim', default='Adam', type=str,
                        choices=['Adam', 'SGD', 'Adagrad', 'RMSprop', 'RAdam'],
                        help='optimizer to use.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate (default: 1e-3)')

    # Gradient updates
    parser.add_argument('--clip', type=float, default=0.07,
                        help='gradient clipping (default: 0.07)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--batch_chunk', type=int, default=1,
                        help='split batch into chunks to save memory')

    # Sequence logistics
    parser.add_argument('--seq_len', type=int, default=100,
                        help='total sequence length')
    parser.add_argument('--subseq_len', type=int, default=75,
                        help='length of subsequence processed each time by DEQ')

    # Regularizations
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='output dropout (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.1,
                        help='input dropout (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.0,
                        help='dropout applied to weights (0 = no dropout)')
    parser.add_argument('--emb_dropout', type=float, default=0.0,
                        help='dropout applied to embedding layer (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.1,
                        help='dropout applied to hidden layers (0 = no dropout)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--wnorm', action='store_false',
                        help='use weight normalization (default: True)')

    # Training techniques
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--not_tied', action='store_true',
                        help='do not tie the word embedding and softmax weights (default: False)')
    parser.add_argument('--anneal', type=int, default=5,
                        help='learning rate annealing criteria (default: 5)')
    parser.add_argument('--when', nargs='+', type=int, default=[15, 20, 23],
                        help='When to decay the learning rate')
    parser.add_argument('--ksize', type=int, default=2,
                        help='conv kernel size (default: 2)')
    parser.add_argument('--dilation', type=int, default=1,
                        help='dilation rate (default: 1)')
    parser.add_argument('--n_experts', type=int, default=0,
                        help='number of softmax experts (default: 0)')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='use multiple GPU')
    parser.add_argument('--f_thres', type=int, default=50,
                        help='forward pass Broyden threshold')
    parser.add_argument('--b_thres', type=int, default=80,
                        help='backward pass Broyden threshold')
    parser.add_argument('--restart', action='store_true',
                        help='restart training from the saved checkpoint')
    parser.add_argument('--restart_dir', type=str, default='',
                        help='restart dir')
    parser.add_argument('--debug', action='store_true',
                        help='run in debug mode (do not create exp dir)')
    parser.add_argument('--gpu0_bsz', type=int, default=-1,
                        help='batch size on gpu 0')
    parser.add_argument('--pretrain_steps', type=int, default=0,
                        help='number of pretrain steps')
    parser.add_argument('--start_train_steps', type=int, default=0,
                        help='starting training step count (default to 0)')
    parser.add_argument('--eval', action='store_true',
                        help='evaluation mode')
    parser.add_argument('--load', type=str, default='',
                        help='path to load weight')
    parser.add_argument('--mode',type=str,default='train',help='mode: train, test ,validate')
    parser.add_argument('--model_dir',type=str,default='',help='path to saved model for validation')

        
    args = parser.parse_args()
    curr_time = strftime("%Y%m%d%H%M%S", localtime())
    if args.mode == 'train':
        logger_dir = './runs/trellisnet/' + curr_time + '/'
        model_dir = './models/trellisnet/' + curr_time + '/'
        os.makedirs(model_dir)
    else:
        logger_dir=None
        model_dir=args.model_dir

    args.tied = not args.not_tied
    args.pretrain_steps += args.start_train_steps
    assert args.seq_len > 0, "For now you must set seq_len > 0 when using deq"
    # args.work_dir += "deq"
    # args.cuda = torch.cuda.is_available()
        
    if args.d_embed < 0:
        args.d_embed = args.nout

    assert args.batch_size % args.batch_chunk == 0

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # if torch.cuda.is_available():
    #     if not args.cuda:
    #         print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    #     else:
    #         torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #         torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    ntokens = 2
    model = DEQTrellisNetLM(n_token=ntokens, n_layer=args.n_layer, ninp=args.d_embed, nhid=args.nhid, nout=args.nout, 
                        kernel_size=args.ksize, emb_dropout=args.emb_dropout, dropouti=args.dropouti, dropout=args.dropout, 
                        dropouth=args.dropouth, wdrop=args.wdrop, wnorm=args.wnorm, tie_weights=args.tied, 
                        pretrain_steps=args.pretrain_steps, dilation=args.dilation, load=args.load)

    if args.multi_gpu:
        print ("in here 1")
        model = model.to(device)
        if args.gpu0_bsz >= 0:
            para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk, model, dim=1).to(device)   # Batch dim is dim 1
        else:
            para_model = nn.DataParallel(model, dim=1).to(device)
    else:
        print ("in here 2")
        para_model = model.to(device)

    loss_fn=nn.MSELoss()
    params = list(model.parameters())
    lr = args.lr
    optimizer = getattr(optim if args.optim != 'RAdam' else radam, args.optim)(params, lr=lr, weight_decay=args.weight_decay)
    
    print("CUDA is ", args.cuda)
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
            num_epochs=args.epochs,writer=None,args=args,modeldir=model_dir, params = params)

    trainer.run()
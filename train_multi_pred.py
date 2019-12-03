"""Multilane and multi prediction experiments.
Exp 1: 1 predictions with one centerline.
Exp 2: 2 predictions with one centerline. (Expectation Maximization should be used here.)
Exp 3: Use image with predictions in centerline frame with one prediction per centerline. (train with prediction error) 
Exp 4: Use image with predictions in centerline frame with multiple prediction per centerline. (train with expectation maximization)
(this should work the best).

Design: dataset 1 will give one centerline and one trajectory for one train frame. while test it will provide candidate centerlines.
Dataset 2 will give one centerline with one trajectory and one image. while test multiple candidate trajectory with images.

Model Exp1: train with lstm with the centerline and social features. Loss is prediction loss with lstm.
Exp 2: train the model with expectation maximization. 
Exp 3: train with prediction error
Exp 4: train with expectation maximization 
"""
from argoverse.map_representation.map_api import ArgoverseMap
from data_new import collate_traj_multilane,collate_traj_xy,collate_traj_social_centerline,Argoverse_MultiLane_Data,Argoverse_Social_Data,Argoverse_Social_Centerline_Data
from model_new import LSTMModel,LSTMModel_CenterlineEmbed,Social_Model,Social_Model_Centerline
from torch.utils.data import Dataset, DataLoader
from argoverse.evaluation.eval_forecasting import get_ade, get_fde
from argoverse.evaluation.competition_util import generate_forecasting_h5
import glob,warnings
from argoverse.visualization.visualize_sequences import viz_sequence
from visualize import viz_predictions
import torch
import pandas as pd
import os
import numpy as np
import pdb
import torch.nn as nn
import argparse
from time import localtime, strftime
class Train():
    def __init__(self,model,optimizer,train_loader,val_loader,test_loader,loss_fn,model_dir,pretrained_dir=None):
        self.model=model
        if pretrained_dir!=None:
            self.model.load_state_dict(torch.load(pretrained_dir+'best-model.pt')['model_state_dict'])
        self.optimizer=optimizer
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.test_loader=test_loader
        self.model_dir=model_dir
        self.loss_fn=loss_fn

        self.best_1_ade = np.inf
        self.best_1_fde = np.inf
        self.best_3_ade = np.inf
        self.best_3_fde = np.inf
    def train_epoch(self):
        total_loss=0
        self.model.train()
        num_batches=len(self.train_loader.batch_sampler)
        batch_size=self.train_loader.batch_size
        eliminated=0
        num_samples=0
        for i_batch,traj_dict in enumerate(self.train_loader):
            pred_traj=self.model(traj_dict,mode='train')
            # pdb.set_trace()
            loss=self.loss_fn(pred_traj,traj_dict['gt_traj'].cuda())
            num_samples+=pred_traj.shape[0]
            total_loss=total_loss+(loss.data*pred_traj.shape[0])
            avg_loss = float(total_loss)/(num_samples)
            eliminated+=batch_size-pred_traj.shape[0]
            print(f"Training Iter {i_batch+1}/{num_batches} Avg Loss {avg_loss:.4f} Batch Loss {loss.data:.4f} Eliminated : {eliminated}/{num_samples}",end="\r")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print()
    def val_epoch(self,epoch):
        total_loss=0
        num_batches=len(self.val_loader.batch_sampler)
        self.model.eval() 
        ade_one_sec,fde_one_sec,ade_three_sec,fde_three_sec=(0,0,0,0)
        ade_one_sec_avg, fde_one_sec_avg ,ade_three_sec_avg, fde_three_sec_avg = (0,0,0,0)
        no_samples=0
        for i_batch,traj_dict in enumerate(self.val_loader):
            pred_traj=self.model(traj_dict,mode='validate')
            gt_traj=traj_dict['gt_unnorm_traj'].cuda()
            loss=self.loss_fn(pred_traj,gt_traj)
            batch_samples=pred_traj.shape[0]
            no_samples+=batch_samples
            total_loss=total_loss+(loss.data*batch_samples)
            avg_loss = float(total_loss)/(no_samples)
            # pdb.set_trace()
            ade_one_sec+=sum([get_ade(pred_traj[i,:10,:],gt_traj[i,:10,:]) for i in range(batch_samples)])
            fde_one_sec+=sum([get_fde(pred_traj[i,:10,:],gt_traj[i,:10,:]) for i in range(batch_samples)])
            ade_three_sec+=sum([get_ade(pred_traj[i,:,:],gt_traj[i,:,:]) for i in range(batch_samples)])
            fde_three_sec+=sum([get_fde(pred_traj[i,:,:],gt_traj[i,:,:]) for i in range(batch_samples)])

            ade_one_sec_avg = float(ade_one_sec)/no_samples
            ade_three_sec_avg = float(ade_three_sec)/no_samples
            fde_one_sec_avg = float(fde_one_sec)/no_samples
            fde_three_sec_avg = float(fde_three_sec)/no_samples
            print(f"Validation Iter {i_batch+1}/{num_batches} Avg Loss {avg_loss:.4f} Batch Loss {loss.data:.4f} \
            One sec:- ADE:{ade_one_sec/(no_samples):.4f} FDE: {fde_one_sec/(no_samples):.4f}\
            Three sec:- ADE:{ade_three_sec/(no_samples):.4f} FDE: {fde_three_sec/(no_samples):.4f}",end="\r")
            # print(f"Validation Iter {i_batch+1}/{num_batches} Avg Loss {avg_loss:.4f} \
            # One sec:- ADE:{ade_one_sec/(no_samples):.4f} FDE: {fde_one_sec/(no_samples):.4f}\
            # Three sec:- ADE:{ade_three_sec/(no_samples):.4f} FDE: {fde_three_sec/(no_samples):.4f}",end="\r")
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
    def run(self,num_epochs):
        for i in range(num_epochs):
            self.train_epoch()
            self.val_epoch(epoch=i)
        # self.test_epoch()
        
class Validate():
    def __init__(self,model,val_loader,multi_val_loader,loss_fn,model_dir):
        self.model=model
        self.val_loader=val_loader
        self.multi_val_loader=multi_val_loader
        self.loss_fn=loss_fn
        self.model_dir=model_dir

    def val_epoch(self):
        total_loss=0
        num_batches=len(self.val_loader.batch_sampler)
        self.model.load_state_dict(torch.load(self.model_dir+'best-model.pt')['model_state_dict'])
        self.model.eval() 
        ade_one_sec,fde_one_sec,ade_three_sec,fde_three_sec=(0,0,0,0)
        ade_one_sec_avg, fde_one_sec_avg ,ade_three_sec_avg, fde_three_sec_avg = (0,0,0,0)
        no_samples=0
        for i_batch,traj_dict in enumerate(self.val_loader):
            pred_traj=self.model(traj_dict,mode='validate')
            loss=self.loss_fn(traj_dict['gt_unnorm_traj'],pred_traj)
            pdb.set_trace()
            total_loss=total_loss+loss.data
            avg_loss = float(total_loss)/(i_batch+1)
            batch_samples=gt_traj.shape[0]
            no_samples+=batch_samples

            ade_one_sec+=sum([get_ade(pred_traj[i,:10,:],gt_traj[i,:10,:]) for i in range(batch_samples)])
            fde_one_sec+=sum([get_fde(pred_traj[i,:10,:],gt_traj[i,:10,:]) for i in range(batch_samples)])
            ade_three_sec+=sum([get_ade(pred_traj[i,:,:],gt_traj[i,:,:]) for i in range(batch_samples)])
            fde_three_sec+=sum([get_fde(pred_traj[i,:,:],gt_traj[i,:,:]) for i in range(batch_samples)])

            ade_one_sec_avg = float(ade_one_sec)/no_samples
            ade_three_sec_avg = float(ade_three_sec)/no_samples
            fde_one_sec_avg = float(fde_one_sec)/no_samples
            fde_three_sec_avg = float(fde_three_sec)/no_samples
            
            print(f"Validation Iter {i_batch+1}/{num_batches} Avg Loss {total_loss/(i_batch+1):.4f} \
            One sec:- ADE:{ade_one_sec/(no_samples):.4f} FDE: {fde_one_sec/(no_samples):.4f}\
            Three sec:- ADE:{ade_three_sec/(no_samples):.4f} FDE: {fde_three_sec/(no_samples):.4f}",end="\r")
        
        
    def save_top_accuracy(self):
        print("running save accuracy")
        self.model.load_state_dict(torch.load(self.model_dir+'best-model.pt')['model_state_dict'])
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

        num_batches=len(self.multi_val_loader.batch_sampler)
        for i_batch,traj_dict in enumerate(self.multi_val_loader):
            print(f"Running {i_batch}/{num_batches}",end="\r")
            gt_traj=traj_dict['gt_unnorm_traj'].numpy()
            pred_traj=self.model(traj_dict,mode='validate_multiple')
            loss=[]
            # import pdb;pdb.set_trace()
            for index in range(len(pred_traj)):
                loss_temp=[]
                for j in range(pred_traj[index].shape[0]):
                    loss_temp.append(np.linalg.norm(pred_traj[index][j]- gt_traj[index]))
                # import pdb;pdb.set_trace()
                loss.append(min(loss_temp))
            # import pdb;pdb.set_trace()
            loss=torch.Tensor(loss).float()
            min_loss,min_index=torch.min(loss,dim=0)
            max_loss,max_index=torch.max(loss,dim=0)

            
            
            input_min_list.append(traj_dict['train_unnorm_traj'][min_index])
            pred_min_list.append(pred_traj[min_index])
            target_min_list.append(traj_dict['gt_unnorm_traj'][min_index])


            input_max_list.append(traj_dict['train_unnorm_traj'][max_index])
            pred_max_list.append(pred_traj[max_index])
            target_max_list.append(traj_dict['gt_unnorm_traj'][max_index])
            
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
        
        high_error_path=self.model_dir+"/visualization/high_errors/"
        low_error_path=self.model_dir+"/visualization/low_errors/"

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
            input_max.append(input_max_list[index].numpy())
            pred_max.append(pred_max_list[index])
            target_max.append(target_max_list[index].numpy())
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
            input_min.append(input_min_list[index].numpy())
            pred_min.append(pred_min_list[index])
            target_min.append(target_min_list[index].numpy())
            city_min.append(city_name_min[index])
            viz_sequence(df=pd.read_csv(seq_path_list_min[index]) ,save_path=f"{low_error_path}/dataframe_{i}.png",show=True,avm=avm)
            centerlines_min.append(avm.get_candidate_centerlines_for_traj(input_min[-1], city_min[-1],viz=False))
        import pdb;pdb.set_trace()
        print("Created min array")
        print(f"Saving max visualizations at {high_error_path}")
        viz_predictions(input_=np.array(input_max), output=pred_max,target=np.array(target_max),centerlines=centerlines_max,city_names=np.array(city_max),avm=avm,save_path=high_error_path)
        print(f"Saving min visualizations at {low_error_path}")
        viz_predictions(input_=np.array(input_min), output=pred_min,target=np.array(target_min),centerlines=centerlines_min,city_names=np.array(city_min),avm=avm,save_path=low_error_path)
    def run(self):
        print("in validation run")
        self.save_top_accuracy()

class Test():
    def __init__(self,model,test_loader,model_dir):
        self.test_model=model
        self.model_dir=model_dir
        self.test_loader=test_loader

    def save_trajectory(self,output_dict,save_path):
        generate_forecasting_h5(output_dict, save_path)
        print("done")

    def test(self):
        num_batches=len(self.test_loader.batch_sampler)
        batch_size=self.test_loader.batch_size
        self.test_model.load_state_dict(torch.load(model_dir+'best-model.pt')['model_state_dict'])
        self.test_model.eval()
        no_samples=0
        output_all = {}
        for i_batch,traj_dict in enumerate(self.test_loader):
            seq_paths=traj_dict['seq_path']
            seq_index=[int(os.path.basename(seq_path).split('.')[0]) for seq_path in seq_paths]
            pred_traj=self.test_model(traj_dict,mode='test')
            for index in range(len(pred_traj)):
                pred_traj_index=pred_traj[index]
                if pred_traj_index.shape[0]>6:
                    pred_traj_index=pred_traj_index[:6]
                else:
                    while pred_traj_index.shape[0]<6:
                        pred_traj_index=np.vstack((pred_traj_index,pred_traj_index[0]))
                output_all.update({seq_index[index]:pred_traj_index})
            print(f"Test Iter {i_batch+1}/{num_batches}",end="\r")
        print()
        print("Saving the test data results in dir",self.model_dir)
        self.save_trajectory(output_all,self.model_dir)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='Sequence Modeling - Argoverse Forecasting Task')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for optimizer (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit (default: 10)')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='model type to execute (LSTM, Social.default: LSTM)')
    parser.add_argument('--data', type=str, default='MultiLane',
                        help='type of data to use for training (default: XY, options: XY,LaneCentre,')
    parser.add_argument('--mode',type=str,default='train',help='mode: train, test ,validate')
    parser.add_argument('--model_dir',type=str,default=None,help='model path for test or validate')
    parser.add_argument('--pretrained_dir',type=str,default=None,help='model path to use as pretrained model')
    args = parser.parse_args()

    curr_time = strftime("%Y%m%d%H%M%S", localtime())
    # curr_time="20191129181134" #i guess lstm with centerline embed
    # curr_time="20191201222432" #i guess social"
    # curr_time="20191203132341" #social centerline"
    if args.mode == 'train':
        model_dir = './models/' + args.model + '/' + curr_time + '/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    else:
        model_dir=args.model_dir
    
    # model=LSTMModel()
    if args.model=="Social":
        model=Social_Model()
    elif args.model=="Social_Centerline":
        model=Social_Model_Centerline()
    # model=LSTMModel_CenterlineEmbed()
    loss_fn=nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    print("Mode is ",args.mode)
    print("Data is", args.data)
    print("Model dir is",model_dir)
    print(f"Training for {args.epochs} epochs")

    if args.data=="MultiLane":
        argoverse_map=ArgoverseMap()
        if args.mode=="train" or args.mode=="validate":
            argoverse_val=Argoverse_MultiLane_Data('data/val/data/',avm=argoverse_map,train_seq_size=20,mode="validate",load_saved=True)
            val_loader = DataLoader(argoverse_val, batch_size=16,
                        shuffle=True, num_workers=4,collate_fn=collate_traj_multilane)
        if args.mode=="train":
            argoverse_train=Argoverse_MultiLane_Data('data/train/data/',avm=argoverse_map,train_seq_size=20,mode="train",load_saved=True)
            train_loader = DataLoader(argoverse_train, batch_size=args.batch_size,
                        shuffle=True, num_workers=8,collate_fn=collate_traj_multilane)
        if args.mode=="validate":
            argoverse_val_multiple=Argoverse_MultiLane_Data('data/val/data/',avm=argoverse_map,train_seq_size=20,mode="validate_multiple")
            val_multi_loader=DataLoader(argoverse_val_multiple, batch_size=args.batch_size,
                        shuffle=False, num_workers=4,collate_fn=collate_traj_multilane)
        if args.mode=="test" or args.mode=="train" or args.mode=="validate":
            argoverse_test = Argoverse_MultiLane_Data('data/test_obs/data',avm=argoverse_map,train_seq_size=20,mode="test")
            test_loader = DataLoader(argoverse_test, batch_size=args.batch_size,
                        shuffle=False, num_workers=16,collate_fn=collate_traj_multilane)
    elif args.data=="Social":
        argoverse_map=ArgoverseMap()
        if args.mode=="train" or args.mode=="validate":
            argoverse_val=Argoverse_Social_Data('data/val/data/',avm=argoverse_map,train_seq_size=20,mode="validate",load_saved=True)
            val_loader = DataLoader(argoverse_val, batch_size=args.batch_size,
                        shuffle=True, num_workers=8,collate_fn=collate_traj_xy)
        if args.mode=="train":
            argoverse_train=Argoverse_Social_Data('data/train/data/',avm=argoverse_map,train_seq_size=20,mode="train",load_saved=True)
            train_loader = DataLoader(argoverse_train, batch_size=args.batch_size,
                        shuffle=True, num_workers=8,collate_fn=collate_traj_xy)
        if args.mode=="validate":
            argoverse_val_multiple=Argoverse_Social_Data('data/val/data/',avm=argoverse_map,train_seq_size=20,mode="validate_multiple")
            val_multi_loader=DataLoader(argoverse_val_multiple, batch_size=args.batch_size,
                        shuffle=False, num_workers=8,collate_fn=collate_traj_xy)
        if args.mode=="test" or args.mode=="train" or args.mode=="validate":
            argoverse_test = Argoverse_Social_Data('data/test_obs/data',avm=argoverse_map,train_seq_size=20,mode="test")
            test_loader = DataLoader(argoverse_test, batch_size=args.batch_size,
                        shuffle=False, num_workers=8,collate_fn=collate_traj_xy)

    elif args.data=="Social_Centerline":
        argoverse_map=ArgoverseMap()
        if args.mode=="train" or args.mode=="validate":
            argoverse_val=Argoverse_Social_Centerline_Data('data/val/data/',avm=argoverse_map,train_seq_size=20,mode="validate",load_saved=True)
            val_loader = DataLoader(argoverse_val, batch_size=args.batch_size,
                        shuffle=True, num_workers=8,collate_fn=collate_traj_social_centerline)
        if args.mode=="train":
            argoverse_train=Argoverse_Social_Centerline_Data('data/train/data/',avm=argoverse_map,train_seq_size=20,mode="train",load_saved=True)
            train_loader = DataLoader(argoverse_train, batch_size=args.batch_size,
                        shuffle=True, num_workers=8,collate_fn=collate_traj_social_centerline)
        if args.mode=="validate":
            argoverse_val_multiple=Argoverse_Social_Centerline_Data('data/val/data/',avm=argoverse_map,train_seq_size=20,mode="validate_multiple")
            val_multi_loader=DataLoader(argoverse_val_multiple, batch_size=args.batch_size,
                        shuffle=False, num_workers=8,collate_fn=collate_traj_social_centerline)
        if args.mode=="test" or args.mode=="train" or args.mode=="validate":
            argoverse_test = Argoverse_Social_Centerline_Data('data/test_obs/data',avm=argoverse_map,train_seq_size=20,mode="test")
            test_loader = DataLoader(argoverse_test, batch_size=args.batch_size,
                        shuffle=False, num_workers=8,collate_fn=collate_traj_social_centerline)

            
    else:
        print(f"No dataset: {args.data}. What are you doing")
    # pdb.set_trace()
    if args.mode=="train":
        model=model.cuda()
        trainer=Train(model=model,optimizer=optimizer,train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,loss_fn=loss_fn,model_dir=model_dir,pretrained_dir=args.pretrained_dir)    
        trainer.run(args.epochs)
    elif args.mode=="validate":
        print("In validate")
        model=model.cuda()
        validater=Validate(model=model,val_loader=val_loader,multi_val_loader=val_multi_loader,loss_fn=loss_fn,model_dir=model_dir)
        validater.run()
    elif args.mode=="test":
        pass

    
    


    
from data import Argoverse_Data,collate_traj
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
def data_visualization(dataloader,social=False):
    for i_batch, traj_dict in enumerate(dataloader):
        print(f"{i_batch} batch")
        if social:   
            input_traj=traj_dict['train_agent']
            gt_traj=traj_dict['gt_agent']      
            neighbour_traj=traj_dict['neighbour']
            # import pdb; pdb.set_trace()
            print(f"Shape of neighbour trajectory in batch are",end=' ')
            for i in range(len(neighbour_traj)):
                print(f"{len(neighbour_traj[i])}",end=" ")
            print()# and {len(neighbour_traj[1])}")
        else:
            input_traj=traj_dict['train_agent']
            gt_traj=traj_dict['gt_agent'] 
            plt.grid(True)
            plt.plot(input_traj[0,:,0].numpy(),input_traj[0,:,1].numpy(),'g-o',gt_traj[0,:,0].numpy(),gt_traj[0,:,1].numpy(),'r-o')
            plt.show(block=False)
        
            plt.pause(5)
            plt.clf()
            if i_batch==5:
                exit()

if __name__=="__main__":
    social=True 
    argoverse_sample=Argoverse_Data('data/forecasting_sample/data/',social=social)
    if social:
        train_loader = DataLoader(argoverse_sample, batch_size=2,collate_fn=collate_traj)
    else:
        train_loader = DataLoader(argoverse_sample, batch_size=2)
    data_visualization(train_loader,social=social)

import glob
import os
import pickle
import numpy as np
from argoverse.map_representation.map_api import ArgoverseMap
from visualize_subplot import viz_predictions
# output1_path="models/LSTM_XY/20191204034620/results/"
output1_path="models/TCN/20191130154208/results/"
output2_path="deq/models/trellisnet/20191202145627/results/"
output3_path="deq/models/trellisnet/20191202145627/results/"
output4_path="deq/models/transformers/20191202161816/results/"

file_paths_1=glob.glob(f"{output1_path}/*.pkl")
file_paths_2=glob.glob(f"{output2_path}/*.pkl")
file_paths_3=glob.glob(f"{output3_path}/*.pkl")
file_paths_4=glob.glob(f"{output4_path}/*.pkl")

output_path="models/plot_results/"
ok.makedirs(output_path)

print(f"Len of file paths are {len(file_paths_1)} {len(file_paths_2)} {len(file_paths_3)} {len(file_paths_4)}")
avm=ArgoverseMap()
for index,path in enumerate(file_paths_2):
    print(f"Running {index}/{len(file_paths_2)}")
    path1=os.path.join(output1_path,os.path.basename(path))
    path2=path
    path3=os.path.join(output3_path,os.path.basename(path))
    path4=os.path.join(output4_path,os.path.basename(path))

    input_array=[]
    pred_array=[]
    target_array=[]
    city_names=[]
    centerlines=[]
    
    with open(path1, 'rb') as f:
        dict1=pickle.load(f)
    with open(path2, 'rb') as f:
        dict2=pickle.load(f)
    with open(path3, 'rb') as f:
        dict3=pickle.load(f)
    with open(path4, 'rb') as f:
        dict4=pickle.load(f)
    seq_index1=dict1['seq_index']
    seq_index2=dict2['seq_index']
    seq_index3=dict2['seq_index']
    seq_index4=dict2['seq_index']
    if seq_index1!=seq_index2 and seq_index3!=seq_index4 and seq_index1!=seq_index2:
        print("Something is wrong")
        exit()
    input_array.extend([dict1['input'],dict2['input'],dict3['input'],dict4['input']])
    target_array.extend([dict1['target'],dict2['target'],dict3['target'],dict4['target']])
    pred_array.extend([[dict1['output']],[dict2['output']],[dict3['output']],[dict4['output']]])
    city_names.extend([dict1['city'],dict2['city'],dict3['city'],dict4['city']])

    centerlines.append(avm.get_candidate_centerlines_for_traj(dict1['input'], dict1['city'],viz=False))
    centerlines.append(avm.get_candidate_centerlines_for_traj(dict2['input'], dict2['city'],viz=False))
    centerlines.append(avm.get_candidate_centerlines_for_traj(dict3['input'], dict3['city'],viz=False))
    centerlines.append(avm.get_candidate_centerlines_for_traj(dict4['input'], dict4['city'],viz=False))
    # import pdb;pdb.set_trace()
    viz_predictions(input_=np.array(input_array), output=pred_array,target=np.array(target_array),
                    centerlines=centerlines,city_names=np.array(city_names),avm=avm,save_path=f"{output_path}/{seq_index1}.png")
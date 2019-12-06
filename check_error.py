import glob
import os
import pickle
import numpy as np
from argoverse.map_representation.map_api import ArgoverseMap
from visualize_subplot import viz_predictions
# output1_path="models/LSTM_XY/20191204034620/results/"
output1_path="models/LSTM_XY/20191205171634/results/"
output2_path="models/LSTM/20191204174545/results/"
output3_path="models/Social_Model_Refined/20191205124545/results/"
output4_path="models/Social_Model_Centerline_Refined/20191205095959/results/"

file_paths_1=glob.glob(f"{output1_path}/*.pkl")
file_paths_2=glob.glob(f"{output2_path}/*.pkl")
file_paths_3=glob.glob(f"{output3_path}/*.pkl")
file_paths_4=glob.glob(f"{output4_path}/*.pkl")
output_path="model_compare_results/second/"
print(f"Len of file paths are {len(file_paths_1)} {len(file_paths_2)} {len(file_paths_3)} {len(file_paths_4)}")
avm=ArgoverseMap()
for index,pkl_file_name in enumerate(['4175.pkl','34467.pkl']):
    print(f"Running {index}/{len(file_paths_2)}")
    path1=os.path.join(output1_path,pkl_file_name)
    path2=os.path.join(output2_path,pkl_file_name)
    path3=os.path.join(output3_path,pkl_file_name)
    path4=os.path.join(output4_path,pkl_file_name)

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
    # print(f"Error of dict 1 is: ADE 1 sec: {}, FDE 1:{}")
    # seq_index1=dict1['seq_index']
    # seq_index2=dict2['seq_index']
    # seq_index3=dict2['seq_index']
    # seq_index4=dict2['seq_index']
    # if seq_index1!=seq_index2 and seq_index3!=seq_index4 and seq_index1!=seq_index2:
    #     print("Something is wrong")
    #     exit()
    input_array.extend([dict1['input'],dict2['input'],dict3['input'],dict4['input']])
    target_array.extend([dict1['target'],dict2['target'],dict3['target'],dict4['target']])
    pred_array.extend([dict1['output'],dict2['output'],dict3['output'],dict4['output']])
    input_array=np.array(input_array)
    pred_array=np.array(pred_array)
    target_array=np.array(target_array)
    mse = (np.square(pred_array - target_array)).mean(axis=(1,2))
    print("Avg Error",mse)
    # import pdb;pdb.set_trace()
    fde = (np.square(pred_array[:,-1,:] - target_array[:,-1,:])).mean(axis=1)
    
    print(" FDE Error",fde)

    # city_names.extend([dict1['city'],dict2['city'],dict3['city'],dict4['city']])

    # centerlines.append(avm.get_candidate_centerlines_for_traj(dict1['input'], dict1['city'],viz=False))
    # centerlines.append(avm.get_candidate_centerlines_for_traj(dict2['input'], dict2['city'],viz=False))
    # centerlines.append(avm.get_candidate_centerlines_for_traj(dict3['input'], dict3['city'],viz=False))
    # centerlines.append(avm.get_candidate_centerlines_for_traj(dict4['input'], dict4['city'],viz=False))
    # # import pdb;pdb.set_trace()
    # viz_predictions(input_=np.array(input_array), output=pred_array,target=np.array(target_array),
    #                 centerlines=centerlines,city_names=np.array(city_names),avm=avm,save_path=f"{output_path}/{seq_index1}.png")
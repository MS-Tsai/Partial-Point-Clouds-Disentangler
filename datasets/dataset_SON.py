import os
import warnings
import h5py
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset
import sys
sys.path.append("./data_utils")
from utils import *
from datasets.mapping2 import *

warnings.filterwarnings('ignore')

root_dirs = {
    "no_bg" : "./datasets/Downstream_scanobjectnn/main_split_nobg",
    "with_bg" : "./datasets/Downstream_scanobjectnn/main_split"
}
dataset_modes = {
    "obj" : "_objectdataset.h5",
    "PB_T25" : "_objectdataset_augmented25_norot.h5",
    "PB_T25_R" : "_objectdataset_augmented25rot.h5",
    "PB_T50_R" : "_objectdataset_augmentedrot.h5",
    "PB_T50_RS" : "_objectdataset_augmentedrot_scale75.h5"
}

class ScanObjNN(Dataset):
    def __init__(self, num_in_points=1024, split="train", normalize_mode=None, bg="no_bg", dataset_mode="obj", is_transfer_to_ModelNet=False):
        self.num_in_points = num_in_points
        self.normalize_mode = normalize_mode
        
        # Prepare data path
        root_dir = root_dirs[bg]
        file_split = "training" if split == "train" else "test" 
        data_path = os.path.join(root_dir, file_split + dataset_modes[dataset_mode])
        print("Now we are using", data_path)

        # Load ScanObjectNN dataset
        dataset = h5py.File(data_path)
        self.points = dataset["data"][:]
        self.target = dataset["label"][:]

        if is_transfer_to_ModelNet:
            filtered_data = []
            filtered_label = []
            for i in range(self.target.shape[0]):
                if (self.target[i] in OBJECTDATASET_TO_MODELNET.keys()):
                    filtered_label.append(OBJECTDATASET_TO_MODELNET[self.target[i]][0])
                    filtered_data.append(self.points[i,:])
                # else:
                #     print("Filter out")
            
            self.points = filtered_data
            self.target= filtered_label

        print("The size of %s data is %d" % (split, len(self.points)))

    def __getitem__(self, index):
        # Get target
        target = self.target[index]
        target = np.array([target]).astype(np.int32)
        
        # Get points
        complete = self.points[index]
        if self.normalize_mode == "normalize":
            complete = pcd_normalize(complete) # Normalize complete data into unit sphere
        points = complete[:self.num_in_points, :]
       
        return points, target

    def __len__(self):
        return len(self.points)

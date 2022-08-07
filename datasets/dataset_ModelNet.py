import os
import random
import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append("./data_utils")
from utils import *

root_dir = "./datasets/Downstream_ModelNet40"
obj_name_file = "ModelNet40_names.txt"

class ModelNet(Dataset):
    def __init__(self, split="train", num_in_points=1024, normalize_mode=None):
        # Setup some parameters
        self.num_in_points = num_in_points
        self.normalize_mode = normalize_mode

        class_names = [line.rstrip() for line in open(os.path.join(root_dir, obj_name_file), "r")]
        self.target_table = dict(zip(class_names, range(len(class_names))))

        self.model_list = []
        for class_name in class_names:
            file_dir = os.path.join(root_dir, class_name, split)
            filenames = os.listdir(file_dir)
            for filename in filenames:
                file_path = os.path.join(file_dir, filename)
                self.model_list.append((class_name, file_path))

        # Shuffle model_list and compute the length of model_list
        random.shuffle(self.model_list)
        self.len = len(self.model_list)

    def __getitem__(self, index):
        # Get complete data
        class_name, complete_path = self.model_list[index]
        complete = read_pcd(complete_path)
        if self.normalize_mode == "normalize":
            complete = pcd_normalize(complete) # Normalize complete data into unit sphere
        
        # Get classification label
        gt_label = self.target_table[class_name]
        gt_label = np.array([gt_label]).astype(np.int32)

        # Random sample complete data into 1,024 points
        points = complete[:self.num_in_points, :]
        
        return points, gt_label

    def __len__(self):
        return self.len

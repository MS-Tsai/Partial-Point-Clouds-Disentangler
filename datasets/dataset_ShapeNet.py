import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random
import sys
import json
sys.path.append("./data_utils")
from utils import *

ID2N = {
    "02691156": "Airplane",
    "02933112": "Cabinet",
    "02958343": "Car",
    "03001627": "Chair",
    "03636649": "Lamp",
    "04256520": "Sofa",
    "04379243": "Table",
    "04530566": "Watercraft"
}

def alignment(pcd):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    # Create rotation matrix
    angle_x = np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    # Rotate
    rotated_pcd = np.dot(pcd, Rx)

    return rotated_pcd

def rotate_point_cloud(partial_pcd, complete_pcd):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    # Create rotation matrix
    angle_x = np.random.uniform() * 2 * np.pi
    angle_y = np.random.uniform() * 2 * np.pi
    angle_z = np.random.uniform() * 2 * np.pi

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])

    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])

    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    # Rotate
    R = Ry
    rotated_partial_pcd = np.dot(partial_pcd, R)
    rotated_complete_pcd = np.dot(complete_pcd, R)

    return rotated_partial_pcd.astype(np.float32), rotated_complete_pcd.astype(np.float32)

def create_6DOF(pose, pose_format):
    # Transfer rotation matrix into rotation angle along x, y and z axis
    R = pose[:3, :3]
    x = np.arctan2(R[2, 1], R[2, 2])
    y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]*R[2, 1] + R[2, 2]*R[2, 2]))
    z = np.arctan2(R[1, 0], R[0, 0])
    
    # Create 6 DOF pose data
    if pose_format == "angle":
        r = np.array([x, y]) * 180 / np.pi # convert radian format to angle format
    elif pose_format == "radian":
        r = np.array([x, y]) 
    
    return r.astype(np.float32)

class ShapeNet(data.Dataset):
    def __init__(self, split= "train", num_in_points=1024, num_out_points=2048, num_objs=1000, pose_format=None, blender_mode="26poses", input_case=None):
        self.num_in_points = num_in_points
        self.num_out_points = num_out_points
        self.num_poses = 26
        self.pose_format = pose_format
        self.input_case = input_case

        # Get partial_root_dir and complete_root_dir
        partial_root_dir = "./datasets/Pretext_ShapeNetCore.v1_blender_" + blender_mode + "/list"
        complete_root_dir = "./datasets/Pretext_ShapeNetCore.v1_complete"
        
        # Get pcd_list and pose_list from partial_root_dir
        data_len = int(num_objs * 26 * 8) if split == "train" else int(num_objs / 10 * 26 * 8)
        pcd_file = os.path.join(partial_root_dir, "All_" + split + "_pcd.list")
        pose_file = os.path.join(partial_root_dir, "All_" + split + "_pose.list")
        self.pcd_list = [line.rstrip() for line in open(pcd_file)][:data_len]
        self.pose_list = [line.rstrip() for line in open(pose_file)][:data_len]
        
        # Get complete_pcd_path from complete_root_dir
        object_list = [path.split("/")[-3:-1] for path in self.pcd_list]
        with open(os.path.join(complete_root_dir, "complete_path.json")) as fp:
            self.complete_dict = json.load(fp) 
        self.complete_path = [self.complete_dict[obj[0]][obj[1]] for obj in object_list]
        
        print('The size of %s data is %d'%(split, len(self.complete_path)))
        
    def __getitem__(self, index):
        # Get pcd_path_a, pose_path_a and complete_path_a based on different index
        pcd_path_a = self.pcd_list[index]
        pose_path_a = self.pose_list[index]
        complete_path_a = self.complete_path[index]

        # Get pcd_path_b, pose_path_b and complete_path_b based on different input_case
        input_case = self.input_case 
        if input_case == 0:
            pose_id_a = os.path.basename(pcd_path_a)[:-4] 
            pose_id_b = str((int(pose_id_a) + np.random.randint(0, self.num_poses)) % self.num_poses)
            pcd_path_b = pcd_path_a.replace(pose_id_a+".pcd", pose_id_b+".pcd") 
            pose_path_b = pose_path_a.replace(pose_id_a+".txt", pose_id_b+".txt") 
            complete_path_b = complete_path_a
        elif input_case == 1:
            pose_id_a = os.path.basename(pcd_path_a)[:-4]
            while True:
                pcd_b_synset_ID = random.choice(list(ID2N.keys()))
                pcd_b_model_ID = random.choice(list(self.complete_dict[pcd_b_synset_ID].keys()))
                pcd_path_b = os.path.join("/".join(pcd_path_a.split("/")[:-5]), ID2N[pcd_b_synset_ID], "pcd", pcd_b_synset_ID, pcd_b_model_ID, pose_id_a+".pcd")
                pose_path_b = pcd_path_b.replace(".pcd", ".txt").replace("/pcd/", "/pose_gt/")
                complete_path_b = self.complete_dict[pcd_b_synset_ID][pcd_b_model_ID]
                if pcd_path_b in self.pcd_list:
                    break
        elif input_case == 2:
            while True:
                pose_id_b = str(np.random.randint(0, self.num_poses))
                pcd_b_synset_ID = random.choice(list(ID2N.keys()))
                pcd_b_model_ID = random.choice(list(self.complete_dict[pcd_b_synset_ID].keys()))
                pcd_path_b = os.path.join("/".join(pcd_path_a.split("/")[:-5]), ID2N[pcd_b_synset_ID], "pcd", pcd_b_synset_ID, pcd_b_model_ID, pose_id_b+".pcd")
                pose_path_b = pcd_path_b.replace(".pcd", ".txt").replace("/pcd/", "/pose_gt/")
                pcd_b_info = pcd_path_b.split("/")[7:9]
                complete_path_b = self.complete_dict[pcd_b_synset_ID][pcd_b_model_ID]
                if pcd_path_b in self.pcd_list:
                    break

        # Get gt_completes, partials from complete_path, pcd_path
        gt_complete_a = self.prepare_pcd(complete_path_a, self.num_out_points)
        gt_complete_b = self.prepare_pcd(complete_path_b, self.num_out_points)
        partial_a = self.prepare_pcd(pcd_path_a, self.num_in_points)
        partial_b = self.prepare_pcd(pcd_path_b, self.num_in_points)
        
        # Augmentation
        partial_a = alignment(partial_a)
        partial_b = alignment(partial_b)
        partial_a, _ = rotate_point_cloud(partial_a, gt_complete_a)
        partial_b, _ = rotate_point_cloud(partial_b, gt_complete_b)
        
        gt_completes = [gt_complete_a, gt_complete_b]
        partials = [partial_a, partial_b]

        # Get gt_partials based on different case
        if input_case == 0:
            gt_pcd_path_a = pcd_path_b
            gt_pcd_path_b = pcd_path_a
        elif input_case == 1:
            gt_pcd_path_a = pcd_path_a
            gt_pcd_path_b = pcd_path_b
        elif input_case == 2:
            gt_pcd_path_a = os.path.join("/".join(pcd_path_a.split("/")[:-1]), pcd_path_b.split("/")[-1])
            gt_pcd_path_b = os.path.join("/".join(pcd_path_b.split("/")[:-1]), pcd_path_a.split("/")[-1])
        gt_partial_a = self.prepare_pcd(gt_pcd_path_a, self.num_in_points)
        gt_partial_b = self.prepare_pcd(gt_pcd_path_b, self.num_in_points)
        gt_partials = [gt_partial_a, gt_partial_b]

        # Read pose file and Create 6 DOF pose as ground truth
        pose_a = np.array([list(map(float, line.rstrip().split())) for line in open(pose_path_a, "r")])
        pose_b = np.array([list(map(float, line.rstrip().split())) for line in open(pose_path_b, "r")])
        gt_pose_a = create_6DOF(pose_a, self.pose_format)
        gt_pose_b = create_6DOF(pose_b, self.pose_format)
        gt_poses = [gt_pose_a, gt_pose_b]

        return partials, gt_completes, gt_partials, gt_poses

    def __len__(self):
        return len(self.complete_path)

    def prepare_pcd(self, path, num_points):
        pcd = read_pcd(path)
        pcd = pcd_normalize(pcd) # Normalize complete data into unit sphere
        pcd = resample_pcd(pcd, num_points) # Sample complete data to num_out_points
        
        return pcd.astype(np.float32)

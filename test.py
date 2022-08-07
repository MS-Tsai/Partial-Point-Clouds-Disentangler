import os
import json
import argparse
import numpy as np
import open3d as o3d
import torch
import sys
sys.path.append("./datasets")
from dataset_ModelNet import ModelNet
from dataset_SON import ScanObjNN
sys.path.append("./models")
from Disentangler import *

parser = argparse.ArgumentParser()
# Official
parser.add_argument("--batch_size", type=int, default=32, help="Input batch size")
parser.add_argument("--workers", type=int, default=8, help="Number of data loading workers")
parser.add_argument("--pretrained_model", type=str, default="", help="optional reload model path") # Load model's weight
parser.add_argument("--num_in_points", type=int, default=1024, help="Number of points")
parser.add_argument("--num_out_points", type=int, default=2048, help="Number of points")
parser.add_argument("--n_primitives", type=int, default=16, help="Number of surface elements")
parser.add_argument("--gpu", type=str, default="0", help="Specify gpu device (Comma-sparated) [default: 0]")
parser.add_argument("--backbone", type=str, default="PN", choices=["PN", "DGCNN"], help="Specify backbone network")
parser.add_argument("--dataset_mode", type=str, default="ModelNet40", choices=["ModelNet40", "ScanObjNN"], help="Specify testing dataset")
parser.add_argument("--normalize_mode", type=str, default="", choices=["", "normalize"], help="Specify normalized method of testing dataset")
args = parser.parse_args()
print("Argument parameter:\n{}\n".format(args))

# HYPER PARAMETER
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # GPU devices
lrate = 0.001 # Learning rate

def backup_python_file(backup_dir):
    # Backup important python file
    os.system("cp ./test.py %s" %(backup_dir))
    os.system("cp ./datasets/dataset_ModelNet.py %s" %(backup_dir))
    os.system("cp ./datasets/dataset_SON.py %s" %(backup_dir))

def create_dataloader():
    if args.dataset_mode == "ModelNet40":
        dataset = ModelNet(split="train", num_in_points=args.num_in_points, normalize_mode=args.normalize_mode)
        dataset_test = ModelNet(split="test", num_in_points=args.num_in_points, normalize_mode=args.normalize_mode)
    elif args.dataset_mode == "ScanObjNN":
        dataset = ScanObjNN(split="train", num_in_points=args.num_in_points, normalize_mode=args.normalize_mode)
        dataset_test = ScanObjNN(split="test", num_in_points=args.num_in_points, normalize_mode=args.normalize_mode)
    
    # Create training and testing dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print("Training set size: {}".format(len(dataset)))
    print("Testing set size: {}".format(len(dataset_test)))

    return dataloader, dataloader_test

def create_network():
    # Create MSN network
    network = Disentangler(num_out_points=args.num_out_points, n_primitives=args.n_primitives, backbone=args.backbone, mode="cls")
    network = torch.nn.DataParallel(network).cuda()

    # Load pre-trained weights
    network.module.load_state_dict(torch.load(args.pretrained_model))
    network.module.eval()
    print("Previous weight loaded.")

    return network

def test(network, dataloader, split=None):            
    # VALIDATION MODE
    with torch.no_grad():
        for data_idx, data in enumerate(dataloader, 0):
            # Get input and gt data from dataloader
            input_data, gt_label = data
            input_data = input_data.float().cuda().transpose(2,1).contiguous()
            
            # Get global feature from MSN
            content_feat, pose_feat = network(input_data, input_data)
            useful_content_feat = len(torch.where(torch.std(content_feat, dim=0) != 0)[0])
            useful_pose_feat = len(torch.where(torch.std(pose_feat, dim=0) != 0)[0])
            print("val [%d: %d/%d]\tuseful content / pose: %d / %d" %(1, data_idx, len(dataloader), useful_content_feat, useful_pose_feat))

            # Record validation loss information
            save_feat[split]["content_feat"].extend(content_feat.cpu().detach().numpy().tolist())
            save_feat[split]["pose_feat"].extend(pose_feat.cpu().detach().numpy().tolist())
            save_feat[split]["gt"].extend(gt_label.cpu().numpy().reshape((-1)).tolist())

if __name__ == "__main__":    
    # Create output directory
    weight_filename = args.pretrained_model.split("/")[-1]
    weight_id = "" if weight_filename == "best_model.pth" else weight_filename[:-4] + "_"
    output_dir = "/".join(args.pretrained_model.split("/")[:-2])
    backup_dir = os.path.join(output_dir, "backup")
    features_dir = os.path.join(output_dir, "features_%s" %(args.dataset_mode))
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    # Backup important .py file
    backup_python_file(backup_dir)

    # Create training and testing dataloader
    dataloader, dataloader_test = create_dataloader()

    # Create MSN network
    network = create_network()

    # Start evaluation
    save_feat = {split: {"content_feat": [], "pose_feat": [], "gt": []} for split in ["train", "test"]}
    test(network, dataloader, split="train")
    test(network, dataloader_test, split="test")

    # Record features and gt_label
    features_filename = "%sfeatures.json" %(weight_id) if args.normalize_mode == "" else "%sfeatures_%s.json" %(weight_id, args.normalize_mode)
    with open(os.path.join(features_dir, features_filename), "w") as f:
        f.write(json.dumps(save_feat, indent=4))

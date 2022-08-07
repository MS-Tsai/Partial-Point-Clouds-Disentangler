import os
import json
import argparse
import random
import numpy as np
import open3d as o3d
import torch
import torch.optim as optim
from tqdm import tqdm
from datasets.dataset_ShapeNet import ShapeNet
import sys
sys.path.append("./models")
from Disentangler import *
sys.path.append("./data_utils")
from MSN_utils import *

parser = argparse.ArgumentParser()
# Official
parser.add_argument("--batch_size", type=int, default=32, help="Input batch size")
parser.add_argument("--workers", type=int, default=8, help="Number of data loading workers")
parser.add_argument("--nepoch", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--pretrained_model", type=str, default="", help="optional reload model path") # Load model's weight
parser.add_argument("--num_in_points", type=int, default=1024, help="Number of points")
parser.add_argument("--num_out_points", type=int, default=2048, help="Number of points")
parser.add_argument("--n_primitives", type=int, default=16, help="Number of surface elements")
# Customized
parser.add_argument("--gpu", type=str, default="0", help="Specify gpu device (Comma-sparated) [default: 0]")
parser.add_argument("--output_dir", type=str, default="./outputs", help="Root output directory for everything")
parser.add_argument("--backbone", type=str, default="PN", choices=["PN", "DGCNN"], help="Choose backbone")
parser.add_argument("--input_case", type=int, default=0, choices=[0, 1, 2, 3, 4], help="Choose input data mode [default: same models, different poses]")
parser.add_argument("--blender_mode", type=str, default="26poses", choices=["6poses", "26poses", "30degree"], help="Choose how blender create the 3DPoseNet data")
parser.add_argument("--pose_format", type=str, default="angle", choices=["angle", "radian"], help="Choose pose format for pose regression")
parser.add_argument("--num_objs", type=int, default=10, help="Number of objects in each category")
args = parser.parse_args()
print("Argument parameter:\n{}\n".format(args))

# GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 

# Setup random seed for random and torch
args.manualSeed = np.random.randint(0, 10000)
print("Fixed Seed: ", args.manualSeed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)
np.random.seed(args.manualSeed)
random.seed(args.manualSeed)

# HYPER PARAMETER
lrate = 0.001 # Learning rate
train_loss = AverageValueMeter() # Calculas training loss
val_loss = AverageValueMeter() # Calculas testing loss
best_val_loss = 10
train_loss_save = {
    "EMD": [],
    "expansion": [],
    "EMD_complete": [],
    "expansion_complete": [],
    "EMD_partial": [],
    "expansion_partial": [],
    "pose": []
}
val_loss_save = {
    "EMD": [],
    "expansion": [],
    "EMD_complete": [],
    "expansion_complete": [],
    "EMD_partial": [],
    "expansion_partial": [],
    "pose": []
}

def print_network(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("%d"%params)


def create_output_dir():
    # Create output directry according to num_objs, blender_mode and rotation_format
    output_dir = os.path.join(args.output_dir, args.backbone + "_" + args.blender_mode + "_list" + str(args.num_objs) + "_case" + str(args.input_case) + "_" + args.pose_format)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir

def backup_python_file(backup_dir):
    # Backup important python file
    os.system("cp ./train.py %s" %(backup_dir))
    os.system("cp ./datasets/dataset_ShapeNet.py %s" %(backup_dir))
    os.system("cp ./models/Disentangler.py %s" %(backup_dir))

def create_dataloader():
    # Create training dataloader
    dataset = ShapeNet(split="train", num_in_points=args.num_in_points, num_out_points=args.num_out_points, blender_mode=args.blender_mode, num_objs=args.num_objs, pose_format=args.pose_format, input_case=args.input_case)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # Create testing dataloader
    dataset_test = ShapeNet(split="test", num_in_points=args.num_in_points, num_out_points=args.num_out_points, blender_mode=args.blender_mode, num_objs=args.num_objs, pose_format=args.pose_format, input_case=args.input_case)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print("Training set size: {}".format(len(dataset)))
    print("Testing set size: {}".format(len(dataset_test)))

    return dataloader, dataloader_test

def create_network():
    # Create MSN network and optimizer
    network = torch.nn.DataParallel(Disentangler(num_in_points=args.num_in_points, num_out_points=args.num_out_points, n_primitives=args.n_primitives, backbone=args.backbone)).cuda()
    network.module.apply(weights_init) # Initialization of the weight
    optimizer = optim.Adam(network.module.parameters(), lr=lrate) # Optimizer
    criterion = get_loss().cuda()

    # Check if pre-trained weight is existed or not
    if args.pretrained_model != '':
        network.module.load_state_dict(torch.load(args.pretrained_model))
        print("Previous weight loaded.")

    print_network(network)
    
    return network, optimizer, criterion

def train(network, optimizer, criterion, dataloader, dataloader_test, log_path, weights_dir):
    for epoch in range(1, args.nepoch + 1):
        # TRAIN MODE
        train_loss.reset()
        network.module.train()

        # Learning rate schedule
        if epoch == 20:
            optimizer = optim.Adam(network.module.parameters(), lr=lrate/10.0)
        if epoch == 40:
            optimizer = optim.Adam(network.module.parameters(), lr=lrate/100.0)
    
        for data_idx, data in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
            optimizer.zero_grad() # Initialize optimizer
            
            # Get partials, gt_completes, gt_partials and gt_poses from dataloader
            partials, gt_completes, gt_partials, gt_poses = data
            partials = [partial.float().cuda().transpose(2, 1).contiguous() for partial in partials]
            gt_completes = [gt_complete.float().cuda().contiguous() for gt_complete in gt_completes]
            gt_partials = [gt_partial.float().cuda().contiguous() for gt_partial in gt_partials]
            gt_poses = [gt_pose.cuda() for gt_pose in gt_poses]

            # Get recon_output, expansion_penalty from MSN
            partial_a, partial_b = partials
            recon_complete_outputs, expansion_completes, recon_partial_outputs, expansion_partials, pred_poses = network(partial_a, partial_b)
            if True in np.isnan(pred_poses[0].data.cpu().numpy()):
                print(path)
                assert False
            
            # Compute EMD loss and total recon loss
            loss_EMD_complete_a, loss_EMD_complete_b = criterion(recon_complete_outputs, gt_completes, 0.005, 50)
            loss_EMD_partial_a, loss_EMD_partial_b = criterion(recon_partial_outputs, gt_partials, 0.005, 50)
            loss_pose_a, loss_pose_b = criterion.compute_pose_loss(pred_poses, gt_poses)
            expansion_complete_a, expansion_complete_b = expansion_completes
            expansion_partial_a, expansion_partial_b = expansion_partials
            loss_recon_complete = (loss_EMD_complete_a.mean() + expansion_complete_a.mean() * 0.1 + loss_EMD_complete_b.mean() + expansion_complete_b.mean() * 0.1) / 2.0
            loss_recon_partial = (loss_EMD_partial_a.mean() + expansion_partial_a.mean() * 0.1 + loss_EMD_partial_b.mean() + expansion_partial_b.mean() * 0.1) / 2.0
            loss_pose = (loss_pose_a + loss_pose_b) / 2.0
            total_loss = loss_recon_complete + loss_recon_partial + loss_pose * 0.01
            
            # Record training loss information
            loss_EMD_complete = (loss_EMD_complete_a + loss_EMD_complete_b) / 2.0
            expansion_complete = (expansion_complete_a + expansion_complete_b) / 2.0
            loss_EMD_partial = (loss_EMD_partial_a + loss_EMD_partial_b) / 2.0
            expansion_partial = (expansion_partial_a + expansion_partial_b) / 2.0
            loss_EMD = (loss_EMD_complete + loss_EMD_partial) / 2.0
            expansion_penalty = (expansion_complete + expansion_partial) / 2.0
            train_loss.update(loss_EMD.mean().item())
            train_loss_save["EMD"].append(loss_EMD.mean().item())
            train_loss_save["expansion"].append(expansion_penalty.mean().item())
            train_loss_save["EMD_complete"].append(loss_EMD_complete.mean().item())
            train_loss_save["expansion_complete"].append(expansion_complete.mean().item())
            train_loss_save["EMD_partial"].append(loss_EMD_partial.mean().item())
            train_loss_save["expansion_partial"].append(expansion_partial.mean().item())
            train_loss_save["pose"].append(loss_pose.item())
            print("Train [%d: %d/%d]\tEMD_com_a: %.4f, expansion_com_a: %.4f, EMD_part_a: %.4f, expansion_part_a: %.4f, pose_a: %.2f" 
                %(epoch, data_idx, len(dataloader), loss_EMD_complete_a.mean().item(), expansion_complete_a.mean().item(), loss_EMD_partial_a.mean().item(), expansion_partial_a.mean().item(), loss_pose_a.item())
            )
            print("Train [%d: %d/%d]\tEMD_com_b: %.4f, expansion_com_b: %.4f, EMD_part_b: %.4f, expansion_part_b: %.4f, pose_b: %.2f" 
                %(epoch, data_idx, len(dataloader), loss_EMD_complete_b.mean().item(), expansion_complete_b.mean().item(), loss_EMD_partial_b.mean().item(), expansion_partial_b.mean().item(), loss_pose_b.item())
            )
            print("")

            # Update network by total loss
            total_loss.backward()
            optimizer.step()
            
        # VALIDATION MODE per 5 epochs
        if epoch % 5 == 0:
            val_loss.reset()
            network.module.eval()
            with torch.no_grad():
                for data_idx, data in tqdm(enumerate(dataloader_test, 0), total=len(dataloader_test)):
                    # Get partials, gt_completes, gt_partials and gt_poses from dataloader
                    partials, gt_completes, gt_partials, gt_poses = data
                    partials = [partial.float().cuda().transpose(2, 1).contiguous() for partial in partials]
                    gt_completes = [gt_complete.float().cuda().contiguous() for gt_complete in gt_completes]
                    gt_partials = [gt_partial.float().cuda().contiguous() for gt_partial in gt_partials]
                    gt_poses = [gt_pose.cuda() for gt_pose in gt_poses]

                    # Get recon_output, loss_EMD, expansion_penalty from MSN
                    partial_a, partial_b = partials
                    recon_complete_outputs, expansion_completes, recon_partial_outputs, expansion_partials, pred_poses = network(partial_a, partial_b)

                    # Compute EMD loss and pose loss
                    loss_EMD_complete_a, loss_EMD_complete_b = criterion(recon_complete_outputs, gt_completes, 0.004, 3000)
                    loss_EMD_partial_a, loss_EMD_partial_b = criterion(recon_partial_outputs, gt_partials, 0.004, 3000)
                    expansion_complete_a, expansion_complete_b = expansion_completes
                    expansion_partial_a, expansion_partial_b = expansion_partials
                    loss_pose_a, loss_pose_b = criterion.compute_pose_loss(pred_poses, gt_poses)

                    # Record validation loss information
                    loss_EMD_complete = (loss_EMD_complete_a + loss_EMD_complete_b) / 2.0
                    loss_EMD_partial = (loss_EMD_partial_a + loss_EMD_partial_b) / 2.0
                    loss_EMD = (loss_EMD_complete + loss_EMD_partial) / 2.0
                    expansion_complete = (expansion_complete_a + expansion_complete_b) / 2.0
                    expansion_partial = (expansion_partial_a + expansion_partial_b) / 2.0
                    expansion_penalty = (expansion_complete + expansion_partial) / 2.0
                    loss_pose = (loss_pose_a + loss_pose_b) / 2.0
                    val_loss.update(loss_EMD.mean().item())
                    val_loss_save["EMD"].append(loss_EMD.mean().item())
                    val_loss_save["expansion"].append(expansion_penalty.mean().item())
                    val_loss_save["EMD_complete"].append(loss_EMD_complete.mean().item())
                    val_loss_save["expansion_complete"].append(expansion_complete.mean().item())
                    val_loss_save["EMD_partial"].append(loss_EMD_partial.mean().item())
                    val_loss_save["expansion_partial"].append(expansion_partial.mean().item())
                    val_loss_save["pose"].append(loss_pose.item())
                    print("Val [%d: %d/%d]\tEMD_com_a: %.4f, expansion_com_a: %.4f, EMD_part_a: %.4f, expansion_part_a: %.4f, pose_a: %.2f" 
                        %(epoch, data_idx, len(dataloader_test), loss_EMD_complete_a.mean().item(), expansion_complete_a.mean().item(), loss_EMD_partial_a.mean().item(), expansion_partial_a.mean().item(), loss_pose_a.item())
                    )
                    print("Val [%d: %d/%d]\tEMD_com_b: %.4f, expansion_com_b: %.4f, EMD_part_b: %.4f, expansion_part_b: %.4f, pose_b: %.2f" 
                        %(epoch, data_idx, len(dataloader_test), loss_EMD_complete_b.mean().item(), expansion_complete_b.mean().item(), loss_EMD_partial_b.mean().item(), expansion_partial_b.mean().item(), loss_pose_b.item())
                    )
                    print("")
        
        # Save training details into log.txt
        log_table = {
            "train_loss" : train_loss.avg,
            "val_loss" : val_loss.avg,
            "epoch" : epoch,
            "lr" : lrate,
            "bestval" : best_val_loss,
        }
        with open(log_path, "a") as f:
            f.write("json_stats: " + json.dumps(log_table) + "\n")

        # Save network weights and loss
        print("Saving weights and loss\n")
        if (epoch == 1) or (epoch % 5 == 0):
            torch.save(network.module.state_dict(), os.path.join(weights_dir, "model_%02d.pth" % (epoch)), _use_new_zipfile_serialization=False)
        np.save(os.path.join(output_dir, "train_loss.npy") , train_loss_save)
        np.save(os.path.join(output_dir, "val_loss.npy") , val_loss_save)

if __name__ == "__main__":
    # Create output directory
    output_dir = create_output_dir()
    log_path = os.path.join(output_dir, "log.txt")
    backup_dir = os.path.join(output_dir, "backup")
    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # Backup important .py file
    backup_python_file(backup_dir)

    # Create training and testing dataloader
    dataloader, dataloader_test = create_dataloader()

    # Create MSN network
    network, optimizer, criterion = create_network()

    # Write argument parameter and network into log.txt
    with open(log_path, "a") as fp:
        fp.write(str(args) + "\n")
        fp.write(str(network.module) + "\n")

    # Start training
    train(network, optimizer, criterion, dataloader, dataloader_test, log_path, weights_dir)

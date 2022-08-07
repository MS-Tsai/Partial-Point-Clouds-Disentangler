from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
sys.path.append("./emd/")
import emd_module as emd

from PointNet_encoder import PointNetfeat
from DGCNN_encoder import DGCNN
from MSN_decoder import MSN_decoder
from pose_regressor import pose_regressor

class Disentangler(nn.Module):
    def __init__(self, num_in_points=1024, num_out_points=2048, bottleneck_size=1024, n_primitives=16, backbone=None, mode="train"):
        super(Disentangler, self).__init__()
        self.num_poses = 26
        self.backbone = backbone
        self.mode = mode
        
        if backbone == "PN":
            # Use PointNet encoder to extract global feature of point clouds
            self.encoder_content = nn.Sequential(
                PointNetfeat(num_in_points, global_feat=True),
                nn.Linear(1024, bottleneck_size),
                nn.BatchNorm1d(bottleneck_size),
                nn.ReLU()
            )
            self.encoder_pose = nn.Sequential(
                PointNetfeat(num_in_points, global_feat=True),
                nn.Linear(1024, bottleneck_size),
                nn.BatchNorm1d(bottleneck_size),
                nn.ReLU()
            )
        elif backbone == "DGCNN":
            # Use DGCNN encoder to extract global feature of point clouds
            bottleneck_size = 512
            self.encoder_content = DGCNN(emb_dims=bottleneck_size)
            self.encoder_pose = DGCNN(emb_dims=bottleneck_size)
            bottleneck_size = bottleneck_size * 2

        # For reconstruction branch
        self.decoder_complete = MSN_decoder(num_out_points, bottleneck_size, n_primitives)
        # self.decoder_partial = MSN_decoder(num_in_points, bottleneck_size*2, n_primitives)
        self.decoder_partial = MSN_decoder(num_in_points, bottleneck_size*2, 1)

        # For pose regression branch
        self.pose_regressor = pose_regressor(bottleneck_size)
        
    def forward(self, partial_a, partial_b):
        # Get content feature from PointNet encoder
        content_feat_a = self.encoder_content(partial_a) # content_feat_a: (B, 1024)
        content_feat_b = self.encoder_content(partial_b) # content_feat_b: (B, 1024)
        content_feat = content_feat_a.clone().detach()

        # Get pose feature from PointNet encoder
        pose_feat_a = self.encoder_pose(partial_a) # pose_feat_a: (B, 1024)
        pose_feat_b = self.encoder_pose(partial_b) # pose_feat_b: (B, 1024)
        pose_feat = pose_feat_a.clone().detach()
        
        # Downstream task ... Classification
        if self.mode == "cls":
            return content_feat, pose_feat

        # Get recon_output and expansion_penalty by complete decoder
        recon_complete_output_a, expansion_complete_a = self.decoder_complete(content_feat_a)
        recon_complete_output_b, expansion_complete_b = self.decoder_complete(content_feat_b)
        recon_complete_outputs = [recon_complete_output_a, recon_complete_output_b]
        expansion_completes = [expansion_complete_a, expansion_complete_b]

        # Get recon_output and expansion_penalty by partial decoder
        feature_ab = torch.cat((content_feat_a, pose_feat_b), dim=1)
        feature_ba = torch.cat((content_feat_b, pose_feat_a), dim=1)
        recon_partial_output_a, expansion_partial_a = self.decoder_partial(feature_ab)
        recon_partial_output_b, expansion_partial_b = self.decoder_partial(feature_ba)
        recon_partial_outputs = [recon_partial_output_a, recon_partial_output_b]
        expansion_partials = [expansion_partial_a, expansion_partial_b]

        # Get pose prediction
        pred_pose_a = self.pose_regressor(pose_feat_a)
        pred_pose_b = self.pose_regressor(pose_feat_b)
        pred_poses = [pred_pose_a, pred_pose_b]
        
        return recon_complete_outputs, expansion_completes, recon_partial_outputs, expansion_partials, pred_poses

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.EMD_criterion = emd.emdModule()
        self.pose_criterion = nn.MSELoss().cuda()

    def forward(self, recon_outputs, gts, eps, iters):
        loss_EMDs = []
        for recon_output, gt in zip(recon_outputs, gts):
            dist, _ = self.EMD_criterion(recon_output, gt, eps, iters)
            loss_EMDs.append(torch.sqrt(dist).mean(1))

        return loss_EMDs

    def compute_pose_loss(self, pred_poses, gt_poses):
        loss_poses = []
        for pred_pose, gt_pose in zip(pred_poses, gt_poses):
            loss_poses.append(self.pose_criterion(pred_pose, gt_pose))
        
        return loss_poses


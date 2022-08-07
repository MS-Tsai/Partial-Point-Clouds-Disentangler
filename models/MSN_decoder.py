from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
sys.path.append("./expansion_penalty/")
import expansion_penalty_module as expansion

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=8192):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))

        return x

class MSN_decoder(nn.Module):
    def __init__(self, num_points=2048, bottleneck_size=1024, n_primitives=16):
        super(MSN_decoder, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives

        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.n_primitives)])
        self.expansion = expansion.expansionPenaltyModule()

    def forward(self, x):
        # Get recon_output and expansion_penalty
        patch_outs = []
        for i in range(0,self.n_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points//self.n_primitives)) # rand_grid: (B, 2, 512)
            rand_grid.data.uniform_(0,1)
            patch = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous() # patch: (B, 1024, 512)
            patch = torch.cat( (rand_grid, patch), 1).contiguous() # patch: (B, 1026, 512)
            patch_outs.append(self.decoder[i](patch)) # patch_outs: (16, B, 3, 512)

        patch_outs = torch.cat(patch_outs, 2).contiguous() # patch_outs: (B, 3, num_points)
        recon_output = patch_outs.transpose(1, 2).contiguous() # recon_output: (B, num_points, 3)
        
        # Compute expansion loss (using Minimum Spanning Tree (MST))
        if self.n_primitives < 16:
            loss_mst = torch.tensor(0).float().cuda()
        else:
            dist, _, mean_mst_dis = self.expansion(recon_output, self.num_points//self.n_primitives, 1.5)
            loss_mst = torch.mean(dist)

        return recon_output, loss_mst

from __future__ import print_function
import torch.nn as nn

class pose_regressor(nn.Module):
    def __init__(self, bottleneck_size=1024):
        super(pose_regressor, self).__init__()
        
        # Create pose estimation branch
        self.regressor = nn.Sequential(
            nn.Linear(bottleneck_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        pred_pose = self.regressor(x)

        return pred_pose

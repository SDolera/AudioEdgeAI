import os

import torch
from torch import nn



class CustomDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block = nn.Sequential(

            nn.Conv1d(1, 16, 3, padding=1), nn.BatchNorm1d(16), nn.PReLU(),
            nn.Conv1d(16, 16, 3, padding= 2, dilation=2), nn.BatchNorm1d(16), nn.PReLU(),   #dilated
            nn.Conv1d(16, 16, 3, padding=1), nn.BatchNorm1d(16), nn.PReLU(),
            nn.Conv1d(16, 16, 3, padding= 4, dilation=4), nn.BatchNorm1d(16), nn.PReLU(),   #dilated
            nn.Conv1d(16, 16, 3, padding=1), nn.BatchNorm1d(16), nn.PReLU(),
            nn.Conv1d(16, 16, 3, padding= 8, dilation=8), nn.BatchNorm1d(16), nn.PReLU(),   #dilated
            nn.Conv1d(16, 16, 3, padding=1), nn.BatchNorm1d(16), nn.PReLU(),
            nn.Conv1d(16, 16, 3, padding= 16, dilation=16), nn.BatchNorm1d(16), nn.PReLU(),   #dilated
            nn.Conv1d(16, 16, 3, padding=1), nn.BatchNorm1d(16), nn.PReLU(),
            nn.Conv1d(16, 16, 3, padding= 32, dilation=32), nn.BatchNorm1d(16), nn.PReLU(),   #dilated
            nn.Conv1d(16, 16, 3, padding=1), nn.BatchNorm1d(16), nn.PReLU(),
            nn.Conv1d(16, 16, 3, padding= 64, dilation=64), nn.BatchNorm1d(16), nn.PReLU(),   #dilated
            nn.Conv1d(16, 16, 3, padding=1), nn.BatchNorm1d(16), nn.PReLU(),
            nn.Conv1d(16, 16, 3, padding= 128, dilation=128), nn.BatchNorm1d(16), nn.PReLU(),   #dilated
            nn.Conv1d(16, 16, 3, padding=1), nn.BatchNorm1d(16), nn.PReLU(),
            nn.Conv1d(16, 16, 3, padding= 256, dilation=256), nn.BatchNorm1d(16), nn.PReLU(),   #dilated
            nn.Conv1d(16, 16, 3, padding=1), nn.BatchNorm1d(16), nn.PReLU(),
            nn.Conv1d(16, 16, 3, padding= 512, dilation=512), nn.BatchNorm1d(16), nn.PReLU(),   #dilated
            nn.Conv1d(16, 16, 3, padding=1), nn.BatchNorm1d(16), nn.PReLU(),
            
        )
        self.Projection = nn.Conv1d(16, 1, 1)

    def forward(self, x):
        x = self.block(x)
        x = self.Projection(x)
        return x
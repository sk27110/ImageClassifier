import torch
import torch.nn as nn


class ResNetModel(nn.Module):
    def __init__(self, num_clasees):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_shannels = 64, kernel_size = 7, padding = 3, stride = 2)
        )

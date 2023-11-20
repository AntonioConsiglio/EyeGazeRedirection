import torch
from typing import Tuple
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self,in_channels: int,
                    out_channels: int,
                    kernel_size: Tuple[int],
                    stride: Tuple[int]= 1,
                    padding: Tuple[int] | str = 0,
                    bias: bool = False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.istn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = self.relu(self.istn(self.conv(x)))
        return out


class DeConvBNReLU(nn.Module):
    def __init__(self,in_channels: int,
                    out_channels: int,
                    kernel_size: Tuple[int],
                    stride: Tuple[int]= 1,
                    padding: Tuple[int] | str = 0,
                    bias: bool = False):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.istn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        return self.relu(self.istn(self.conv(x)))

class ConvLReLU(nn.Module):
    def __init__(self,in_channels: int,
                    out_channels: int,
                    kernel_size: Tuple[int],
                    stride: Tuple[int]= 1,
                    padding: Tuple[int] | str = 0,
                    bias: bool = False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.relu = nn.LeakyReLU()
    
    def forward(self,x):
        return self.relu(self.conv(x))
    

class ResBlock(nn.Module):
    def __init__(self,in_channels: int,
                    out_channels: int,
                    kernel_size: Tuple[int],
                    stride: Tuple[int]= 1,
                    padding: Tuple[int] | str = 0,
                    bias: bool = False):
        super().__init__()
        self.convbnrel = ConvBNReLU(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias)
        self.conv = nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=bias)
        self.instancen = nn.InstanceNorm2d(out_channels)
    
    def forward(self,x):

        residual = x

        out = self.convbnrel(x)
        out = self.instancen(self.conv(out))

        return out+residual


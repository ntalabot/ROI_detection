#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions/classes for model manipulation with PyTorch.
Created on Mon Oct 22 13:54:19 2018

@author: nicolas
"""

import torch


def weights_initialization(model):
    """Initialize the weights of the given PyTorch model."""
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


#class CustomNet(torch.nn.Module):
#    """Definition of a custom written model."""
#    def __init__(self, shape, activation=torch.nn.ReLU, device=torch.device("cpu")):
#        """Initialize the model."""
#        super(CustomNet, self).__init__()
#        self.height, self.width, self.in_channels = shape
#        self.activation = activation
#        
#        # Initialize the networks modules
#        self.conv1 = torch.nn.Conv2d(self.in_channels, 4, 3, stride=3, padding=1)
#        self.bn1 = torch.nn.BatchNorm2d(4)
#        self.conv2 = torch.nn.Conv2d(4, 8, 3, stride=3, padding=1)
#        self.bn2 = torch.nn.BatchNorm2d(8)
#        self.conv_t1 = torch.nn.ConvTranspose2d(8, 4, 3, stride=3, padding=1, output_padding=(1,0))
#        self.bn3 = torch.nn.BatchNorm2d(4)
#        self.conv_t2 = torch.nn.ConvTranspose2d(4, 1, 3, stride=3, padding=(2, 0), output_padding=(1,1))
#        
#        # Initialize weights
#        weights_initialization(self)
#        
#        # Make sure the model is in the correct device
#        self.to(device)
#
#    def forward(self, x):
#        """Perform the forward pass."""
#        x = self.activation(self.conv1(x))
#        x = self.bn1(x)
#        x = self.activation(self.conv2(x))
#        x = self.bn2(x)
#        x = self.activation(self.conv_t1(x))
#        x = self.bn3(x)
#        logits = self.conv_t2(x)
#        logits = logits.view(-1, self.height, self.width)
#        return logits
#    
#    def to(self, *args, **kwargs):
#        """Modifiy model.device and call Module.to()."""
#        device, _, _ = torch._C._nn._parse_to(*args, **kwargs)
#        self.device = device
#        return super(CustomNet, self).to(*args, **kwargs)


class UNetConv(torch.nn.Module):
    """U-Net like convolution block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, 
                 activation=torch.nn.ReLU(), batchnorm=False):
        super(UNetConv, self).__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        if self.batchnorm:
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        if self.batchnorm:
            self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        if self.batchnorm:
            x = self.bn1(x)
        out = self.activation(self.conv2(x))
        if self.batchnorm:
            out = self.bn2(out)
        return out


class UNetUpConv(torch.nn.Module):
    """U-Net like up-convolution block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 activation=torch.nn.ReLU(), batchnorm=False):
        super(UNetUpConv, self).__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        
        self.upconv = torch.nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        if self.batchnorm:
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        if self.batchnorm:
            self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x, bridge):
        x = self.upconv(x)
        out = torch.cat([x, bridge], 1)
        
        out = self.activation(self.conv1(out))
        if self.batchnorm:
            out = self.bn1(out)
        out = self.activation(self.conv2(out))
        if self.batchnorm:
            out = self.bn2(out)
        return out
   

class CustomUNet(torch.nn.Module):
    """
    Definition of a U-Net like network. 
    
    NB: Depth and initial number of channels are tunable, zero-padding is used
    so that input and output are of the same size, optional batch normalization
    layers.
    
    Args:
        in_channels: int
            Number of channels of the input images.
        u_depth: int (default = 1)
            Depth of the network's U-shape (should be > 0). I.e., there will
            be `u_depth` MaxPooling, and `u_depth` transposed convolutions.
        out1_channels: int (default = 8)
            Number of output channels after the first convolutional block.
            Note that after this, each ConvBlock doubles the channels, and
            each UpConvBlock halves the channels.
        activation: PyTorch activation function (default = torch.nn.ReLU())
            The non-linear activation to apply after each convolution.
            Note that with the current implementation, this one is reused 
            everywhere. Therefore, it cannot have any learnable parameters.
        batchnorm: bool (default = True)
            If True, the network will have a batch normalization layer after
            each convolution (and after the non-linearity).
        device: PyTorch device (default = torch.device("cpu"))
            Device to which the model is to be placed
    """
    def __init__(self, in_channels, u_depth=1, out1_channels=8,
                 activation=torch.nn.ReLU(), batchnorm=True,
                 device=torch.device("cpu")):
        """Initialize the model (see class docstring for arguments description)."""
        super(CustomUNet, self).__init__()
        self.in_channels = in_channels
        self.activation = activation
        self.maxpool = torch.nn.MaxPool2d(2)
        if u_depth < 1:
            raise ValueError("Depth of the U-Net has to be at least 1 (given u_depth=%d)" % u_depth)
        
        # Initialize the networks modules
        self.convs = torch.nn.ModuleList()
        self.convs.append(UNetConv(self.in_channels, out1_channels, activation=activation, batchnorm=batchnorm))
        for i in range(1, u_depth):
            self.convs.append(UNetConv(out1_channels * (2 ** (i-1)), out1_channels * (2 ** i), 
                                       activation=activation, batchnorm=batchnorm))
            
        self.midconv = UNetConv(out1_channels * (2 ** (u_depth-1)), out1_channels * (2 ** u_depth),
                                activation=activation, batchnorm=batchnorm)
        
        self.upconvs = torch.nn.ModuleList()
        for i in range(u_depth, 0, -1):
            self.upconvs.append(UNetUpConv(out1_channels * (2 ** i), out1_channels * (2 ** (i-1)), 
                                           activation=activation, batchnorm=batchnorm))
            
        self.outconv = torch.nn.Conv2d(out1_channels, 1, 1)
        
        # Initialize weights
        weights_initialization(self)
        
        # Make sure the model is in the correct device
        self.to(device)

    def forward(self, x):
        """Perform the forward pass."""
        x_convs = []
        x_pool = x
        for i in range(len(self.convs)):
            x_convs.append(self.convs[i](x_pool))
            x_pool = self.maxpool(x_convs[i])
        
        x_mid = self.midconv(x_pool)
        
        x_up = self.upconvs[0](x_mid, x_convs.pop(-1))
        for i in range(1, len(self.upconvs)):
            x_up = self.upconvs[i](x_up, x_convs.pop(-1))
            
        logits = self.outconv(x_up)
        logits.squeeze_(dim=1)
        return logits
    
    def to(self, *args, **kwargs):
        """Send the model to the device, and modify accordingly its `device` attribute."""
        output = super(CustomUNet, self).to(*args, **kwargs)
        device, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.device = device
        return output
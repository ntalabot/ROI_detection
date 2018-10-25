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


class CustomNet(torch.nn.Module):
    def __init__(self, shape):
        """Initialize the model."""
        super(CustomNet, self).__init__()
        self.height, self.width, self.n_channels_in = shape
        
        # Initialize the networks modules
        self.conv1 = torch.nn.Conv2d(self.n_channels_in, 8, 3, stride=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv_t1 = torch.nn.ConvTranspose2d(8, 1, 3, stride=3, padding=1, output_padding=(2,0))
        
        # Initialize weights
        weights_initialization(self)

    def forward(self, x):
        """Perform the forward pass."""
        x = self.conv1(x).clamp(min=0)
        x = self.bn1(x)
        logits = self.conv_t1(x)
        logits = logits.view(-1, self.height, self.width)
        return logits
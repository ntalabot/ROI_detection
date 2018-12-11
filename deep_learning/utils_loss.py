#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for computing losses and metrics with PyTorch.
Created on Tue Nov 27 13:57:06 2018

@author: nicolas
"""

import torch

from utils_common.metrics import dice_coef, crop_metric


def get_crop_loss(loss_fn, scale=4.0, reduction="elementwise_mean", device=torch.device("cpu")):
    """
    Return a loss function that works on cropped targets' connected regions.
    /!\ Note: 
        1. it requires to send the tensor to the cpu for numpy!
        2. the return function can be used as a metric, but no guarantee that 
        it works as a loss for the training!
    
    Size of the cropped region will be bounding_box * scale.
    
    Use this with any loss function (that will be called as loss(predictions, targets))
    that you want to use on the cropped targets' connected regions. This will
    return a callable function as you would get with loss_class, but that automatically
    searches for these connected region.
    """
    # Create a subsitute loss function that makes sure its input are tensors
    to_torch_loss_fn = lambda pred, target, mask: loss_fn(torch.from_numpy(pred[mask]).to(device),
                                                          torch.from_numpy(target[mask]).to(device))
    
    return lambda preds, targets, masks: torch.tensor(crop_metric(
            to_torch_loss_fn, 
            preds.cpu().detach().numpy(), 
            targets.cpu().detach().numpy(), 
            masks=masks.cpu().detach().numpy(),
            scale=scale, 
            reduction=reduction))
   
def get_dice_metric(reduction='mean'):
    """Return a metric function that computes the dice coefficient."""
    return lambda preds, targets, masks: torch.tensor(
            dice_coef((torch.sigmoid(preds.cpu()) > 0.5).detach().numpy(),
                      targets.cpu().detach().numpy(),
                      masks=masks.cpu().detach().numpy(),
                      reduction=reduction))

def get_crop_dice_metric(scale=4.0, reduction='mean'):
    """Return a metric function that computes the cropped dice coefficient."""
    return lambda preds, targets, masks: torch.tensor(
            crop_metric(dice_coef,
                        (torch.sigmoid(preds.cpu()) > 0.5).detach().numpy(),
                        targets.cpu().detach().numpy(),
                        masks=masks.cpu().detach().numpy(),
                        scale=scale,
                        reduction=reduction))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful metric functions for ROI detection.
Created on Mon Nov 12 09:37:00 2018

@author: nicolas
"""

import numpy as np
from skimage import measure


def loss_mae(predictions, targets, reduction='mean'):
    """Compute the (Mean) Average Error between predictions and targets."""
    if reduction in ["elementwise_mean", "mean", "ave", "average"]:
        return np.abs(targets[:] - predictions[:]).mean()
    elif reduction in ["sum"]:
        return np.abs(targets[:] - predictions[:]).sum()
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)

def loss_l2(predictions, targets, reduction='mean'):
    """Compute the L2-norm loss between predictions and targets."""
    loss = 0.0
    for i in range(len(targets)):
        loss += np.linalg.norm(targets[i] - predictions[i])
    
    if reduction in ["elementwise_mean", "mean", "ave", "average"]:
        return loss / len(targets)
    elif reduction in ["sum"]:
        return loss
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)

def dice_coef(predictions, targets, reduction='mean'):
    """Compute the Dice coefficient between predictions and targets."""
    dice = 0.0
    for i in range(len(targets)):
        total_pos = targets[i].sum() + predictions[i].sum()
        if total_pos == 0: # No true positive, and no false positive --> correct
            dice += 1.0
        else:
            dice += 2.0 * np.logical_and(targets[i], predictions[i]).sum() / total_pos
    
    if reduction in ["elementwise_mean", "mean", "ave", "average"]:
        return dice / len(targets)
    elif reduction in ["sum"]:
        return dice
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)

def crop_metric(metric_fn, predictions, targets, scale=4.0, reduction='mean'):
    """
    Compute the metric around the cropped targets' connected regions.
    
    Size of the cropped region will be bounding_box * scale.
    """
    metric = 0.0
    n_no_positive = 0 # number of target with no positive pixels (fully background)
    for i in range(len(targets)):
        labels = measure.label(targets[i])
        # If no true positive region, does not consider the image
        if labels.max() == 0:
            n_no_positive += 1.0
            continue
        regionprops = measure.regionprops(labels)
        
        # Loop over targets' connected regions
        for region in regionprops:
            min_row, min_col, max_row, max_col = region.bbox
            height = max_row - min_row
            width = max_col - min_col
            max_row = int(min(targets[i].shape[0], max_row + height * (scale-1) / 2))
            min_row = int(max(0, min_row - height * (scale-1) / 2))
            max_col = int(min(targets[i].shape[1], max_col + width * (scale-1) / 2))
            min_col = int(max(0, min_col - width * (scale-1) / 2))
            
            metric += metric_fn(np.array([predictions[i][min_row:max_row, min_col:max_col]]), 
                                np.array([targets[i][min_row:max_row, min_col:max_col]])) / \
                      len(regionprops) # Averaging factor for multiple region in same image
    
    if reduction in ["elementwise_mean", "mean", "ave", "average"]:
        return metric / (len(targets) - n_no_positive)
    elif reduction in ["sum"]:
        return metric
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)
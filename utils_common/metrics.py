#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful metric functions for ROI detection.
Created on Mon Nov 12 09:37:00 2018

@author: nicolas
"""

import numpy as np
from skimage import measure


def loss_mae(predictions, targets, masks=None, reduction='mean'):
    """Compute the (Mean) Average Error between predictions and targets."""
    if masks is None:
        masks = np.ones(targets.shape)
    masks = masks.astype(np.bool) 
        
    if reduction in ["elementwise_mean", "mean", "ave", "average"]:
        return np.abs(targets[masks] - predictions[masks]).mean()
    elif reduction in ["sum"]:
        return np.abs(targets[masks] - predictions[masks]).sum()
    elif reduction in ["array", "no_reduction", "full"]:
        return np.abs(targets[masks] - predictions[masks])
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)

def loss_l2(predictions, targets, masks=None, reduction='mean'):
    """Compute the L2-norm loss between predictions and targets."""
    if masks is None:
        masks = np.ones(targets.shape)
    masks = masks.astype(np.bool) 
        
    loss = []
    for i in range(len(targets)):
        loss.append(np.linalg.norm(targets[i][masks[i]] - predictions[i][masks[i]]))
    
    if reduction in ["elementwise_mean", "mean", "ave", "average"]:
        return np.mean(loss)
    elif reduction in ["sum"]:
        return np.sum(loss)
    elif reduction in ["array", "no_reduction", "full"]:
        return np.array(loss)
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)

def dice_coef(predictions, targets, masks=None, reduction='mean'):
    """Compute the Dice coefficient between predictions and targets."""
    if masks is None:
        masks = np.ones(targets.shape)
    masks = masks.astype(np.bool) 
    
    dice = []
    for i in range(len(targets)):
        total_pos = targets[i][masks[i]].sum() + predictions[i][masks[i]].sum()
        if total_pos == 0: # No true positive, and no false positive --> correct
            dice.append(1.0)
        else:
            dice.append(2.0 * np.logical_and(targets[i][masks[i]], predictions[i][masks[i]]).sum() / total_pos)
    
    if reduction in ["elementwise_mean", "mean", "ave", "average"]:
        return np.mean(dice)
    elif reduction in ["sum"]:
        return np.sum(dice)
    elif reduction in ["array", "no_reduction", "full"]:
        return np.array(dice)
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)

def crop_metric(metric_fn, predictions, targets, masks=None, scale=4.0, reduction='mean'):
    """
    Compute the metric around the cropped targets' connected regions.
    
    Size of the cropped region will be bounding_box * scale.
    """
    metric = []
    n_no_positive = 0 # number of target with no positive pixels (fully background)
    for i in range(len(targets)):
        labels = measure.label(targets[i])
        # If no true positive region, does not consider the image
        if labels.max() == 0:
            n_no_positive += 1.0
            continue
        regionprops = measure.regionprops(labels)
        
        local_metric = 0.0
        # Loop over targets' connected regions
        for region in regionprops:
            min_row, min_col, max_row, max_col = region.bbox
            height = max_row - min_row
            width = max_col - min_col
            max_row = int(min(targets[i].shape[0], max_row + height * (scale-1) / 2))
            min_row = int(max(0, min_row - height * (scale-1) / 2))
            max_col = int(min(targets[i].shape[1], max_col + width * (scale-1) / 2))
            min_col = int(max(0, min_col - width * (scale-1) / 2))
            
            if masks is None:
                local_mask = None
            else:
                local_mask = np.array([masks[i][min_row:max_row, min_col:max_col]], dtype=np.bool)
                
            local_metric += metric_fn(np.array([predictions[i][min_row:max_row, min_col:max_col]]), 
                                np.array([targets[i][min_row:max_row, min_col:max_col]]),
                                local_mask) / \
                            len(regionprops) # Averaging factor for multiple region in same image
        metric.append(local_metric)
    
    if reduction in ["elementwise_mean", "mean", "ave", "average"]:
        return np.sum(metric) / (len(targets) - n_no_positive)
    elif reduction in ["sum"]:
        return np.sum(metric)
    elif reduction in ["array", "no_reduction", "full"]:
        return np.array(metric)
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)
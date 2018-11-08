#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for detecting ROIs on stacks.
Created on Mon Oct  1 17:48:29 2018

@author: nicolas
"""

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, io, color, measure
from skimage import morphology as morph
from skimage.morphology import square, disk, diamond


## Stacks/images manipulations
def imread_to_float(filename, scaling=None, return_scaling=False):
    """Return the loaded stack/image from filename, casted to float32 and rescaled."""
    stack = io.imread(filename)
    # If no scaling is precised, use max value of stack
    if scaling is None:
        scaling = stack.max()
    stack = stack.astype(np.float32) / scaling
    if return_scaling:
        return stack, scaling
    else:
        return stack

def to_npint(stack, dtype=np.uint8, scaling=None):
    """Scale and cast the stack/image to dtype (should be ndarray integer)."""
    # If no scaling is precised, use max range of 16 bits
    if scaling is None:
        scaling = np.iinfo(dtype).max
    
    stack_int = (stack * scaling).astype(dtype)
    return stack_int

def overlay_mask(image, mask, opacity=0.25, mask_color=[1.0, 0.0, 0.0]):
    """Merge the mask as an overlay over the image."""
    mask_color = np.array(mask_color)
    if image.ndim == 2:
        overlay = color.gray2rgb(image)
    else:
        overlay = image.copy()
        
    overlay[mask.astype(np.bool), :] *= 1 - opacity
    overlay[mask.astype(np.bool), :] += mask_color * opacity
    return overlay

def overlay_mask_stack(stack, mask, opacity=0.25, mask_color=[1.0, 0.0, 0.0]):
    """Merge the mask as an overlay over the stack."""
    mask_color = np.array(mask_color)
    if stack.ndim == 3:
        overlay = color.gray2rgb(stack)
    else:
        overlay = stack.copy()
        
    for i in range(len(stack)):
        overlay[i] = overlay_mask(overlay[i], mask[i], opacity=opacity, mask_color=mask_color)
    return overlay


## Processing
def hline(length):
    """Horizontal line element for morpholgical operations."""
    selem = np.zeros((length, length), dtype=np.uint8)
    selem[int(length / 2), :] = 1
    return selem

def vline(length):
    """Vertical line element for morpholgical operations."""
    selem = np.zeros((length, length), dtype=np.uint8)
    selem[:, int(length / 2)] = 1
    return selem

def identity(stack, selem=None):
    """Identity function, return the same stack."""
    return stack

def median_filter(stack, selem=None):
    """Apply median filtering to all images in the stack."""
    # Median works with uint8 or 16
    if stack.dtype in [np.uint8, np.uint16]:
        median_type = stack.dtype
    else:
        median_type = np.uint8
    
    # Median filtering
    filtered_stack = np.zeros(stack.shape, dtype=median_type)
    for i in range(len(stack)):
        filtered_stack[i] = filters.median(stack[i], selem=selem)
        
    # Cast back if needed
    if median_type != stack.dtype:
        # if uint32 or 64, can simply cast
        if isinstance(stack.dtype, np.unsignedinteger):
            filtered_stack = filtered_stack.astype(stack.dtype)
        # if float, change range to [0,1]
        elif stack.dtype in [np.float16, np.float32, np.float64, np.float128]:
            filtered_stack = (filtered_stack / 255).astype(stack.dtype)
        else:
            print("Warning: Unable to cast back to %s after median filtering. "
                  "Returning an np.uint8 array." % stack.dtype)
    
    return filtered_stack

def morph_open(stack, selem=None):
    """Apply morphological opening to all images in the stack."""   
    filtered_stack = np.zeros(stack.shape, dtype=stack.dtype)
    
    for i in range(len(stack)):
        filtered_stack[i] = morph.opening(stack[i], selem=selem)
        
    return filtered_stack

def preprocess_stack(stack):
    """Apply the preprocessing function to the stack."""
    filtered_stack = median_filter(stack, disk(1))
    return morph_open(filtered_stack, disk(1))

def hysteresis_threshold(image, low, high):
    """Apply hysteresis threshold to the image/stack. Taken from skimage module."""
    low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
    mask_low = image > low
    mask_high = image > high
    # Connected components of mask_low
    labels_low, num_labels = ndi.label(mask_low)
    # Check which connected components contain pixels from mask_high
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]
    return thresholded


# Loss/metrics for comparing predictions with ground truths
def loss_mae(predictions, targets, reduction='mean'):
    """Compute the (Mean) Average Error between predictions and targets."""
    if reduction in ["mean", "ave", "average"]:
        return np.abs(targets[:] - predictions[:]).mean()
    elif reduction in ["sum"]:
        return np.abs(targets[:] - predictions[:]).sum()
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)

def loss_l2_segmentation(predictions, targets, reduction='mean'):
    """Compute the L2-norm loss between predicted and target ROI segmentations."""
    loss = 0.0
    for i in range(len(targets)):
        loss += np.linalg.norm(targets[i] - predictions[i])
    
    if reduction in ["mean", "ave", "average"]:
        return loss / len(targets)
    elif reduction in ["sum"]:
        return loss
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)

def dice_coef(predictions, targets, reduction='mean'):
    """Compute the Dice coefficient between predicted and target ROI segmentations."""
    loss = 0.0
    for i in range(len(targets)):
        total_pos = targets[i].sum() + predictions[i].sum()
        if total_pos == 0: # No true positive, and no false positive --> correct
            loss += 1.0
        else:
            loss += 2.0 * np.logical_and(targets[i], predictions[i]).sum() / total_pos
    
    if reduction in ["mean", "ave", "average"]:
        return loss / len(targets)
    elif reduction in ["sum"]:
        return loss
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)

def crop_dice_coef(predictions, targets, scale=4.0, reduction='mean'):
    """Compute the Dice coefficient around  the cropped targets' ROI.
    
    Height and width of the ROI cropping will be increased by scale.
    """
    loss = 0.0
    for i in range(len(targets)):
        labels = measure.label(targets[i])
        regionprops = measure.regionprops(labels)
        # Loop over target ROI
        for region in regionprops:
            min_row, min_col, max_row, max_col = region.bbox
            height = max_row - min_row
            width = max_col - min_col
            max_row = int(min(targets[i].shape[0], max_row + height * (scale-1) / 2))
            min_row = int(max(0, min_row - height * (scale-1) / 2))
            max_col = int(min(targets[i].shape[1], max_col + width * (scale-1) / 2))
            min_col = int(max(0, min_col - width * (scale-1) / 2))
            
            loss += 2.0 * np.logical_and(targets[i][min_row:max_row, min_col:max_col],
                                         predictions[i][min_row:max_row, min_col:max_col]).sum() / \
                    (targets[i][min_row:max_row, min_col:max_col].sum() + \
                     predictions[i][min_row:max_row, min_col:max_col].sum()) / \
                    len(regionprops) # Averaging factor for multiple ROI in same image
    
    if reduction in ["mean", "ave", "average"]:
        return loss / len(targets)
    elif reduction in ["sum"]:
        return loss
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)
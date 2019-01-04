#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for loading/saving/displaying stacks and images.
Created on Mon Oct  1 17:48:29 2018

@author: nicolas
"""

import numpy as np
from skimage import io, color, measure


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

def to_npint(stack, dtype=np.uint8, float_scaling=None):
    """Scale and cast the stack/image to dtype."""
    # If already correct type, do nothing
    if stack.dtype == dtype:
        return stack
        
    if issubclass(stack.dtype.type, np.floating):
        # If no scaling is precised, use max range of given type
        if float_scaling is None:
            float_scaling = np.iinfo(dtype).max
        stack_int = (stack * float_scaling).astype(dtype)
    elif stack.dtype == np.bool:
        # If boolean, we set 1 to max range of dtype (e.g., 255 for np.uint8)
        stack_int = stack.astype(dtype) * np.iinfo(dtype).max
    else:
        stack_int = stack.astype(dtype)
    return stack_int

def gray2red(image):
    """Create an RGB image with image in the red channel, and 0 in the others.
    /!\ It does not check for grayscale images in order to work with stacks!"""
    red = image
    green = np.zeros_like(image)
    blue = np.zeros_like(image)
    return np.stack([red, green, blue], axis=-1)

def overlay_mask(image, mask, opacity=0.25, mask_color=[1.0, 0.0, 0.0], rescale_img=False):
    """Merge the mask as an overlay over the image."""
    mask_color = np.array(mask_color, dtype=np.float32)
    if image.ndim == 2:
        overlay = color.gray2rgb(image)
    else:
        overlay = image.copy()
        
    if rescale_img:
        overlay /= overlay.max()
        
    overlay[mask.astype(np.bool), :] *= 1 - opacity
    overlay[mask.astype(np.bool), :] += mask_color * opacity
    return overlay

def overlay_mask_stack(stack, mask, opacity=0.25, mask_color=[1.0, 0.0, 0.0], rescale_img=False):
    """Merge the mask as an overlay over the stack."""
    mask_color = np.array(mask_color, dtype=np.float32)
    if stack.ndim == 3:
        overlay = color.gray2rgb(stack)
    else:
        overlay = stack.copy()
        
    for i in range(len(stack)):
        overlay[i] = overlay_mask(overlay[i], mask[i], opacity=opacity, 
               mask_color=mask_color, rescale_img=rescale_img)
    return overlay

def overlay_preds_targets(predictions, targets, masks=None):
    """Create an image with prediction and target (and mask) for easy comparison."""
    # Select correct overlay function in order to work with image, and stack
    if predictions.ndim == 3:
        overlay_fn = overlay_mask_stack
    else:
        overlay_fn = overlay_mask
    
    # Add predicted annotations as green
    correct = overlay_fn(predictions, np.logical_and(predictions, targets), 
                         opacity=1, mask_color=[0,1,0])
    # Add missed annotations as red
    incorrect = overlay_fn(correct, np.logical_and(targets, np.logical_not(predictions)), 
                           opacity=1, mask_color=[1,0,0])
    if masks is None:
        final = incorrect
    else:
        final = overlay_fn(incorrect, masks, opacity=0.5, mask_color=[1,1,0])
    return final

def overlay_contours(image, mask, rescale_img=False):
    """Put the contours of mask over image."""
    contour = np.zeros(mask.shape, dtype=np.bool)
    coords = measure.find_contours(mask, 0.5) # 0.5 to work with both float and int
    for coord in coords:
        rows = np.rint(coord[:,0]).astype(np.int)
        cols = np.rint(coord[:,1]).astype(np.int)
        contour[rows, cols] = True
    return overlay_mask(image, contour, opacity=1.0, mask_color=[1,1,1], rescale_img=rescale_img)

def overlay_contours_stack(stack, mask, rescale_img=False):
    """Put the contours of mask over image."""
    contour = np.zeros(mask.shape, dtype=np.bool)
    for i in range(len(mask)):
        coords = measure.find_contours(mask[i], 0.5) # 0.5 to work with both float and int
        for coord in coords:
            rows = np.rint(coord[:,0]).astype(np.int)
            cols = np.rint(coord[:,1]).astype(np.int)
            contour[i, rows, cols] = True
    return overlay_mask_stack(stack, contour, opacity=1.0, mask_color=[1,1,1], rescale_img=rescale_img)
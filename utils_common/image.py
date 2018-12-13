#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for loading/saving stacks.
Created on Mon Oct  1 17:48:29 2018

@author: nicolas
"""

import numpy as np
from skimage import io, color


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
    """Scale and cast the stack/image to dtype (assumes stack is float)."""
    # If no scaling is precised, use max range of given type
    if scaling is None:
        scaling = np.iinfo(dtype).max
    
    stack_int = (stack * scaling).astype(dtype)
    return stack_int

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
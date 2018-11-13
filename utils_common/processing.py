#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for processing stacks.
Created on Mon Nov 12 09:28:07 2018

@author: nicolas
"""

import warnings
import numpy as np
from skimage import filters
from skimage import morphology as morph
from skimage.morphology import disk


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
    
    # Median filtering with disabled warnings
    filtered_stack = np.zeros(stack.shape, dtype=median_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

# TODO: delete after checking that skimage.filters.apply_hysteresis_threshold is out
#def hysteresis_threshold(image, low, high):
#    """Apply hysteresis threshold to the image/stack. Taken from skimage module."""
#    low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
#    mask_low = image > low
#    mask_high = image > high
#    # Connected components of mask_low
#    labels_low, num_labels = ndi.label(mask_low)
#    # Check which connected components contain pixels from mask_high
#    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
#    connected_to_high = sums > 0
#    thresholded = connected_to_high[labels_low]
#    return thresholded
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains useful functions for detecting ROI with computer vision.
Created on Mon Dec 10 14:52:18 2018

@author: nicolas
"""


import numpy as np
import cv2
from skimage import measure, filters
from skimage import morphology as morph
from skimage.morphology import disk

from utils_common.image import to_npint


def cv_detect(rgb_stack):
    """Use computer vision to detect ROI in given RGB stack."""
    min_area = 5
    temporal_window_size = 5
    search_window_size = 21
    
    stack = (rgb_stack[...,0] + rgb_stack[...,1]) / 3 # Mean of channels (assuming 0 in blue channel)
    stack = to_npint(stack)
    # Loop the stack so that masks can be made for first and last images
    loop_stack = np.concatenate((stack[- (temporal_window_size - 1)//2:], 
                                 stack, 
                                 stack[:(temporal_window_size - 1)//2]))
    output = np.zeros(stack.shape, dtype=np.bool)
    for i in range(len(stack)):
        # Denoising
        denoised = cv2.fastNlMeansDenoisingMulti(loop_stack, i + (temporal_window_size - 1)//2, 
                                                 temporal_window_size, None, 4, 7, search_window_size)
        
        # Segmentation
        denoised_pp = filters.gaussian(denoised, sigma=2)
        seg = denoised_pp > filters.threshold_otsu(denoised_pp)
        seg = morph.erosion(seg, selem=disk(1))
        
        # Mask creation
        mask = np.zeros_like(seg)
        labels = measure.label(seg)
        for region in measure.regionprops(labels):
            if region.area > min_area: # discard small elements
                rows, cols = region.coords.T
                mask[rows, cols] = 1
        
        output[i] = mask
    return output
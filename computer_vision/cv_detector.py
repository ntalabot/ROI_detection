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
from utils_common.multiprocessing import run_parallel
from utils_common.register_cc import register_stack, shift_image


def cv_detect(rgb_stack, thresholding_fn=filters.threshold_otsu, 
              registration=False, selem=disk(1)):
    """Use computer vision to detect ROI in given RGB stack."""
    min_area = 6
    temporal_window_size = 5
    search_window_size = 21
    h_red = 11
    h_green = 11
    
    stack = to_npint(rgb_stack)
    if registration:
        stack, reg_rows, reg_cols = register_stack(stack, channels=[0,1], return_shifts=True)
    
    # Loop the stack so that masks can be made for first and last images
    loop_stack = np.concatenate((stack[- (temporal_window_size - 1)//2:], 
                                 stack, 
                                 stack[:(temporal_window_size - 1)//2]))
    
    # Denoise each channel
    def denoise_stack(channel_num, h_denoise):
        """Denoise selected channel from loop_stack (function used for parallelization)."""
        loop_channel = loop_stack[..., channel_num]
        denoised = np.zeros_like(loop_channel)
        for i in range(len(stack)):
            denoised[i] = cv2.fastNlMeansDenoisingMulti(loop_channel, i + (temporal_window_size - 1)//2, 
                    temporal_window_size, None, h_denoise, 7, search_window_size)
        return denoised
    
    denoised_r, denoised_g = run_parallel(
        lambda: denoise_stack(0, h_red),
        lambda: denoise_stack(1, h_green)
    )
    denoised = np.maximum(denoised_r, denoised_g)
    
    output = np.zeros(stack.shape[:-1], dtype=np.bool)
    for i in range(len(stack)):
        # Segmentation
        denoised_pp = filters.gaussian(denoised[i], sigma=2)
        seg = denoised_pp > thresholding_fn(denoised_pp)
        seg = morph.erosion(seg, selem=selem)
        
        # Mask creation
        mask = seg.copy()
        labels = measure.label(mask)
        for region in measure.regionprops(labels):
            if region.area < min_area: # discard small elements
                rows, cols = region.coords.T
                mask[rows, cols] = 0
        # Shift back the mask if the stack was registered
        if registration:
            mask = shift_image(mask, -reg_rows[i], -reg_cols[i])
        
        output[i] = mask
    return output
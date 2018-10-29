#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing functions used for registration with cross-correlation.
Created on Mon Oct 29 10:13:11 2018

@author: nicolas
"""

import numpy as np


def _compute_shifts(ref_image, shifted_image, return_error=False):
    """
    col_shift, row_shift, phase_shift, [error] = 
        compute_shifts(ref_image, shifted_image, return_error=False)
    
    Translated from MATLAB code written by Manuel Guizar
    downloaded from http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation
    """
    # Compute FFTs, and cross-correlation
    buf1ft = np.fft.fft2(ref_image)
    buf2ft = np.fft.fft2(shifted_image)
    m, n = buf1ft.shape
    CC = np.fft.ifft2(buf1ft * buf2ft.conjugate())

    # Look for max cross-correlation
    max1 = np.max(CC, axis=0)
    loc1 = np.argmax(CC, axis=0)
    cloc = np.argmax(max1)

    rloc = loc1[cloc]
    CCmax = CC[rloc, cloc]
    
    if return_error:
        rfzero = np.sum(np.abs(buf1ft.ravel()) ** 2) / (m * n)
        rgzero = np.sum(np.abs(buf2ft.ravel()) ** 2) / (m * n)

        error = 1.0 - CCmax * CCmax.conjugate() / (rgzero * rfzero)
        error = np.sqrt(np.abs(error));

    # Compute shifts
    phase_shift = np.angle(CCmax); # probably useless as always 0 for non-negative images

    md2 = np.fix(m/2.0) #should this be float?
    nd2 = np.fix(n/2.0)
    if rloc > md2:
        row_shift = rloc - m; #CHECK!
    else:
        row_shift = rloc;#CHECK!

    if cloc > nd2:
        col_shift = cloc - n;#CHECK!
    else:
        col_shift = cloc;#CHECK!
    
    if return_error:
        output = [col_shift, row_shift, phase_shift, error]
    else:
        output = [col_shift, row_shift, phase_shift]
        
    return output


def _register_image(shifted_image, col_shift, row_shift, phase_shift):
    """Register the shifted image by the given row, col and phase shifts"""
    buf2ft = np.fft.fft2(shifted_image)
    nr, nc = buf2ft.shape

    # Compute registered version of buf2ft
    Nr = np.fft.ifftshift(np.arange(-np.fix(nr/2.0), np.ceil(nr/2.0)));
    Nc = np.fft.ifftshift(np.arange(-np.fix(nc/2.0), np.ceil(nc/2.0)));
    [Nc, Nr] = np.meshgrid(Nc, Nr)
    
    greg = buf2ft * np.exp(2j * np.pi * (-row_shift * Nr/nr - col_shift * Nc/nc));
    greg = greg * np.exp(1j * phase_shift);

    if np.can_cast(np.float32, shifted_image.dtype): # need to check this, too
        registered_image = np.abs(np.fft.ifft2(greg))
    else:
        registered_image = np.round(np.abs(np.fft.ifft2(greg))).astype(shifted_image.dtype)

    return registered_image


def register_stack(stack, ref_num=0, channels=[0,1]):
    """Register the stack using cross-correlation."""
    reg_stack = np.zeros(stack.shape, dtype=stack.dtype)
    
    # Compute the stack to process, and reference image
    if stack.ndim == 3: # for grayscale images
        stack_to_reg = stack
    else: # for color images (n_channels != 0)
        stack_to_reg = np.mean(np.stack([stack[:,:,:,channel] for channel in channels], 
                                        axis=-1),
                               axis=-1)
    ref_img = stack_to_reg[ref_num]
        
    # Compute the shifts
    col_list=[]
    row_list=[]
    phase_list=[]
    for i in range(0, len(stack_to_reg)): 
        img = stack_to_reg[i]
    
        col, row, phase = _compute_shifts(ref_img, img, return_error=False)
        col_list.append(col)
        row_list.append(row)
        phase_list.append(phase)
    
    # Register the images
    for i in range(0, len(reg_stack)): 
        if stack.ndim == 3:
            reg_stack[i] = _register_image(stack[i], col_list[i], row_list[i], phase_list[i])
        else:
            for c in channels:
                reg_stack[i,:,:,c] = _register_image(stack[i,:,:,c], col_list[i], row_list[i], phase_list[i])
    
    return reg_stack.clip(min=0.0, max=1.0)
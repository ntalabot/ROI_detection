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
    Compute and return the row and column shifts between the reference and shifted images.
    
    Args:
        ref_image: ndarray
            Reference image for the shifts computation, it should be 
            2-dimensional (i.e., a grayscale image).
        shifted_image: ndarray
            Shifted reference image, it should also be 2-dimensional.
        return_error: bool (default = False)
            If True, the normalized root-mean-square error between `ref_image` 
            and `shifted_image` is returned as a third output.
    
    Returns:
        row_shift: int
            The shift in the row coordinate.
        col_shift: int
            The shift in the col coordinate.
        [error]: float, optional
            The normalized root-mean-square error between `ref_image` 
            and `shifted_image`
    
    Translated by #TODO: from MATLAB code written by Manuel Guizar, downloaded 
    from http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation
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
        output = [row_shift, col_shift, error]
    else:
        output = [row_shift, col_shift]
        
    return output


def shift_image(image, row_shift, col_shift):
    """Shift the image by the given row and col shifts."""
    buf2ft = np.fft.fft2(image)
    nr, nc = buf2ft.shape

    # Compute registered version of buf2ft
    Nr = np.fft.ifftshift(np.arange(-np.fix(nr/2.0), np.ceil(nr/2.0)));
    Nc = np.fft.ifftshift(np.arange(-np.fix(nc/2.0), np.ceil(nc/2.0)));
    [Nc, Nr] = np.meshgrid(Nc, Nr)
    
    greg = buf2ft * np.exp(2j * np.pi * (-row_shift * Nr/nr - col_shift * Nc/nc));

    if np.can_cast(np.float32, image.dtype): # need to check this, too
        shifted_image = np.abs(np.fft.ifft2(greg))
    else:
        shifted_image = np.round(np.abs(np.fft.ifft2(greg))).astype(image.dtype)

    return shifted_image


def register_stack(stack, ref_num=0, channels=[0,1], return_shifts=False):
    """
    Register the stack using the cross-correlation method.
    For more details, see: Manuel Guizar-Sicairos, Samuel T. Thurman, and 
    James R. Fienup, "Efficient subpixel image registration algorithms," 
    Opt. Lett. 33, 156-158 (2008). 
    
    Args:
        stack: ndarray
            The stack of images. They can be grayscale or color (see `channels`
            argument).
        ref_num: int (default = 0)
            The number of the frame to take as reference. Every frame will be 
            registered to this one.
        channels: list of int (default = [0,1])
            Channels over which to compute the average if the images are color,
            as the alogorithm requires grayscale images.
        return_shifts: bool (default = False)
            If True, the function also returns the list of row and column shifts
            for each frame in the stack.
          
    Returns:
        reg_stack: ndarray
            The registered stack to the reference frame.
        [row_list]: list, optional
            The list of row shifts for all frames.
        [col_list]: list, optional
            The list of col shifts for all frames.
    """
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
    for i in range(0, len(stack_to_reg)): 
        img = stack_to_reg[i]
    
        row, col = _compute_shifts(ref_img, img, return_error=False)
        row_list.append(row)
        col_list.append(col)
    
    # Register the images
    for i in range(0, len(reg_stack)): 
        if stack.ndim == 3:
            reg_stack[i] = shift_image(stack[i], row_list[i], col_list[i])
        else:
            for c in channels:
                reg_stack[i,:,:,c] = shift_image(stack[i,:,:,c], row_list[i], col_list[i])
    reg_stack = reg_stack.clip(min=0.0, max=1.0)
    
    if return_shifts:
        return reg_stack, row_list, col_list
    else:
        return reg_stack
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate the synthetic dataset for the training.
See synthetic_generation.ipynb for more details.
Created on Thu Nov 22 10:01:56 2018

@author: nicolas
"""

import os, time
import warnings
import math
import ipywidgets as widgets
from ipywidgets import interact

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, draw, color
from scipy.stats import multivariate_normal
from imgaug import augmenters as iaa

from utils_common.image import to_npint
from utils_common.processing import flood_fill


def synthetic_stack(shape, n_images, n_neurons=-1):
    """
    Return a stack of synthetic neural images, with its ground truth segmentation.
    
    Args:
        shape: tuple of int
            Tuple (height, width) representing the shape of the images.
        n_images: int
            Number of images in the stack.
        n_neurons: int, default to -1
            Number of neurons to be present on the stack.
            If -1, will be randomly sampled between [2, 4].
            
    Returns:
        synth_stack: ndarray of shape NxHxW
            Stack of N synthetic images.
        synth_seg: ndarray of shape NxHxW
            Stack of N synthetic segmentations.
    """ 
    # Initialization
    n_samples = 1000 # number of samples for gaussian neurons
    grid_size = 8 # for the elastic deformation
    if n_neurons == -1:
        n_neurons = np.random.randint(2, 4 + 1)
    
    ## Create the gaussians representing the neurons
    max_neurons = []
    neurons = np.zeros((n_neurons,) + shape)
    neurons_segs = np.zeros((n_neurons,) + shape, dtype=np.bool)
    # Meshgrid for the gaussian weights
    rows, cols = np.arange(shape[0]), np.arange(shape[1])
    meshgrid = np.zeros(shape + (2,))
    meshgrid[:,:,0], meshgrid[:,:,1] = np.meshgrid(cols, rows) # note the order!
    for i in range(n_neurons):
        # Create neuron infinitly until in image and no overlap with another
        # TODO: change this to something that cannot loop to infinity
        while True:
            # Note that x and y axes are col and row (so, inversed!)
            mean = np.array([np.random.randint(shape[1]), np.random.randint(shape[0])])
            cross_corr = np.random.randint(-15, 15)
            cov = np.array([
                [np.random.randint(5, 15), cross_corr],
                [cross_corr, np.random.randint(50, 150)]
            ])

            # Bounding ellipses
            val, vec = np.linalg.eig(cov)
            rotation = math.atan2(vec[0, np.argmax(val)], vec[1, np.argmax(val)])
            rr, cc = draw.ellipse(mean[1], mean[0], 
                                  2*np.sqrt(cov[1,1]), 2*np.sqrt(cov[0,0]),
                                  rotation=rotation)
            # Check if outside the image
            if (rr < 0).any() or (rr >= shape[0]).any() or (cc < 0).any() or (cc >= shape[1]).any():
                continue
            # Check if overlapping with any existing neuron
            elif (neurons_segs[:, rr, cc] == True).any():
                continue
            else:
                break
        neurons_segs[i, rr, cc] = True
        
        # Create gaussian weight image
        neurons[i,:,:] = multivariate_normal.pdf(meshgrid, mean, cov)
        neurons[i,:,:] /= neurons[i,:,:].sum()

        # Sample randomly the neuron maximum 
        if np.random.rand() < _ROI_MAX_1:
            max_neurons.append(1.0)
        else:
            loc = _ROI_MAX_MEAN
            scale = _ROI_MAX_STD
            max_neurons.append(np.clip(np.random.normal(loc=loc, scale=scale), 0, 1))
        
    # Reduce segmentations to one image
    neurons_segs = neurons_segs.sum(axis=0)
    
    ## Warp neurons for each image to create the stack
    # Define warping sequence
    wrpseq = iaa.Sequential([
        iaa.PiecewiseAffine(scale=0.025, nb_rows=grid_size, nb_cols=grid_size)
    ])
    wrp_segs = np.zeros((n_images,) + shape, dtype=np.bool)
    wrp_neurons = np.zeros((n_images,) + shape, dtype=neurons.dtype)
    for i in range(n_images):
        # Set the warping to deterministic for warping both neurons and segmentation the same way
        seq_det = wrpseq.to_deterministic()
        
        ## Warp the neurons
        for j in range(n_neurons):
            # Warp gaussian defining it
            wrp_gaussian = seq_det.augment_image(neurons[j])
            wrp_gaussian /= wrp_gaussian.sum()
            # Sample from it
            x = np.random.choice(shape[0] * shape[1], size=n_samples, p=wrp_gaussian.ravel())
            y, x = np.unravel_index(x, shape)
            hist = plt.hist2d(x, y, bins=[shape[1], shape[0]], range=[[0, shape[1]], [0, shape[0]]])
            plt.close()
            wrp_neurons[i] = np.maximum(wrp_neurons[i], hist[0].T / hist[0].max() * max_neurons[j])
            
        ## Warp the segmentation
        wrp_segs[i] = seq_det.augment_image(neurons_segs)
        # Fill the possible holes in warped segmentation
        # It adds a border of background to avoid the case where a neuron is 
        # at the origin of the filling (this assumes that there aren't neurons EVERYWHERE
        # on the border of the original image)
        wrp_segs[i] = flood_fill(np.pad(wrp_segs[i], 1, 'constant'))[1:-1, 1:-1]
    
    ## Add noise (sampled from an exponential distribution)
    noise = np.random.exponential(scale=_BKG_MEAN, size=(n_images,) + shape)
    
    synth_stack = np.maximum(wrp_neurons, noise)
    synth_seg = wrp_segs
    return synth_stack, synth_seg


def gray2red(image):
    """Convert the grayscale image to red in RGB mode."""
    red = image
    green = np.zeros_like(image)
    blue = np.zeros_like(image)
    return np.stack([red, green, blue], axis=-1)


if __name__ == "__main__":
    ## Parameters and constants
    n_neurons = 2 # -1 for random
    n_stacks = 78
    n_images = 600
    synth_dir = "../dataset/synthetic/"
    shape = (192, 256)
    # Following are pre-computed on real data (dating of 21 Nov 2018). See stats_181121.pkl & README.md.
    _BKG_MEAN = 0.041733140976778674 # mean value of background
    _ROI_MAX_1 = 0.2276730082246407 # fraction of ROI with 1 as max intensity
    _ROI_MAX_MEAN = 0.6625502112855037 # mean of ROI max (excluding 1.0)
    _ROI_MAX_STD = 0.13925117610178622 # std of ROI max (excluding 1.0)
    
    date = time.strftime("%y%m%d", time.localtime())
        
    start = time.time()
    for i in range(n_stacks):
        folder = os.path.join(synth_dir, "synth_{}neur_{}_{:03d}".format(
            n_neurons, date, i))
        print("Creating stack %d/%d" % (i + 1, n_stacks), end="")
        print("  - folder:", folder)
        
        synth_stack, synth_seg = synthetic_stack(shape, n_images, n_neurons=n_neurons)
        
        # Change synth_stack to red in RGB mode (to be consistent with deep learning code)
        synth_stack = gray2red(synth_stack)
        
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, "rgb_frames"), exist_ok=True)
        os.makedirs(os.path.join(folder, "seg_frames"), exist_ok=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Save full stacks
            io.imsave(os.path.join(folder, "RGB.tif"), to_npint(synth_stack))
            io.imsave(os.path.join(folder, "seg_ROI.tif"), to_npint(synth_seg))
            # Save image per image
            for j in range(n_images):
                io.imsave(os.path.join(folder, "rgb_frames", "rgb_{:04}.png".format(j)), to_npint(synth_stack[j]))
                io.imsave(os.path.join(folder, "seg_frames", "seg_{:04}.png".format(j)), to_npint(synth_seg[j]))
    
    duration = time.time() - start
    print("\nScript took {:02.0f}min {:02.0f}s.".format(duration // 60, duration % 60))
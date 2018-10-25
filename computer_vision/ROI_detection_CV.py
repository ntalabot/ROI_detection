#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to find ROIs using only computer vision methods.
Created on Thu Oct 11 17:25:14 2018

@author: nicolas
"""

import os, pickle, warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import measure, io, feature, color, filters
from skimage import morphology as morph
from skimage.morphology import disk

from utils_ROI_detection import *


if __name__ == "__main__":
    # Parameters
    datadir = "dataset/"
    result_dir = "results_CV/"
    cmap = matplotlib.cm.get_cmap("autumn")
    channel_to_process = [0] # R,G,B <--> 0,1,2
    
    os.makedirs(result_dir, exist_ok=True)
    losses_mae = []
    losses_l2 = []
    losses_dice = []
    image_counter = 0
    # Loop over stacks
    data_dirs = sorted(os.listdir(datadir))
    for folder_num, subdir in enumerate(data_dirs):
        print("Starting processing folder %d/%d." % (folder_num + 1, len(data_dirs)))
        # Load stacks
        rgb_stack = imread_to_float(os.path.join(datadir, subdir, "RGB.tif"))
        stack = np.zeros(rgb_stack.shape[:-1], dtype=rgb_stack.dtype)
        for channel in channel_to_process:
            stack += rgb_stack[:,:,:,channel] / len(channel_to_process)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stack_pp = preprocess_stack(stack)
        
        # Loop through images
        seg_ROI = np.zeros(stack_pp.shape, dtype=np.bool)
        peak_ROI = np.zeros(stack_pp.shape, dtype=np.bool)
        centroids = []
        label_ROI = np.zeros(stack_pp.shape, dtype=np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(stack_pp)):
                # Segment neurons
                img = stack_pp[i]
#                img = filters.gaussian(img, 1, truncate=3)
                img_mean = img.mean()
                img = np.uint8(hysteresis_threshold(img, img_mean, img_mean * 10))
#                img = morph.closing(img, hline(5))
                img = morph.erosion(img)
                
                labels = measure.label(img)
                regions = measure.regionprops(labels)
                areas = np.array([region.area for region in regions])
                argsort = np.argsort(areas)
                if len(areas) > 1: # need at least 2 areas to compute gradient
                    large_labels = argsort[np.argmax(areas[argsort][1:] / areas[argsort][:-1]) + 1:] + 1
                else: # else, we can simply try to get the only area if it exists
                    large_labels = [1]
                
                for large_label in large_labels:
                    seg_ROI[i][labels == large_label] = True
                
                seg_ROI[i] = morph.dilation(seg_ROI[i])
                seg_ROI[i] = morph.closing(seg_ROI[i])
                
                # Find number and location of ROI
#                img = stack[i].copy()
#                img[seg_ROI[i] == False] = 0
#                
#                img = filters.gaussian(img, 3, truncate=4)
#                peak_ROI[i] = feature.peak_local_max(img, indices=False, footprint=np.ones((11,11)),
#                        labels=seg_ROI[i])
#                centroids.append(feature.peak_local_max(img, indices=True, footprint=np.ones((11,11)),
#                        labels=seg_ROI[i]))
#                markers = measure.label(peak_ROI[i])
#                peak_ROI[i] = morph.dilation(peak_ROI[i])
#                
#                # Segment ROI
#                label_ROI[i] = morph.watershed(-img, markers, mask=seg_ROI[i])
                label_ROI[i] = measure.label(seg_ROI[i])
                
                regions = measure.regionprops(label_ROI[i])
                centroids_img = []
                for region in regions:
                    centroids_img.append(region.centroid)
                    peak_ROI[i, int(centroids_img[-1][0]), int(centroids_img[-1][1])] = True
                centroids.append(np.array(centroids_img))
                peak_ROI[i] = morph.dilation(peak_ROI[i])
        
        # Create the overlay
        overlay_ROI = color.gray2rgb(stack)
        for i in range(len(stack)):
            for ROI_num in range(1, label_ROI[i].max() + 1):
                overlay_ROI[i] = overlay_mask(overlay_ROI[i], label_ROI[i] == ROI_num,
                           opacity=0.25, mask_color=cmap(ROI_num / label_ROI[i].max())[:3])
        
        overlay_ROI = overlay_mask_stack(overlay_ROI, peak_ROI, opacity=1.0,
                                         mask_color=[1.0, 1.0, 0.0])
        
        # Save results and disable warnings about low contrast
        os.makedirs(os.path.join(result_dir, subdir), exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(os.path.join(result_dir, subdir, "seg_ROI.tif"), to_npint(seg_ROI, dtype=np.uint8))
            io.imsave(os.path.join(result_dir, subdir, "label_ROI.tif"), label_ROI)
            io.imsave(os.path.join(result_dir, subdir, "overlay_ROI.tif"), to_npint(overlay_ROI, dtype=np.uint16))
        with open(os.path.join(result_dir, subdir, "centroids.pkl"), 'wb') as file:
            pickle.dump(centroids, file)
        
        ## Compute losses
        # Load ground truth
        gt_seg_ROI = io.imread(os.path.join(datadir, subdir, "seg_ROI.tif"))
        with open(os.path.join(datadir, subdir, "centroids.pkl"), 'rb') as file:
            centroids = pickle.load(file)
        # Compute losses
        losses_mae.append(loss_mae(np.max(label_ROI, axis=(1,2)),
                                   np.array([centroid.shape[0] for centroid in centroids]),
                                   reduction='sum'))
        losses_l2.append(loss_l2_segmentation(seg_ROI, gt_seg_ROI, reduction='sum'))
        losses_dice.append(dice_coef(seg_ROI, gt_seg_ROI, reduction='sum'))
        
        image_counter += len(stack)
    
    # Compute final losses, and print them
    loss_mae = np.sum(losses_mae) / image_counter
    loss_l2 = np.sum(losses_l2) / image_counter
    loss_dice = np.sum(losses_dice) / image_counter
    print("Losses:")
    print("MAE over ROI number: {:.3f}".format(loss_mae))
    print("L2 over segmentation: {:.3f}".format(loss_l2))
    print("\nAverage Dice coefficient: {:.3f}".format(loss_dice))
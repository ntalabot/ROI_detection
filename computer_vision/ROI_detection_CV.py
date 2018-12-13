#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/!\ OLD ONE
Script to find ROIs using only computer vision methods.
Created on Thu Oct 11 17:25:14 2018

@author: nicolas
"""

import os, pickle, warnings
import argparse
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, io, feature, color, filters
from skimage import morphology as morph

from utils_common.image import imread_to_float, overlay_mask_stack, to_npint
from utils_common.processing import preprocess_stack
from utils_common.metrics import dice_coef, crop_metric
import utils_common.register_cc as reg

def main(args):
    # Parameters
    datadir = "/data/talabot/dataset/validation/"
    result_dir = "results_CV_old/"
    channels_to_process = [0] # R,G,B <--> 0,1,2
    scale_dice = 4.0 # scale of the cropping (w.r.t. ROI's bounding box) for cropped dice
    
    os.makedirs(result_dir, exist_ok=True)
    losses_dice = []
    losses_diC = []
    image_counter = 0
    # Loop over stacks
    data_dirs = sorted(os.listdir(datadir))
    for folder_num, subdir in enumerate(data_dirs):
        print("Starting processing folder %d/%d." % (folder_num + 1, len(data_dirs)))
        # Load stacks
        rgb_stack = imread_to_float(os.path.join(datadir, subdir, "RGB.tif"))
        stack = np.zeros(rgb_stack.shape[:-1], dtype=rgb_stack.dtype)
        for channel in channels_to_process:
            stack += rgb_stack[:,:,:,channel] / len(channels_to_process)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stack_pp = preprocess_stack(stack)
        
        # Compute number of ROI if unknown
        if args.num_peaks == -1:
            reg_stack, rows, cols = reg.register_stack(stack_pp, return_shifts=True)
            
            # Find centroids on mean temporal image
            mean_img = reg_stack.mean(axis=0)
            mean_pp = filters.gaussian(mean_img, 3, truncate=4)
            mean_bin = filters.apply_hysteresis_threshold(mean_pp, mean_pp.mean(), 
                                                          mean_pp.max() * args.threshold_rel)
            peaks = feature.peak_local_max(mean_pp, footprint=np.ones((11,11)), 
                                           threshold_rel=args.threshold_rel,
                                           labels=mean_bin)
            args.num_peaks = peaks.shape[0]
        
        # Segment ROI by looping through images
        seg_ROI = np.zeros(stack_pp.shape, dtype=np.bool)
        peak_ROI = np.zeros(stack_pp.shape, dtype=np.bool)
        centroids = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(stack_pp)):
                img = stack_pp[i]
                img = filters.gaussian(img, 3, truncate=4)
                
                # Find the centroids location on the image
                peaks = feature.peak_local_max(
                    img, footprint=np.ones((11,11)), threshold_rel=args.threshold_rel,
                    labels=mean_bin, num_peaks=args.num_peaks) 
                
                for j in range(peaks.shape[0]):
                    peak_ROI[i, peaks[j,0], peaks[j,1]] = True
                    
                # Segment the ROI by thresholding locally (keep closest region 
                # to each centroid)
                img_bin = np.zeros(img.shape, dtype=np.bool)
                for j in range(peaks.shape[0]):
                    # Threshold by centroid's value (with img.mean as minimum)
                    c_row, c_col = peaks[j]
                    local_bin = img > max(img[c_row, c_col] * args.threshold_rel_peak, 
                                          img.mean())
                    # Keep only the region with the centroid
                    local_labels = measure.label(local_bin)
                    centroid_label = local_labels[c_row, c_col]
                    if centroid_label != 0:
                        local_bin = local_labels == centroid_label 
                    else: # Or find closest region if centroid is in background
                        closest_label = -1
                        closest_dist = np.inf
                        for region in measure.regionprops(local_labels):
                            dist = ((region.coords - peaks[j]) ** 2).sum(axis=1).min()
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_label = region.label
                        local_bin = local_labels == closest_label 
                    # Add the region to the ROI segmentation
                    img_bin[local_bin] = True
                # Erode to cancel initial gaussian filtering
                img_bin = morph.erosion(img_bin)
                
#                # Segment individual ROI with their centroid
#                img_dist = ndi.distance_transform_edt(img_bin)
#                local_maxi = feature.peak_local_max(img_dist, indices=False, footprint=np.ones((3,3)),
#                                                    labels=img_bin, num_peaks=peaks.shape[0])
#                markers = measure.label(local_maxi)[0] # local_maxi/peak_ROI
#                img_labels = morph.watershed(-img_dist, markers, mask=img_bin) # img_pp/img_dist
                
                # Put results into the variables
                seg_ROI[i] = img_bin
                peak_ROI[i] = morph.dilation(peak_ROI[i])
                centroids.append(peaks)
        
        # Create the overlay
        overlay_ROI = overlay_mask_stack(rgb_stack.copy(), seg_ROI, opacity=0.25,
                                             mask_color=[0.0, 0.0, 1.0])
        overlay_ROI = overlay_mask_stack(overlay_ROI, peak_ROI, opacity=1.0,
                                               mask_color=[1.0, 1.0, 0.0])
        
        # Save results and disable warnings about low contrast
        os.makedirs(os.path.join(result_dir, subdir), exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(os.path.join(result_dir, subdir, "seg_ROI.tif"), to_npint(seg_ROI, dtype=np.uint8))
            io.imsave(os.path.join(result_dir, subdir, "overlay_ROI.tif"), to_npint(overlay_ROI, dtype=np.uint16))
        with open(os.path.join(result_dir, subdir, "centroids.pkl"), 'wb') as file:
            pickle.dump(centroids, file)
        
        ## Compute losses
        # Load ground truth
        gt_seg_ROI = io.imread(os.path.join(datadir, subdir, "seg_ROI.tif"))
#        with open(os.path.join(datadir, subdir, "centroids.pkl"), 'rb') as file:
#            gt_centroids = pickle.load(file)
        # Compute losses
        losses_dice.append(dice_coef(seg_ROI, gt_seg_ROI, reduction='sum'))
        losses_diC.append(crop_metric(dice_coef, seg_ROI, gt_seg_ROI, scale=scale_dice, reduction='sum'))
        
        image_counter += len(stack)
    
    # Compute final losses, and print them
    loss_dice = np.sum(losses_dice) / image_counter
    loss_diC = np.sum(losses_diC) / image_counter
    print("Losses:")
    print("Average Dice coefficient: {:.3f}".format(loss_dice))
    print("Average Crop Dice coef.:  {:.3f}".format(loss_diC))
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect ROI with computer vision in the data.")
    parser.add_argument(
            '--threshold_rel', 
            type=float,
            default=0.3, 
            help="relative threshold for detecting centroid numbers. "
            "Unused if num_peaks != -1 (default=0.3)"
    )
    parser.add_argument(
            '--threshold_rel_peak', 
            type=float,
            default=0.36, 
            help="relative threshold for ROI segmentation (default=0.36)"
    )
    parser.add_argument(
            '--num_peaks', 
            type=float,
            default=-1, 
            help="number of expected ROI, use -1 for 'unknown' (default=-1)"
    )
    args = parser.parse_args()
    
    main(args)
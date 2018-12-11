#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to detect ROI on experiments in a given folder.
Created on Mon Dec 10 14:50:39 2018

@author: nicolas
"""    

import os, warnings, time
import argparse
import numpy as np
from skimage import io

from cv_detector import cv_detect
from utils_common.image import imread_to_float, to_npint
from utils_common.metrics import dice_coef, crop_metric


def main(args):
    if args.timeit:
        start_time = time.time()
        
    # Parameters
    datadir = "/data/talabot/dataset/validation/"
    result_dir = "results_CV/"
    scale_dice = 4.0 # scale of the cropping (w.r.t. ROI's bounding box) for cropped dice
    
    os.makedirs(result_dir, exist_ok=True)
    losses_dice = []
    losses_diC = []
    losses_dice_mask = []
    image_counter = 0
    # Loop over stacks
    data_dirs = sorted(os.listdir(datadir))
    for folder_num, subdir in enumerate(data_dirs):
        print("Starting processing folder %d/%d." % (folder_num + 1, len(data_dirs)))
        # Load stacks
        rgb_stack = imread_to_float(os.path.join(datadir, subdir, "RGB.tif"))
        true_seg = imread_to_float(os.path.join(datadir, subdir, "seg_ROI.tif"))
        try:
            mask_stack = imread_to_float(os.path.join(datadir, subdir, "mask.tif"))
        except FileNotFoundError:
            mask_stack = np.zeros_like(true_seg)
        
        # Compute results
        predictions = cv_detect(rgb_stack)
        
        # Save results and disable warnings about low contrast
        os.makedirs(os.path.join(result_dir, subdir), exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(os.path.join(result_dir, subdir, "preds.tif"), to_npint(predictions, dtype=np.uint8))
        
        ## Compute losses
        losses_dice.append(dice_coef(predictions, true_seg, reduction='sum'))
        losses_diC.append(crop_metric(dice_coef, predictions, true_seg, scale=scale_dice, reduction='sum'))
        losses_dice_mask.append(dice_coef(predictions, true_seg, 
                                          masks=1 - mask_stack, reduction='sum'))
        
        image_counter += len(true_seg)
    
    # Compute final losses, and print them
    loss_dice = np.sum(losses_dice) / image_counter
    loss_diC = np.sum(losses_diC) / image_counter
    loss_dice_mask = np.sum(losses_dice_mask) / image_counter
    print("Losses:")
    print("Average Dice coefficient: {:.3f}".format(loss_dice))
    print("Average Crop Dice coef.:  {:.3f}".format(loss_diC))
    print("Average Dice coef. with masks: {:.3f}".format(loss_dice_mask))
    
    # Display script duration if applicable
    if args.timeit:
        duration = time.time() - start_time
        duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
        print("\nScript took %s." % duration_msg)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect ROI with computer vision in the data.")
    parser.add_argument(
            '-t', '--timeit', 
            action="store_true",
            help="time the script"
    )
    args = parser.parse_args()
    
    main(args)
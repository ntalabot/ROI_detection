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
from skimage import io, filters
from skimage.morphology import disk

from cv_detector import cv_detect
from utils_common.image import imread_to_float, to_npint
from utils_common.metrics import dice_coef, crop_metric


def main(args, thresholding_fn, registration, selem, datadir=None):
    if args.timeit:
        start_time = time.time()
        
    # Parameters
    if datadir is None:
        datadir = "/data/talabot/dataset/validation/"
    scale_dice = 4.0 # scale of the cropping (w.r.t. ROI's bounding box) for cropped dice
    
    if args.save:
        os.makedirs(args.result_dir, exist_ok=True)
        
    losses_dice = []
    losses_diC = []
    losses_dice_mask = []
    image_counter = 0
    # Loop over stacks
    data_dirs = sorted(os.listdir(datadir))
    for folder_num, subdir in enumerate(data_dirs):
        if args.verbose:
            print("Starting processing folder %d/%d." % (folder_num + 1, len(data_dirs)))
        # Load stacks
        rgb_stack = imread_to_float(os.path.join(datadir, subdir, "RGB.tif"))
        true_seg = imread_to_float(os.path.join(datadir, subdir, "seg_ROI.tif"))
        try:
            mask_stack = imread_to_float(os.path.join(datadir, subdir, "mask.tif"))
        except FileNotFoundError:
            mask_stack = np.zeros_like(true_seg)
        
        # Compute results
        predictions = cv_detect(rgb_stack, thresholding_fn, registration, selem)
        
        # Save results and disable warnings about low contrast
        if args.save:
            os.makedirs(os.path.join(args.result_dir, subdir), exist_ok=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(os.path.join(args.result_dir, subdir, "preds.tif"), to_npint(predictions, dtype=np.uint8))
        
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
            '--result_dir', 
            type=str,
            default="results_CV", 
            help="name of the directory where to save the results (see --save "
            "argument) (default = results_CV)"
    )
    parser.add_argument(
            '-s', '--save', 
            action="store_true",
            help="save the results (see --result_dir argument)"
    )
    parser.add_argument(
            '-t', '--timeit', 
            action="store_true",
            help="time the script"
    )
    parser.add_argument(
            '-v', '--verbose', 
            action="store_true",
            help="increase the verbosity (display folder being processed)"
    )
    args = parser.parse_args()
    
    for phase in ["train", "validation", "test"]:
        args.result_dir = os.path.join("results_CV/", phase)
        print("\nProcessing %s set." % phase)
        main(args, filters.threshold_otsu, registration=False, selem=disk(1), 
             datadir=os.path.join("/data/talabot/dataset/", phase))
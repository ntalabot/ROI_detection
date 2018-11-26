#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train a network.
Created on Thu Nov  1 10:45:50 2018

@author: nicolas
"""

import os, time, shutil
import argparse
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

from utils_common.metrics import dice_coef, crop_dice_coef
from utils_data import get_all_dataloaders
from utils_model import CustomUNet
from utils_train import train
from utils_test import evaluate

def main(args, model=None):
    """Main function of the run_train script, can be used as is with correct arguments (and optional model)."""
    ## Initialization
    if args.timeit:
        start_time = time.time()
    
    # Seed the script
    seed = 1
    random.seed(seed)
    np.random.seed(seed*10 + 1234)
    torch.manual_seed(seed*100 + 4321)
    
    # Device selection (note that the current code does not work with multi-GPU)
    if torch.cuda.is_available() and not args.no_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if args.verbose:
        print("Device set to '{}'.".format(device))
        
    ## Data preparation    
    # Create dataloaders
    dataloaders = get_all_dataloaders(
        args.data_dir, 
        args.batch_size, 
        input_channels = args.input_channels, 
        test_dataloader = args.eval_test,
        synthetic_data = args.synthetic_data,
        synthetic_ratio = args.synthetic_ratio,
        train_transform = None, train_target_transform = None,
        eval_transform = None, eval_target_transform = None
    )
    
    N_TRAIN = len(dataloaders["train"].dataset)
    N_VALID = len(dataloaders["valid"].dataset)
    if args.eval_test:
        N_TEST = len(dataloaders["test"].dataset)
    # Positive class weight (pre-computed)
    pos_weight = torch.tensor(120.0).to(device)
    
    if args.verbose:
        print("%d train images" % N_TRAIN, end="")
        if args.synthetic_data:
            if args.synthetic_ratio is None:
                print(" (with synthetic data).")
            else:
                print(" (with %d%% of synthetic data)." % (args.synthetic_ratio * 100))
        else:
            print(".")
        print("%d validation images." % N_VALID)
        if args.eval_test:
            print("%d test images." % N_TEST)
        print("{:.3f} positive weighting.".format(pos_weight.item()))
    
    ## Model, loss, and optimizer definition
    if model is None:
        model = CustomUNet(len(args.input_channels), u_depth=4,
                           out1_channels=16, batchnorm=True, device=device)
        if args.model_dir is not None:
            # Save the "architecture" of the model by copy/pasting the class definition file
            os.makedirs(os.path.join(args.model_dir), exist_ok=True)
            shutil.copy("utils_model.py", os.path.join(args.model_dir, "utils_model_save.py"))
    # make sure the given model is on the correct device
    else: 
        model.to(device)
        
    if args.verbose:
        print("\nModel definition:", model, "\n")
    
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='elementwise_mean', pos_weight=pos_weight)
    
    dice_metric = lambda preds, targets: torch.tensor(
            dice_coef((torch.sigmoid(preds.cpu()) > 0.5).detach().numpy(),
                      targets.cpu().detach().numpy()))
    
    diceC_metric = lambda preds, targets: torch.tensor(
        crop_dice_coef((torch.sigmoid(preds.cpu()) > 0.5).detach().numpy(),
                       targets.cpu().detach().numpy(),
                       scale = args.scale_dice))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    ## Train the model
    best_model, history = train(model,
                                dataloaders,
                                loss_fn,
                                optimizer,
                                args.epochs,
                                metrics = {"dice": dice_metric, 
                                           "diC%.1f" % args.scale_dice: diceC_metric},
                                criterion_metric = "dice",
                                model_dir = args.model_dir,
                                replace_dir = True,
                                verbose = args.verbose)
    
    ## Save a figure if applicable
    if args.save_fig and args.model_dir is not None:
        fig = plt.figure(figsize=(12,6))
        plt.subplot(131)
        plt.title("Loss")
        plt.plot(history["epoch"], history["loss"])
        plt.plot(history["epoch"], history["val_loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["train loss", "valid loss"])
        plt.subplot(132)
        plt.title("Dice coefficient")
        plt.plot(history["epoch"], history["dice"])
        plt.plot(history["epoch"], history["val_dice"])
        plt.xlabel("Epoch")
        plt.ylabel("Dice coef.")
        plt.ylim(0,1)
        plt.legend(["train dice", "valid dice"])
        plt.subplot(133)
        plt.title("Cropped Dice coefficient (scale = %.1f)" % args.scale_dice)
        plt.plot(history["epoch"], history["diC%.1f" % args.scale_dice])
        plt.plot(history["epoch"], history["val_diC%.1f" % args.scale_dice])
        plt.xlabel("Epoch")
        plt.ylabel("Cropped Dice coef.")
        plt.ylim(0,1)
        plt.legend(["train diC%.1f" % args.scale_dice, "valid diC%.1f" % args.scale_dice])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join(args.model_dir, "train_fig.png"), dpi=400)
        print("Training figure saved at %s." % os.path.join(args.model_dir, "train_fig.png"))
    if args.model_dir is not None:
        print("Best model saved under %s." % args.model_dir)
       
    ## Evaluate best model over test data
    if args.eval_test:
        test_metrics = evaluate(best_model, dataloaders["test"], 
                                {"loss": loss_fn, "dice": dice_metric,
                                 "diC%.1f" % args.scale_dice: diceC_metric})
        if args.verbose:
            print("\nTest loss = {}".format(test_metrics["loss"]))
            print("Test dice = {}".format(test_metrics["dice"]))
            print("Crop dice = {}".format(test_metrics["diC%.1f" % args.scale_dice]))
        
    ## Display script duration if applicable
    if args.timeit:
        duration = time.time() - start_time
        duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
        print("\nScript took %s." % duration_msg)
        
    # If model was evaluated on test data, return the best metric values, and 
    # return in any case the history. This is useless in this script, but allow 
    # this function to be reused somewhere else, e.g. for the gridsearch.
    if args.eval_test:
        return history, test_metrics
    else:
        return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a network to detect ROI on single images.")
    parser.add_argument(
            '--batch_size', 
            type=int,
            default=32, 
            help="batch_size for the dataloaders (default=32)"
    )
    parser.add_argument(
            '--data_dir',
            type=str,
            default="../dataset/", 
            help="directory to the train, validation, and test data. It should contain "
            "train/, validation/, and test/ subdirs (test/ is not mandatory, see --eval_test). "
            "These should be structured as: "
            "train_dir-->subdirs-->rgb_frames: folder with input images; and "
            "train_dir-->subdirs-->seg_frames: folder with target images (default=../dataset/)"
    )
    parser.add_argument(
            '--epochs', 
            type=int,
            default=5, 
            help="number of epochs (default=5)"
    )
    parser.add_argument(
            '--eval_test', 
            action="store_true",
            help="perform a final evaluation over the test data"
    )
    parser.add_argument(
            '--input_channels', 
            type=str,
            default="R", 
            help="channels of RGB input images to use (default=R)"
    )
    parser.add_argument(
            '--learning_rate', 
            type=float,
            default=0.001,
            help="learning rate for the stochastic gradient descent (default=0.001)"
    )
    parser.add_argument(
            '--model_dir', 
            type=str,
            help="directory where the model is to be saved (if not set, the model won't be saved)"
    )
    parser.add_argument(
            '--no_gpu', 
            action="store_true",
            help="disable gpu utilization (not needed if no gpu are available)"
    )
    parser.add_argument(
            '--save_fig', 
            action="store_true",
            help="save a figure of the training loss and metrics with the model "
            "(requires the --model_dir argument to be set)"
    )
    parser.add_argument(
            '--scale_dice', 
            type=float,
            default=4.0,
            help="scaling of the cropping (w.r.t. ROI's bounding box) for "
            "the cropped dice coef. (default=4.0)"
    )
    parser.add_argument(
            '--seed', 
            type=int,
            default=1,
            help="initial seed for RNG (default=1)"
    )
    parser.add_argument(
            '--synthetic_data', 
            action="store_true",
            help="enable the use of synthetic data for training"
    )
    parser.add_argument(
            '--synthetic_ratio', 
            type=float,
            default=None,
            help="(requires synthetic_data to be set) ratio of synthetic data "
            "vs. real data. If not set, all real and synthetic data are used"
    )
    parser.add_argument(
            '-t', '--timeit', 
            action="store_true",
            help="time the script"
    )
    parser.add_argument(
            '-v', '--verbose', 
            action="store_true",
            help="enable output verbosity"
    )
    args = parser.parse_args()
    
    main(args)
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
import matplotlib.pyplot as plt
from skimage import io

import torch

from utils_common.ROI_detection import dice_coef
from utils_data import ImageLoaderDataset, get_filenames, get_dataloaders
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
    # Create lists of filenames
    x_train, y_train = get_filenames(os.path.join(args.data_dir, "train/"))
    if args.no_test:
        pass
    else:
        x_test, y_test = get_filenames(os.path.join(args.data_dir, "test/"))
    
    # Create dataloaders
    dataloaders = get_dataloaders(x_train, y_train, args.train_ratio, args.batch_size,
                                  input_channels = args.input_channels,
                                  train_transform = None,  valid_transform = None)
    if args.no_test:
        pass
    else:
        test_loader = torch.utils.data.DataLoader(
                ImageLoaderDataset(x_test, y_test, input_channels=args.input_channels, 
                                   transform=None, target_transform=None),
                batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    N_TRAIN = len(dataloaders["train"].dataset)
    N_VALID = len(dataloaders["valid"].dataset)
    if args.no_test:
        pass
    else:
        N_TEST = len(test_loader.dataset)
    HEIGHT, WIDTH = io.imread(x_train[0]).shape[:2]
    # Positive class weight (pre-computed)
    pos_weight = torch.tensor(120.946829).to(device)
    
    if args.verbose:
        print("{:.3f} positive weighting.".format(pos_weight.item()))
        print("%d train images of size %dx%d (%d to train, %d to validation)." % \
              (len(x_train), HEIGHT, WIDTH, N_TRAIN, N_VALID))
        if args.no_test:
            pass
        else:
            print("%d test images." % N_TEST)
    
    ## Model, loss, and optimizer definition
    if model is None:
        model = CustomUNet(len(args.input_channels), batchnorm=False, device=device)
        if args.model_dir is not None:
            # Save the "architecture" of the model by copy/pasting the class definition file
            os.makedirs(os.path.join(args.model_dir), exist_ok=True)
            shutil.copy("utils_model.py", os.path.join(args.model_dir, "utils_model_save.py"))
    else:
        model.to(device)
        
    if args.verbose:
        print("\nModel definition:", model, "\n")
    
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='elementwise_mean', pos_weight=pos_weight)
    dice_metric = lambda preds, targets: torch.tensor(dice_coef((torch.sigmoid(preds.cpu()) > 0.5).detach().numpy(),
                                                                targets.cpu().detach().numpy()))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    ## Train the model
    best_model, history = train(model,
                                dataloaders,
                                loss_fn,
                                optimizer,
                                args.epochs,
                                metrics = {"dice": dice_metric},
                                criterion_metric = "dice",
                                model_dir = args.model_dir,
                                replace_dir = True,
                                verbose = args.verbose)
    
    ## Save a figure if applicable
    if args.save_fig and args.model_dir is not None:
        fig = plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.title("Loss")
        plt.plot(history["epoch"], history["loss"])
        plt.plot(history["epoch"], history["val_loss"])
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.legend(["train loss", "valid. loss"])
        plt.subplot(122)
        plt.title("Dice coefficient")
        plt.plot(history["epoch"], history["dice"])
        plt.plot(history["epoch"], history["val_dice"])
        plt.xlabel("Epoch"); plt.ylabel("Dice coef.")
        plt.legend(["train dice", "valid. dice"])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join(args.model_dir, "train_fig.png"), dpi=400)
       
    ## Evaluate best model over test data
    if args.no_test:
        pass
    else:
        test_metrics = evaluate(best_model, test_loader, {"loss": loss_fn, "dice": dice_metric})
        if args.verbose:
            print("\nTest loss = {}\nTest dice = {}".format(test_metrics["loss"], test_metrics["dice"]))
        
    ## Display script duration if applicable
    if args.timeit:
        duration = time.time() - start_time
        duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
        print("\nScript took %s." % duration_msg)
        
    # If model was evaluated on test data, return the best metric values
    # This is useless in this script, but allow this function to be reused 
    # somewhere else, e.g. for the gridsearch.
    if args.no_test:
        return
    else:
        return test_metrics


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
            help="directory to the train and test data. It should contain "
            "train/ and test/ subdirs (test/ is not mandatory, see --no_test). "
            "These should be structured as: "
            "train or test_dir-->subdirs-->rgb_frames: folder with input images; and "
            "train or test_dir-->subdirs-->seg_frames: folder with target images (default=../dataset/)"
    )
    parser.add_argument(
            '--epochs', 
            type=int,
            default=5, 
            help="number of epochs (default=5)"
    )
    parser.add_argument(
            '--input_channels', 
            type=str,
            default="RG", 
            help="channels of RGB input images to use (default=RG)"
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
            '--no_test', 
            action="store_true",
            help="disable final evaluation over the test data (use this if you have no test data)"
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
            '--seed', 
            type=int,
            default=1,
            help="initial seed for RNG (default=1)"
    )
    parser.add_argument(
            '--train_ratio',
            type=float,
            default=0.8,
            help="percentage of train data actually going to training vs. validation (default=0.8)"
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
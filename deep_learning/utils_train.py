#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for training with PyTorch.
Created on Mon Oct 22 13:54:19 2018

@author: nicolas
"""

import os, time, tempfile, shutil, copy
import numpy as np

import torch


def train(model, dataloaders, loss_fn, optimizer, n_epochs, metrics={},
          criterion_metric="", model_dir=None, replace_dir=True, verbose=1):
    """
    Train the model, and return the best found, as well as the training history.
    
    Args:
        model: the PyTorch model
        dataloaders (dict): contains the train and validation DataLoaders with
            respective keys "train" and "valid"
        loss (callable): the PyTorch loss function. Should take 2 tensors in 
            (predictions and targets), and output a tensor
        optimizer: the PyTorch optimizer
        n_epochs (int): number of epochs (pass over the total data)
        metrics (dict): dict of metrics to compute over the data. Should take 
            2 tensors in (predictions and targets), and output a tensor
        criterion_metric(str): key of the metric to use for early stopping
            (can be "loss")
        model_dir (str): path of the folder in which the best model is saved.
            If None, the model won't be saved.
        replace_dir (bool): If True and model_dir is already existing, it will
            be over-written
        verbose (int): verbosity of the function (0 is silent)
    
    Returns:
        best_model: the best model found, based on validation metrics[criterion_metric] 
            or negative of loss if no metrics given or if criterion_metric == "loss"
        history: a dictionary with the training history. Validation keys
            are like train keys with a preceding "val_"
    """
    best_val_criterion = - np.inf
    if not metrics:
        criterion_metric = "loss"
    
    # If no model folder for saving, create a temporary one
    if model_dir is None:
        save_dir = tempfile.mkdtemp()
    else:
        save_dir = model_dir
        
    history = {"loss": [], "val_loss": [], "epoch": []}
    for key in metrics.keys():
        history[key] = []
        history["val_" + key] = []
    
    
    if verbose:
        start_time = time.time()
    
    for epoch in range(n_epochs):
        if verbose:
            duration = time.time() - start_time
            duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(
                    duration // 3600, (duration // 60) % 60, duration % 60)
            epoch_msg = "Epoch %d/%d  (Elapsed time: %s)" % (epoch + 1, n_epochs, duration_msg)
            print(epoch_msg)
            print("-" * len(epoch_msg))
        
        history["epoch"].append(epoch)
        
        for phase in ["train", "valid"]:
            if phase == 'train':
                model.train()  # Set model to training mode
                if verbose:
                    print("Batch (over %d): " % len(dataloaders["train"]), end="")
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0
            running_metrics = {}
            for key in metrics.keys():
                running_metrics[key] = 0
            
            # Iterate over the data
            for i, (batch_x, batch_y) in enumerate(dataloaders[phase]): 
                # TODO: input to device
                
                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(batch_x)
                    
                    # Loss
                    loss = loss_fn(y_pred, batch_y)
                    running_loss += loss.item() * batch_x.shape[0]
                    
                    # Metrics
                    for key in metrics.keys():
                        running_metrics[key] += \
                            metrics[key](y_pred, batch_y).item() * batch_x.shape[0]
                        
                    if phase == "train":
                        if ((i + 1) % 25 == 0 or i == 0) and verbose: 
                            print("%d..." % (i + 1), end="")
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
            ## End of phase
            # Save statistics
            if phase == "train":
                if verbose:
                    print()
                history["loss"].append(running_loss / len(dataloaders[phase].dataset))
                for key in metrics.keys():
                    history[key].append(running_metrics[key] / len(dataloaders[phase].dataset))
                
            elif phase == "valid":
                history["val_loss"].append(running_loss / len(dataloaders[phase].dataset))
                for key in metrics.keys():
                    history["val_" + key].append(running_metrics[key] / len(dataloaders[phase].dataset))
            
            # Print them
            if verbose:
                phase_msg = "{} loss: {:.6f}".format(phase.capitalize(), 
                             running_metrics[key] / len(dataloaders[phase].dataset))
                for key in metrics.keys():
                    phase_msg += " - {}: {:.6f}".format(key,
                                     running_metrics[key] / len(dataloaders[phase].dataset))
                print(phase_msg)
            
        # Copy the model if best found so far
        criterion_val = history["val_" + criterion_metric][-1]
        if criterion_metric == "loss":
            criterion_val *= -1.0
        
        if criterion_val > best_val_criterion:
            best_val_criterion = criterion_val
            # Save model state dict
            # TODO: save model architecture
            torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pt"))
        
        if verbose:
            print()
    
    if verbose:
        duration = time.time() - start_time
        duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
        print("Training took %s." % duration_msg)
        print("Best validation {} = {:.3f}".format(criterion_metric, best_val_criterion))
    
    # Load best model, and remove tmpdir if applicable
    best_model = copy.deepcopy(model)
    best_model.load_state_dict(torch.load(os.path.join(save_dir, "model_best.pt")))
    if model_dir is None:
        shutil.rmtree(save_dir)
    
    return best_model, history
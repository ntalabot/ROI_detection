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
          criterion_metric="loss", model_dir=None, replace_dir=True, verbose=1):
    """
    Train the model, and return the best found, as well as the training history.
    
    Args:
        model: PyTorch model
            The model, based on Torch.nn.Module. It should have a `device` 
            attribute.
        dataloaders: dict of dataloaders
            Contains the train and validation DataLoaders with respective keys 
            "train" and "valid".
        loss_fn: callable
            The PyTorch loss function. It should take 3 tensors as input
            (predictions, targets, and masks), and output a scalar tensor
        optimizer: PyTorch optimizer
            Optimzer for the SGD algorithm.
        n_epochs: int
            Number of epochs (pass over the whole data).
        metrics: dict of callable
            Dictionary of metrics to be computed over the data. It should take 
            3 tensors as input (predictions, targets, and masks), and output a 
            scalar tensor. Keys should be their name, value the callable.
        criterion_metric: str (default = "loss")
            Name of the metric to use for early stopping. It can be "loss", in
            this case, it is based on the highest negative loss. Otherwise, it
            should be the same as the key in the `metrics` dictionary.
            Note that it is automatically based on the validation set.
        model_dir: str (default = None)
            Directory/path of the folder in which the best model is saved.
            If None, the model won't be saved.
        replace_dir: bool (default = True)
            If True and model_dir is already existing, it will be over-written.
        verbose: int (default = 1)
            Verbosity of the function (0 means silent).
    
    Returns:
        best_model: PyTorch model
            The best model found, i.e. corresponding to the epoch where the 
            validation metrics[criterion_metric] (or negative loss) is the highest.
        history: dict
            Dictionary with the training history. Validation keys are like 
            their training counterparts, with the prefix "val_".
    """
    best_val_criterion = -np.inf
    best_epoch = -1
    
    # If no model folder for saving, create a temporary one
    if model_dir is None:
        save_dir = tempfile.mkdtemp()
    else:
        save_dir = model_dir
    os.makedirs(save_dir, exist_ok=replace_dir)
        
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
            for i, (batch_x_cpu, batch_y_cpu, batch_mask_cpu) in enumerate(dataloaders[phase]): 
                # Copy tensor to the model device
                batch_x = batch_x_cpu.to(model.device)
                batch_y = batch_y_cpu.to(model.device)
                batch_mask = batch_mask_cpu.to(model.device)
                
                with torch.set_grad_enabled(phase == "train"):
                    # Forward pass
                    y_pred = model(batch_x)
                    
                    # Masked loss
                    masking = (1 - batch_mask)
                    loss = loss_fn(y_pred[masking], batch_y[masking])
                    running_loss += loss.item() * batch_x.shape[0]
                    
                    # Metrics
                    y_pred_cpu = y_pred.cpu()
                    for key in metrics.keys():
                        running_metrics[key] += \
                            metrics[key](y_pred_cpu, batch_y_cpu, 
                                   (1 - batch_mask_cpu)).item() * batch_x.shape[0]
                        
                    if phase == "train":
                        if ((i + 1) % int(len(dataloaders[phase]) / 10) == 0 or i == 0) and verbose: 
                            print("%d..." % (i + 1), end="", flush=True)
                        
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
                             running_loss / len(dataloaders[phase].dataset))
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
            best_epoch = epoch
            # Save model state dict
            torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pth"))
        
        if verbose:
            print()
    
    if verbose:
        duration = time.time() - start_time
        duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
        print("Training took %s." % duration_msg)
        print("Best validation {} = {:.3f} at epoch {}.".format(
                criterion_metric, best_val_criterion, best_epoch + 1))
        print("According validation loss = {:.3f}".format(history["val_loss"][best_epoch]),
              end="")
        for key in metrics.keys():
            if key == criterion_metric:
                continue
            print(" - {} = {:.3f}".format(key, history["val_"+key][best_epoch]), end="")
        print()
    
    # Load best model, and remove tmpdir if applicable
    best_model = copy.deepcopy(model)
    best_model.load_state_dict(torch.load(os.path.join(save_dir, "model_best.pth")))
    if model_dir is None:
        shutil.rmtree(save_dir)
    
    return best_model, history
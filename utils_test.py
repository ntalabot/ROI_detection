#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for testing with PyTorch.
Created on Thu Oct 25 16:09:50 2018

@author: nicolas
"""

import numpy as np
import matplotlib.pyplot as plt

import torch, torchvision

from utils_data import make_images_valid


def predict(model, dataloader, post_processing=None, discard_target=True):
    """Output predictions for the given dataloader and model."""
    predictions = []
    
    # Compute predictions
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if discard_target:
                batch = batch[0]
                
            predictions.append(model(batch))
    
    # Post-process them
    predictions = torch.cat(predictions)
    if post_processing is not None:
        predictions = post_processing(predictions)
    return predictions
    
    
def evaluate(model, dataloader, metrics):
    """Return the metric values for the given data and model."""
    values = {}
    for key in metrics.keys():
        values[key] = 0
    
    # Compute metrics over all data
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(dataloader):
            y_pred = model(batch_x)

            for key in metrics.keys():
                values[key] += metrics[key](y_pred, batch_y).item() * batch_x.shape[0]
            
    for key in metrics.keys():
        values[key] /= len(dataloader.dataset)
    return values


def show_sample(model, dataloader, n_samples=4, post_processing=None, metrics=None):
    """Display sample images of some inputs, predictions, and targets."""
    indices = np.random.randint(0, len(dataloader.dataset), n_samples)
    
    inputs = torch.stack([torch.from_numpy(dataloader.dataset[i][0]) for i in indices])
    targets = torch.stack([torch.from_numpy(dataloader.dataset[i][1]) for i in indices])
    
    with torch.no_grad():
        model.eval()
        preds = model(inputs)
    if post_processing is not None:
        post_preds = post_processing(preds)
    else:
        post_preds = preds
    
    # Modify inputs to make sure it is a valid image
    inputs = make_images_valid(inputs)
    
    height, width = inputs.shape[-2:]
    outs = torchvision.utils.make_grid(inputs, pad_value=1.0)
    outs_p = torchvision.utils.make_grid(post_preds.view([-1, 1, height, width]), pad_value=1.0)
    outs_t = torchvision.utils.make_grid(targets.view([-1, 1, height, width]), pad_value=1.0)
    
    if metrics is not None:
        for i, idx in enumerate(indices):
            print("Image %d: " % idx, end="")
            for key in metrics.keys():
                print("{} = {:.6}; ".format(key, metrics[key](preds[i].unsqueeze(0), 
                                                              targets[i].unsqueeze(0))))
        
    plt.figure(figsize=(10,8))
    plt.subplot(311); plt.title("Inputs")
    plt.imshow(outs.numpy().transpose([1,2,0]))
    plt.subplot(312); plt.title("Predictions")
    plt.imshow(outs_p.numpy().transpose([1,2,0]))
    plt.subplot(313); plt.title("Ground truths")
    plt.imshow(outs_t.numpy().transpose([1,2,0]))
    plt.tight_layout()
    plt.show()
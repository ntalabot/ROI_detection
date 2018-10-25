#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions/classes for data manipulation with PyTorch.
Created on Mon Oct 22 13:54:19 2018

@author: nicolas
"""

import os, warnings
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils import data

from utils_ROI_detection import imread_to_float


class ImageLoaderDataset(data.Dataset):
    """Dataset that loads image online for efficient memory usage."""
    
    def __init__(self, x_filenames, y_filenames, input_channels="RGB", transform=None, target_transform=None):
        """
        Args:
            x_filenames (list of str): contains the filenames/path to the input images
            y_filenames (list of str): contains the filenames/path to the target images
            input_channels (str): indicates the channels to load from the input images
            transform (callable): transformation to apply to input images
            target_transform (callable): transformation to apply to target images
        """
        super(ImageLoaderDataset, self).__init__()
        self.x_filenames = x_filenames
        self.y_filenames = y_filenames
        
        if len(self.x_filenames) != len(self.y_filenames):
            raise ValueError("Not the same number of files in input and target lists (%d != %d)." %
                            (len(self.x_filenames), len(self.y_filenames)))
            
        self.input_channels = input_channels
        if self.input_channels == "":
            raise ValueError("At least one input channel has to be used.")
        
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.x_filenames)
    
    def __getitem__(self, idx):
        # Load images to float in range [0,1]
        image = imread_to_float(self.x_filenames[idx], scaling=255)
        target = imread_to_float(self.y_filenames[idx], scaling=255)
        
        # Keep only relevant channels
        channel_imgs = {"R": image[:,:,0], "G": image[:,:,1], "B": image[:,:,2]}
        image = np.stack([channel_imgs[channel] for channel in self.input_channels], axis=0)
        
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target


def get_filenames(data_dir, valid_extensions = ('.png', '.jpg', '.jpeg')):
    """
    Return two lists with the input and target filenames respectively.
    
    The data directory is assumed to be organised as follow:
        data_dir:
            subdir1:
                rgb_frames: folder with input images
                seg_frames: folder with target images
            subdir2:
                rgb_frames: folder with input images
                seg_frames: folder with target images
            ...
    """
    if not isinstance(valid_extensions, tuple):
        valid_extensions = tuple(valid_extensions)
    x_filenames = []
    y_filenames = []
    
    for data_subdir in sorted(os.listdir(data_dir)):
        # Inputs
        for frame_filename in sorted(os.listdir(os.path.join(data_dir, data_subdir, "rgb_frames"))):
            if frame_filename.lower().endswith(valid_extensions):
                x_filenames.append(os.path.join(data_dir, data_subdir, "rgb_frames", frame_filename))
        # Targets
        for frame_filename in sorted(os.listdir(os.path.join(data_dir, data_subdir, "seg_frames"))):
            if frame_filename.lower().endswith(valid_extensions):
                y_filenames.append(os.path.join(data_dir, data_subdir, "seg_frames", frame_filename))
        break # XXX:### Take only 1 stack over whole dataset ###
        
    return x_filenames, y_filenames


def get_dataloaders(x_filenames, y_filenames, train_ratio, batch_size,
                    input_channels="RG", train_transform=None, valid_transform=None):
    """Return the train and validation dataloaders."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_train, x_valid, y_train, y_valid = train_test_split(x_filenames, y_filenames, 
                                                              train_size=train_ratio)
    
    train_dataset = ImageLoaderDataset(x_train, y_train, input_channels=input_channels, 
                                 transform=train_transform)
    valid_dataset = ImageLoaderDataset(x_valid, y_valid, input_channels=input_channels, 
                                 transform=valid_transform)
    
    train_loader = data.DataLoader(train_dataset,
                                   batch_size = batch_size,
                                   shuffle = True,
                                   num_workers = 1)
    valid_loader = data.DataLoader(valid_dataset,
                                   batch_size = batch_size,
                                   shuffle = False,
                                   num_workers = 1)
    return train_loader, valid_loader


## Image manipulations

def normalize_symmetric_range(images):
    """Normalize the given float images by changing their range from [0,1] to [-1,1]."""
    return images * 2.0 - 1.0


def make_images_valid(images):
    """Make sure the given images have correct value range and number of channels."""
    images = (images - images.min()) / (images.max() - images.min())
    if images.shape[1] == 2:
        shape = (images.shape[0], 1) + images.shape[2:]
        images = torch.cat([images, torch.zeros(shape, dtype=images.dtype)], 1)
    return images
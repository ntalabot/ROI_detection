#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions/classes for data manipulation with PyTorch.
Created on Mon Oct 22 13:54:19 2018

@author: nicolas
"""

import os
import numpy as np

import torch
from torch.utils import data

from utils_common.image import imread_to_float


class ImageLoaderDataset(data.Dataset):
    """Dataset that loads image online for efficient memory usage."""
    
    def __init__(self, x_filenames, y_filenames, input_channels="RGB", 
                 transform=None, target_transform=None):
        """
        Args:
            x_filenames: list of str
                Contains the filenames/path to the input images.
            y_filenames: list of str
                Contains the filenames/path to the target images.
            input_channels: str (default = "RGB")
                Indicates the channels to load from the input images, e.g. "RG"
                for Red and Green.
            transform: callable (default = None)
                Transformation to apply to the input images.
            target_transform: callable (default = None)
                Transformation to apply to the target images.
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


def get_filenames(data_dir, valid_extensions=('.png', '.jpg', '.jpeg')):
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
    
    Args:
        data_dir: str
            Directory/path to the data.
        valid_extensions: tuple of str (default = ('.png', '.jpg', '.jpeg'))
            Tuple of the valid image extensions.
    
    Returns:
        x_filenames, y_filenames: lists of str 
            Contain the input and target image paths respectively.
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
        
    return x_filenames, y_filenames


def get_dataloader(data_dir, batch_size, input_channels="RG", shuffle=True,
                   transform=None, target_transform=None, num_workers=1):
    """
    Return a dataloader with the data in the given directory.
    
    Args:
        data_dir: str
            Directory/path to the data (see get_filenames() for the structure).
        batch_size: int
            Number of samples to return as a batch.
        input_channels: str (default = "RG")
            Indicates the channels to load from the input images, e.g. "RG"
            for Red and Green.
        shuffle: bool (default = True)
            If True, the data is shuffled before being returned as batches.
        transform: callable (default = None)
            Transformation to apply to the input images.
        target_transform: callable (default = None)
            Transformation to apply to the target images.
        num_workers: int (default = 1)
            Number of workers for the PyTorch Dataloader.
    """
    x, y = get_filenames(data_dir)
    dataset = ImageLoaderDataset(x, y, input_channels=input_channels,
                                 transform=transform, target_transform=target_transform)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_all_dataloaders(data_dir, batch_size, input_channels="RG", test_dataloader=False,
                        train_transform=None, train_target_transform=None,
                        eval_transform=None, eval_target_transform=None):
    """
    Return a dataloader dictionary with the train, validation, and (optional) test dataloaders.
    
    Args:
        data_dir: str
            Directory/path to the data, it should contain "train/", "validation/",
            and (optional) "test/" subdirs (see get_filenames() for their structure).
        batch_size: int
            Number of samples to return as a batch.
        input_channels: str (default = "RG")
            Indicates the channels to load from the input images, e.g. "RG"
            for Red and Green.
        test_dataloader: bool (default = False)
            If True, the dictionary will contain the test_loader under "test".
        train_transform: callable (default = None)
            Transformation to apply to the train input images.
        train_target_transform: callable (default = None)
            Transformation to apply to the train target images.
        eval_transform: callable (default = None)
            Transformation to apply to the validation/test input images.
        eval_target_transform: callable (default = None)
            Transformation to apply to the validation/test target images.
    
    Returns:
        A dictionary with the train, validation and (optional) test dataloaders
        under the respective keys "train", "valid", and "test".
    """
    train_loader = get_dataloader(
            os.path.join(data_dir, "train/"),
            batch_size=batch_size,
            input_channels=input_channels,
            shuffle=True,
            transform=train_transform,
            target_transform=train_target_transform
    )
    valid_loader = get_dataloader(
            os.path.join(data_dir, "validation/"),
            batch_size=batch_size,
            input_channels=input_channels,
            shuffle=False,
            transform=eval_transform,
            target_transform=eval_target_transform
    )
    if test_dataloader:
        test_loader = get_dataloader(
                os.path.join(data_dir, "test/"),
                batch_size=batch_size,
                input_channels=input_channels,
                shuffle=False,
                transform=eval_transform,
                target_transform=eval_target_transform
        )
        return {"train": train_loader, "valid": valid_loader, "test": test_loader}
    else:
        return {"train": train_loader, "valid": valid_loader}


## Image manipulations

def normalize_range(images):
    """Normalize the given float image(s) by changing the range from [0,1] to [-1,1]."""
    return images * 2.0 - 1.0


def make_images_valid(images):
    """Make sure the given images have correct value range and number of channels."""
    # Set range from [min,max] to [0,1]
    images = (images - images.min()) / (images.max() - images.min())
    # If only 2 channel (e.g. "RG"), add an empty third one
    if images.shape[1] == 2:
        shape = (images.shape[0], 1) + images.shape[2:]
        zero_padding = torch.zeros(shape, dtype=images.dtype).to(images.device)
        images = torch.cat([images, zero_padding], 1)
    return images

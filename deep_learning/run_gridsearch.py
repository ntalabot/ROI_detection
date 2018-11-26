#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to launch a grid-search over hyperparamters.
Created on Wed Oct 31 16:38:54 2018

@author: nicolas
"""

import time
import numpy as np

from utils_common.script import Arguments
from utils_model import CustomUNet
import run_train

# Parameters
n_epochs = 10
synth_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

def main():
    print("Starting on %s\n\nResults over validation data (%d epochs):\n" % (time.ctime(), n_epochs))
    start_time = time.time()
    
    # Arguments for run_train
    args = Arguments(
            batch_size = 32,
            data_dir = "../dataset/",
            epochs = n_epochs,
            eval_test = False,
            input_channels = "R",
            learning_rate = 0.001,
            model_dir = None,
            no_gpu = False,
            save_fig = False,
            scale_dice = 4.0,
            seed = 1,
            synthetic_data = True,
            synthetic_ratio = None,
            timeit = False,
            verbose = False
    )
    model = None
    
    for synth_ratio in synth_ratios:
        args.synthetic_ratio = synth_ratio
        print("synth_ratio={: <4}".format(synth_ratio), end="", flush=True)
                    
        try:
            model = CustomUNet(len(args.input_channels), 
                               u_depth = 4,
                               out1_channels = 16, 
                               batchnorm = True)
            history = run_train.main(args, model=model)
            best_epoch = np.argmax(history["val_dice"])
            print(" | loss={:.6f} - dice={:.6f} - diC{:.1f}={:.6f}".format(
                    history["val_loss"][best_epoch], history["val_dice"][best_epoch],
                    args.scale_dice, history["val_diC%.1f" % args.scale_dice][best_epoch]))
            
        except RuntimeError as err: # CUDA out of memory
            print(" | RuntimeError ({})".format(err))
                            
    # If an error occured, this is not printed
    # TODO: is there a way to force this printing ? try-except does not work with KeyboardInterrupt
    duration = time.time() - start_time
    duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
    print("\nEnding on %s" % time.ctime())
    print("Script duration: %s." % duration_msg)


if __name__ == "__main__":
    main()
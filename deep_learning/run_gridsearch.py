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
u_depths = [1, 2, 3, 4, 5, 6]
out1_channels = [16]
learning_rates = [1e-3]
batch_sizes = [32]

def main():
    print("Starting on %s\n\nResults over validation data (%d epochs):" % (time.ctime(), n_epochs))
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
            timeit = False,
            verbose = False
    )
    model = None
    
    for u_depth in u_depths:
        for out1_c in out1_channels:
            print("\nu_depth={} - out1_c={}:".format(u_depth, out1_c))
            
            for lr in learning_rates:
                args.learning_rate = lr
                for bs in batch_sizes:
                    args.batch_size = bs
                    print("lr={:.0E} - bs={:d}".format(lr, bs), end="")
                    
                    try:
                        model = CustomUNet(len(args.input_channels), 
                                           u_depth = u_depth,
                                           out1_channels = out1_c, 
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
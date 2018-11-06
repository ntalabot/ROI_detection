#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to launch a grid-search over hyperparamters.
Created on Wed Oct 31 16:38:54 2018

@author: nicolas
"""

import time

from utils_common.script import Arguments
from utils_model import CustomUNet
import run_train

# Parameters
n_epochs = 5
u_depths = [1, 2, 3, 4]
out1_channels = [8, 16, 32]
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [16, 32, 64]

def main():
    print("Starting on %s\n\nResults over test data (%d epochs):" % (time.ctime(), n_epochs))
    start_time = time.time()
    
    # Arguments for run_train
    args = Arguments(
            batch_size = 32,
            data_dir = "../dataset/",
            epochs = n_epochs,
            input_channels = "RG",
            learning_rate = 0.001,
            model_def = None,
            model_dir = None,
            no_test = False,
            no_gpu = False,
            save_fig = False,
            seed = 1,
            train_ratio = 0.8,
            timeit = False,
            verbose = False
    )
    model = None
    
    try:
        for u_depth in u_depths:
            for out1_c in out1_channels:
                model = CustomUNet(len(args.input_channels), out1_channels=out1_c, 
                                   u_depth=u_depth, batchnorm=True)
                print("\nu_depth={} - out1_c={}:".format(u_depth, out1_c))
                for lr in learning_rates:
                    args.learning_rate = lr
                    for bs in batch_sizes:
                        args.batch_size = bs
                        
                        try:
                            test_metrics = run_train.main(args, model)
                            print("lr={:.0E} - bs={:d} | loss={:.6f} - dice={:.6f}".format(
                                    lr, bs, test_metrics["loss"], test_metrics["dice"]))
                        except RuntimeError as err: # CUDA out of memory
                            print("lr={:.0E} - bs={:d} | RuntimeError "
                                  "({})".format(lr, bs, err))
    # If an error occured, still print the elapsed time
    except Exception as err:
        print("\nAn unexpected error occured.")
        duration = time.time() - start_time
        duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
        print("\nEnding on %s" % time.ctime())
        print("Script duration: %s.\n" % duration_msg)
        raise err
    
    duration = time.time() - start_time
    duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
    print("\nEnding on %s" % time.ctime())
    print("Script duration: %s." % duration_msg)


if __name__ == "__main__":
    main()
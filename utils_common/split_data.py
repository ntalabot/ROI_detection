#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split the data subfolders of "dataset/" into train, validation and test set.
"""

import os, shutil
import numpy as np

train_ratio = 0.6
valid_ratio = 0.2
# test_ratio = 1 - (train_ratio + valid_ratio)

if __name__ == "__main__":
    exp_names = sorted(os.listdir("../dataset/"))
    a1_list = [exp_name for exp_name in exp_names if exp_name.startswith("A1")]
    man_list = [exp_name for exp_name in exp_names if exp_name.startswith("MAN")]
    mdn_list = [exp_name for exp_name in exp_names if exp_name.startswith("MDN")]

    for exp_list in [a1_list, man_list, mdn_list]:
        n_exp = len(exp_list)
        indices = np.random.permutation(n_exp)
        train_idx = indices[:int(np.ceil(train_ratio * n_exp))]
        valid_idx = indices[int(np.ceil(train_ratio * n_exp:
                            int((train_ratio + valid_ratio) * n_exp)))]
        test_idx = indices[int((train_ratio + valid_ratio) * n_exp)]

        for idx in train_idx:
            shutil.move(os.path.join("../dataset/" + exp_list[idx], "../dataset/train"))
        for idx in valid_idx:
            shutil.move(os.path.join("../dataset/" + exp_list[idx], "../dataset/validation"))
        for idx in test_idx:
            shutil.move(os.path.join("../dataset/" + exp_list[idx], "../dataset/test"))

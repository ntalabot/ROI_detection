#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:53:08 2018

@author: nicolas
"""

from multiprocessing import Process, Manager


def run_parallel(*fns):
    """Run the called functions in parallel (use lambda keyword if needed)."""
    manager = Manager()
    return_dict = manager.dict()
    def fn_return(fn, i, return_dict):
        return_dict[i] = fn()
    
    proc = []
    for i, fn in enumerate(fns):
        p = Process(target=fn_return, args=(fn, i, return_dict))
        p.start()
        proc.append(p)
    for p in proc:
        p.join()
    
    returns = []
    for i in range(len(fns)):
        returns.append(return_dict[i])
    return returns
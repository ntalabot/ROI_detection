#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing functions useful for python scripts.
Created on Thu Nov  1 16:03:09 2018

@author: nicolas
"""

class Arguments:
    """Container object that stores arguments through its attributes.
    
    Useful for simulating arguments passed through command line to a script."""
    
    def __init__(self, **kwargs):
        """Create attributes equivalent to keyword arguments."""
        for key, value in kwargs.items():
            self.__setattr__(key, value)
    
    def __str__(self):
        """Return a string showing the different attributes and their values."""
        string = "%s(" % self.__class__.__name__
        for key, value in vars(self).items():
            string += key
            if isinstance(value, str):
                string += "='%s', " % value.replace("'", r"\'")
            else:
                string += "={}, ".format(value)
        string += "\b\b)"
        return string
            
    def __repr__(self):
        """Simply returns self.__str__()."""
        return self.__str__()
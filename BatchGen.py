# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 19:13:18 2018

@author: Pranav Devarinti
"""
import numpy as np
import pandas as pd
def GenXY(ds,splits):
    ds = np.array(ds)
    x_list = []
    x_list.append(np.hsplit(ds,splits))
    x_list = np.array(x_list[0])
    print(x_list.shape)
    y_list = x_list[1:,7]
    return x_list,y_list
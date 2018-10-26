# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 15:45:44 2018

@author: Pranav Devarinti
"""
import pandas as pd
import numpy as np
import os

def getcombineall(ptr):
    dry = os.listdir(ptr)
    datasets = []
    for i in dry:
        j = pd.read_csv(os.path.join(ptr,i))
        j = j.transpose()
        datasets.append(j.values.tolist())
    datasets = np.array(datasets)
    datasets = datasets[:,0,:]
    datasets = datasets[:,1:]
    return datasets

y = getcombineall('RawData\GoogleTrends\CMCSA')

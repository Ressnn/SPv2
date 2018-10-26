# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:16:59 2018

@author: ASUS
"""

import pandas as pd
import numpy as np
import TrendCombine
import Synth
import BatchGen
import GenerateModel
from BatchGen import GenXY
from Synth import rescale
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from MakeDiffrence import UndoDif
Utrends = TrendCombine.getcombineall(r'RawData\GoogleTrends\AMZN')
Stockpath = r'RawData\StockData\AMZN.csv'
splits = 2
size = 250
loadweights = True
loadweightpath = 'model.h5'
scale = MinMaxScaler(feature_range=(-1,1))   
y,r = rescale(Utrends,Stockpath)
a = np.array(y).reshape(4,-1)
a = a[:,0:size]
c = pd.read_csv(Stockpath)
c = c.transpose()
c = np.array(c[:-2])
frames = [a[:,:size],c[1:,:size]]
result = np.append(frames[0],frames[1],axis=0)
result = result.tolist()
result = pd.DataFrame(result).transpose()


result = np.array(result)
result = scale.fit_transform(result)
result = pd.DataFrame(result).diff()
result = pd.DataFrame(result).transpose()

result = np.array(result)
result = result[:,1:]
result = np.append(result,result,axis=1)
result = result[:,:size]

x_l,y_l = GenXY(result,splits)
result = pd.DataFrame(result)
result.to_csv('RawData/SD/AMZN.csv')

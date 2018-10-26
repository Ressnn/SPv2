# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 16:33:34 2018

@author: Pranav Devarinti
"""
import pandas as pd
import numpy as np
import TrendCombine
import Synth
import BatchGen
import GenerateModel
from BatchGen import GenXY
from Synth import rescale
import keras
from keras.layers import LSTM,GRU
from keras.models import Sequential
from keras.layers import Dropout, Dense
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

result = np.array(result)
model = GenerateModel.build(int(size/splits),loadweights,loadweightpath)
cc = 0
xz = []
for i in x_l:
    i = pd.DataFrame(i)
    i = i.transpose()
    xz.append(np.array(i))
    
    cc=cc+1
xz = np.array(xz)
v = []

scale.fit_transform(y_l)
for i in range(0,splits-1):
    model.fit(xz[i].reshape(1,xz.shape[1],xz.shape[2]),y_l[i].reshape(1,-1,1),epochs=1)
    future=[]
    future.append(model.predict(xz[i+1].reshape(1,xz.shape[1],xz.shape[2])))
    seeing = []
    seeing.append(model.predict(xz[i].reshape(1,xz.shape[1],xz.shape[2])))
    
    v.append(model.predict(xz[-2].reshape(1,xz.shape[1],xz.shape[2])))
    model.save_weights('model.h5')



# In[]
    '''
for i in (v[1].reshape(-1),np.array(seeing).reshape(-1)):
    z = (i[0]-i[1])*(i[0]-i[1])''' 
# In[]
model.save_weights('model.h5')
real = UndoDif(result[4])
rz = []
vz = []
for i in range(int(size/splits),size):
    rz.append(i)
    vz.append(i+(size/splits))

se = np.array(seeing[0]).reshape(-1)
se = UndoDif(se)
e = []
cs = []
for i in se:
    e.append(i+real[int(size-(size/splits))])
future = np.array(future).reshape(-1)
future = UndoDif(future)
for i in future:
    cs.append(i+real[-1])
cs = np.array(cs)
# In[]
plt.plot(UndoDif(result[4]))
plt.plot(np.array(rz).reshape(-1,),e)
plt.plot(vz,cs.reshape(-1))

for i in v:
    plt.plot(np.array(rz).reshape(-1,),UndoDif(i.reshape(-1))+real[int(size-(size/splits))])
model.save_weights('model.h5')
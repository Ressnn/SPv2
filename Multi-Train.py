# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 15:52:30 2018

@author: ASUS
"""
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense,LSTM,Flatten,InputLayer,Dropout,CuDNNLSTM
from keras.models import Sequential
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
# In[]

directory = r'RawData/SD'
predictdir = r'RawData/DNS/DAL.csv'
Use_Cuda = True
load_w = False

batches = 2
timesteps = 250
features = 8
fullsize = 251
start = fullsize-timesteps
lastvar = timesteps-(timesteps/batches)
nsub = int(timesteps/batches)
lstmlist = [1024,512,256]
denselist = [512,1]

if Use_Cuda == True:
    from keras.layers import CuDNNLSTM

def buildmodel(lstmlist,denselist,cuda):
    model = Sequential()
    if cuda == False:
        for i in lstmlist:
            model.add(LSTM(i,return_sequences = True))
            model.add(Dropout(.15))
        
        for a in denselist:
            model.add(Dense(a))
        model.compile(optimizer='Nadam',loss = 'mse')
        return model
    if cuda == True:
        for i in lstmlist:
            model.add(CuDNNLSTM(i,return_sequences = True))
            model.add(Dropout(.15))
        
        for a in denselist:
            model.add(Dense(a))
        model.compile(optimizer='Nadam',loss = 'mse')
        return model

class Data:
    datalist = os.listdir(directory)
    ddr = []
    Datacsv = []
    Split = []
    Batched = []
    Testdata = []
for i in Data.datalist:
    Data.ddr.append(os.path.join(directory,i))
for i in Data.ddr:
    Data.Datacsv.append(np.array(pd.read_csv(i))[:,start:])
for i in Data.Datacsv:
    i = np.array(pd.DataFrame(i).transpose())
    Data.Split.append(np.vsplit(i,batches))
for i in np.array(Data.Split):
    
    counter = 0
    while counter+1 != i.shape[0]:
        r = []
        r.append(i[counter])
        r.append(i[counter+1])
        Data.Batched.append(r)
        counter = counter + 1
z = np.array(Data.Batched)
print("Finished working with data")
model = buildmodel(lstmlist,denselist,Use_Cuda)
print("Finished building model, testing...")
z = np.array(z)
# In[]
model.build()
if load_w == True:
    try:
        for i in z:
            model.fit(i[0].reshape(-1,nsub,features),i[1,:,4].reshape(-1,nsub,1),epochs=1)
            model.load_weights('wheights')
            print('load sucessful')
    except:
        print('load failed')
# In[]
c = 0
seeing = []
for i in z:
    model.fit(i[0].reshape(-1,nsub,features),i[1,:,4].reshape(-1,nsub,1),epochs=150)   
    
    for a in z:
        seeing.append(model.predict(a[0].reshape(-1,nsub,features)))
model.save_weights('wheights')
# In[]
Data.test = np.array(pd.read_csv(predictdir).transpose())
Data.test = np.array(Data.test[start:,:])
Data.test = np.vsplit(Data.test,batches)
datatopred = np.array(Data.test)
# In[]
style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
xs = []
ys = []
def animate(x,y):
    ax1.clear()
    xs.append(x)
    ys.append(y)
    ax1.plot(xs,ys)
    
bigr = []
data_pred = model.predict(datatopred)
finals = []

for i in range(0,1001):
    bigr.append(i)
yee = datatopred[:,:,6]
yee = yee.reshape(-1)
yee = yee.tolist()

for i in zip(bigr[0:timesteps],yee):
    animate(i[0],i[1])
    
    
plt.show()
# In[]

class predictme:
    def getdata(path):
        return np.array(np.split(np.array(pd.read_csv(path)).transpose()[1:,:],batches))
    def predict_data(batched_data):
        return model.predict(batched_data).reshape(-1)
    def fp(path):
        return predictme.predict_data(predictme.getdata(path))
    def undodiff(ltud,start):
        nl = [start]
        for i in ltud:
            nl.append(nl[-1]+i)
        return nl
    def getpredandundo(path,hstart):
        data = predictme.fp(path)
        return predictme.undodiff(data,hstart)
    def getys(path):
        return np.array(predictme.getdata(path)[:,:,6]).reshape(-1)
    def getundoys(path):
        return predictme.undodiff(predictme.getys(path),0)
    def rangelist(start,stop):
        nl = []
        for i in range(start,stop):
            nl.append(i)
        return nl
ys = predictme.getundoys(predictdir)
plt.plot(predictme.rangelist(nsub,timesteps+nsub+1) ,np.array(predictme.getpredandundo(predictdir,ys[nsub])).reshape(-1))
plt.plot(predictme.getundoys(predictdir))



# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 19:31:28 2018

@author: Pranav Devarinti
"""
import keras
import numpy
from keras.layers import LSTM,GRU
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
def build(size,loadweights,lopath):
    model = Sequential()
    model.add(LSTM(200,return_sequences=True,input_shape=(size,8),activation = 'relu'))

    model.add(LSTM(300,return_sequences=True,activation = 'relu'))

    model.add(LSTM(450,return_sequences=True,activation = 'relu'))

    model.add(LSTM(500,return_sequences=True,activation = 'relu'))
    model.add(Dropout(.4))
    model.add(Dense(250,activation = 'relu'))
    model.add(Dense(100,activation = 'relu'))
    model.add(Dense(75,activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='Nadam')
    model.build()
    if loadweights == True:
        model.load_weights(lopath)
    print("model has been built")
    
    return model
def train(x,y):
    model.fit(x,y)
def test(x,y):
    return model.evaluate(x,y)
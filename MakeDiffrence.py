# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 08:05:25 2018

@author: Pranav Devarinti
"""

def Makediffrence(l1):
    nl = []
    for i in l1:
        counter = 0
        for z in i:
            if counter == 0:
                nl.append(int(0))
            else:
                nl.append(z-i[counter+1])
                print("v")
            counter = counter+1
    return nl

def UndoDif(l1):
    nl = []
    counter=0
    value = 0
    for i in l1:
        counter = counter+1
        value = value+i
        nl.append(value)
    return nl
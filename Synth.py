# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 17:17:45 2018

@author: Pranav Devarinti
"""
import pandas as pd
import numpy as np
def finder(list1,list2):
    shape1 = list1.shape()[-1]
    shape2 = list2.shape()[-1]
    return (shape1,shape2)
def getcsv(path):
    z = pd.read_csv(path)
    return z.transpose() 

def rescale(spath,l2):
    s = spath
    counter = 0
    new_list = []
    for i in s:
        
        counter = 0
        for a in i:
            counter = counter+1
            if counter % 9 == 0:
                for b in range(0,4):
                    new_list.append(a)
            else:
                for b in range(0,5):
                    new_list.append(a)
                    
    ns = np.array(new_list).shape
    print(s.shape[1])
    rz = int(ns[0]/s.shape[1])
    
    Check = True
    
    while Check == True:
        if rz<s.shape[0]:
            print("Check In Progress")
            new_list = new_list[0:-2]
            rz = int(ns[0]/s.shape[1])
        else:
            Check == False
            print("Done")
            break
    return new_list,ns
                    
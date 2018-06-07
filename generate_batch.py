#!/usr/bin/python
# -*-coding:utf-8-*-
import numpy as np

'''
To generate the batch for training and testing shoul be an important and independent part of the model.
Here is a very simple example.
'''
def generate(size):
    x_train=np.random.uniform(-1.,1.,size)
    y_train=[1.6*x+0.04*x**2+0.02*math.cos(0.6*x+math.exp(2.9*x)) for x in x_train]
    return x_train,y_train

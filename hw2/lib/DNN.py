from numpy import *
import math
import json
import codecs
import matplotlib.pyplot as plt
import random as rd
import csv
import sys

class Config:
    def __init__(self, LR, nEpoch, ep, RC, batch = 0):
        self.LearningRate = LR
        self.Epoch = nEpoch
        self.Epsilon = ep
        self.RegularizationConstant = RC 
        self.Batch = batch 

class Activation:
    def __call__(self,z):
        raise NotImplementedError("Subclass must implement abstract method")
    def backprop(self,a):
        raise NotImplementedError("Subclass must implement abstract method")

class RelU(Activation):
    def __call__(self,z):
        a = matrix(zeros(len(z))))
        for i in range(len(z)):
            if z[0,i] > 0:
                a[0,i] = z[0,i]
        return a
    def backprop(self,a):
        partial = matrix(zeros(len(a)))
        for i in range(len(a)): 
            if a[0,i] > 0:
                partial[0,i] = 1
        return partial
            
class Sigmoid(Activation):
    def __call__(self,z):
        a = matrix(zeros(len(z)))
        for i in range(len(z)):
            if z[0,i] > -700:
                a[0,i]= 1/(1 + exp(-z[0,i]))
            else:
                a[0,i]= 0
        return a
    def backprop(self,a):
        return a * (1 - a)

Act = {'sigmoid': Sigmoid(), 'relu':RelU()}

class NeuronLayer:
    def __init__(self, nNeuron, dimension, activation='relu'):
        self.W = matrix(random.rand(nNeuron, dimension))
        self.B = matrix(random.rand(nNeuron))
        self.act = Act[activation]
        self.a = matrix(zeros(nNeuron))
    def __call__(self, a_prev):
        z = self.W * a_prev + self.B
        self.a = self.act(z)
        return self.a
    def backprop(self):
        return self.act.backprop(self.a)
         
class DNN:
    def __init__(self, inputDimension):
        self._Layers = []
        self.inputDimension = inputDimension
        self.tempInputDimension = self.inputDimension
        self.Act = Act
        self.gradientC= {'W':[], 'B':[]}
    def add(self, nNeuron, activation='relu'):
        self._Layers.append(NeuronLayer(nNeuron, self.tempInputDimension), activation)
        self.gradientC['W'].append(zeros(self._Layers[-1].W.shape))
        self.gradientC['B'].append(zeros(self._Layers[-1].B.shape))
        self.tempInputDimension = nNeuron
    def cleanGradientC(self):
        for i in range(len(self.gradientC['W'])):
            (self.gradientC['W'])[i] = zeros((self.gradientC['W'])[i].shape)
            (self.gradientC['B'])[i] = zeros((self.gradientC['B'])[i].shape)
    def propagate(self, x): #TODO compute the self.a in each layer
        temp = x
        for i in range(len(self._Layers)):
            temp = self._Layers[i](temp)
        return temp
    def backprop(self, x, yhead): #TODO compute all the delta and update gradientC
        Layers = self._Layers
        y = Layers[-1].a
        if y == 0:
            y = 1e-10
        if y == 1:
            y = 1 - 1e-10
        gradientCrYr = matrix(-(yhead / (y) - (1-yhead) / (1-y)))
        deltaL = multiply(Layers[-1].backprop(),gradientCrYr)
        grandientC['W'][-1][0] = multiply(Layers[-2].a, deltaL[0,0] + zeros(Layers[-2].a.shape))
        for i in reversed(range(len(self._Layers))):
    def train(self, X, Y, config): #TODO train with a single batch and update parameter
    def compile(self, X, Y, config):
        for i in range(config.Epoch):



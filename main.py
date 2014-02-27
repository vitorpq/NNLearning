#/usr/bin/env python
# -*- coding: utf-8 -*-

"""Neural Networks code for learning
Implementations:
1- Perceptron (1 neuron)
2- MLP
5- Classification of breast cancer
6- Classification of dataset similar to mine
7- Classification similar to mine and Isabel
3- RBF
4- LVQ
"""
#del data, inp, minmax, target
import neurolab as nl
from numpy import loadtxt, array
from pddin import pddin
# Load data from file
# First column: time (1s)
# Second column: input
# Third column: output

data = loadtxt('data/exchanger.dat', delimiter=';')
inp = pddin(array([data[:,1]]).T, 2)

target = array([data[1:,2]]).T
minmax = [[inp.min(), inp.max()],[inp.min(), inp.max()]]

net = nl.net.newp(minmax, 1)
error = net.train(inp, target, epochs=100, show=10, lr=0.1)

import pylab as pl
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('Train error')
pl.grid()
pl.show()

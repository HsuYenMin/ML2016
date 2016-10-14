import json
import codecs
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import csv
import sys
# settings
LR          = 0.000000027           # learning rate
if sys.argv[1] == 'kaggle_best.csv': 
    nEpoch  = 100000
else:
    nEpoch  = 5000                  # # of epoch
epsilon     = 10**(-8)              # AdaGrad & AdaDelta
RC          = 100                   # regularization 
PlotName    = 'plot_0.png'
PLOT_STORE  = True                  #
filename    = 'train.csv'           #
outputFilename = sys.argv[1] 

# data processing
table = []
with codecs.open(filename,"r", encoding='big5', errors='ignore') as f:
        table = [line.strip().split(',') for line in f]
table = table[1:]

for i in range(18):
    for j in range(18, len(table), 18):
        if table[i][2] == table[i + j][2]:
            table[i] = table[i] + table[i + j][3:]

for i in range(len(table)):
    table[i] = table[i][3:]

table = table[:18]

for i in range(len(table[10])):
    if table[10][i] == "NR":
        table[10][i] = "0"
# generating X and Y

X = []
Y = []
throw = False
for i in range(0, len(table[0]) - 9):
    y = float(table[9][i + 9])
    x = [float(s) for s in table[9][i:i + 9]]
    for s in x:
        if s < 0:
            throw = True
            break
    if (y>0 and (not throw)): 
        X.append(np.array(x))
        Y.append(float(table[9][i + 9]))

# generating W, and W[-1] is B
W = np.array([random.random() for i in range(len(X[0]) + 1)])
Loss = []
L = np.array( [( Y[j] - W[-1] - np.inner(W[:-1],X[j]) ) for j in range(len(X)) ] ) 
Loss.append( np.inner(L,L) / len(X) )
print("Average Loss = ", Loss[0], " before training")

Delta = np.zeros(len(X[0]) + 1)
for ii in range(nEpoch): 
#   compute Delta and change parameters 
    for j in range(len(X[0])):
        g_j = (-2) * math.fsum(L * np.array([X[k][j] for k in range(len(X))])) + 2 * RC *W[j]
        Delta[j] = - LR * g_j
        W[j] = W[j] + Delta[j]
    g_j = (-2) * (math.fsum(L))
    Delta[-1] = - LR * g_j
    W[-1] = W[-1] + Delta[-1]
    L = np.array( [( Y[j] - W[-1] - np.inner(W[:-1],X[j]) ) for j in range(len(X)) ] ) 
    Loss.append(np.inner(L,L) / len(X))
    print("Average Loss = %5.2f after epoch %4d " % (Loss[ii + 1], ii ))

# Plot the loss
if(PLOT_STORE):
    plt.ylim(0,100)
    plt.plot(Loss, lw = 2)
    plt.xlabel("epoch")
    plt.ylabel("Average Loss")
    plt.savefig(PlotName, bbox_inches='tight')

# data processing
table = []
with codecs.open('test_X.csv',"r", encoding='big5', errors='ignore') as f:
    table = [line.strip().split(',') for line in f]
for i in range(len(table)):
    table[i] = table[i][2:]

# generating X
X = []
throw = False
for i in range(0, len(table), 18):
    x = [float(s) for s in table[i + 9]]
    X.append(np.array(x))

Y = [ ( W[-1] + np.inner(W[:-1],X[j]) ) for j in range(len(X)) ]

with open(outputFilename, 'w') as csvfile:
    fieldnames = ['id','value']
    w = csv.DictWriter(csvfile,fieldnames=fieldnames)
    w.writeheader()
    for i in range(len(Y)):
        w.writerow({'id':'id_' + str(i), 'value': Y[i]})

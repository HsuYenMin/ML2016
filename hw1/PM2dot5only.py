import json
import codecs
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
import math
import csv
import sys
# settings
LR          = 0.000000027           # learning rate
nEpoch      = 500                  # # of epoch
epsilon     = 10**(-8)              # AdaGrad & AdaDelta
rho         = 1                     #           AdaDelta
RC          = 100                     # regularization 
PLOT_STORE  = True                  #
filename    = 'train.csv'           #
PlotName    = 'LearningRate_0.png'      #
readModel   = False                 #
ModelName   = 'lastHope_0.json'     #
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
'''usefulRow = [9]
# usefulRow = [9,8,5,6,12,7,13]

# convert to numpy array and normalize
for i in range(len(table)):
    table[i] = np.array([float(s) for s in table[i]])
    if i == 9:
        Y = table[9][9:] 
#    table[i] = (table[i] - np.average(table[i])) / np.std(table[i]) 

# generating X and Y
X = []
throw = False
for i in range(len(table[0]) - 9):
    x = np.array([])
    for j in usefulRow:
        x = np.concatenate((x,table[j][i:i + 9]))
    X.append(x)
'''
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

# nomalize
'''
for i in range(len(X)):
    if np.average(X[i]) == 0:
        print(i)
    X[i] = (X[i] - np.average(X[i])) / np.std(X[i])
'''

# generating W
W = np.array([random.random() for i in range(len(X[0]) + 1)])
Loss = []
# W[-1] is B
L = np.array( [( Y[j] - W[-1] - np.inner(W[:-1],X[j]) ) for j in range(len(X)) ] ) 
Loss.append( np.inner(L,L) / len(X) )
print("Average Loss = ", Loss[0], " before training")

################ start training! ####################
G = np.zeros(len(X[0]) + 1)    # AdaGrad & AdaDelta
D = np.zeros(len(X[0]) + 1)    #           AdaDelta
Delta = np.zeros(len(X[0]) + 1)#           AdaDelta

for ii in range(nEpoch): 
#   compute Delta and change parameters 
    for j in range(len(X[0])):
        g_j = (-2) * math.fsum(L * np.array([X[k][j] for k in range(len(X))])) + 2 * RC *W[j]
#        lr = LR * math.sqrt(D[j] + epsilon)/ math.sqrt(G[j] + epsilon)
#        G[j] = rho * G[j] + (1 - rho) * g_j * g_j
        Delta[j] = - LR * g_j
        W[j] = W[j] + Delta[j]
    g_j = (-2) * (math.fsum(L))
#    lr = LR * math.sqrt(D[-1] + epsilon) / math.sqrt(G[-1] + epsilon)
    Delta[-1] = - LR * g_j
    W[-1] = W[-1] + Delta[-1]
#    G[-1] = rho * G[-1] + (1 - rho) * g_j * g_j
    # compute D
#    for j in range(len(G)):# Adagrad
#        D[j] = rho * D[j] + (1 - rho) * Delta[j] * Delta[j]
    L = np.array( [( Y[j] - W[-1] - np.inner(W[:-1],X[j]) ) for j in range(len(X)) ] ) 
    Loss.append(np.inner(L,L) / len(X))
    print("Average Loss = %5.2f after epoch %4d " % (Loss[ii + 1], ii ))

# second round....
LR = 0.00000016
W = np.array([random.random() for i in range(len(X[0]) + 1)])
Loss2 = []
# W[-1] is B
L = np.array( [( Y[j] - W[-1] - np.inner(W[:-1],X[j]) ) for j in range(len(X)) ] ) 
Loss2.append( np.inner(L,L) / len(X) )
print("Average Loss = ", Loss2[0], " before training")

################ start training! ####################
G = np.zeros(len(X[0]) + 1)    # AdaGrad & AdaDelta
D = np.zeros(len(X[0]) + 1)    #           AdaDelta
Delta = np.zeros(len(X[0]) + 1)#           AdaDelta

for ii in range(nEpoch): 
#   compute Delta and change parameters 
    for j in range(len(X[0])):
        g_j = (-2) * math.fsum(L * np.array([X[k][j] for k in range(len(X))])) + 2 * RC *W[j]
#        lr = LR * math.sqrt(D[j] + epsilon)/ math.sqrt(G[j] + epsilon)
#        G[j] = rho * G[j] + (1 - rho) * g_j * g_j
        Delta[j] = - LR * g_j
        W[j] = W[j] + Delta[j]
    g_j = (-2) * (math.fsum(L))
#    lr = LR * math.sqrt(D[-1] + epsilon) / math.sqrt(G[-1] + epsilon)
    Delta[-1] = - LR * g_j
    W[-1] = W[-1] + Delta[-1]
#    G[-1] = rho * G[-1] + (1 - rho) * g_j * g_j
    # compute D
#    for j in range(len(G)):# Adagrad
#        D[j] = rho * D[j] + (1 - rho) * Delta[j] * Delta[j]
    L = np.array( [( Y[j] - W[-1] - np.inner(W[:-1],X[j]) ) for j in range(len(X)) ] ) 
    Loss2.append(np.inner(L,L) / len(X))
    print("Average Loss = %5.2f after epoch %4d " % (Loss2[ii + 1], ii ))

# third round....
LR = 0.00000000027
W = np.array([random.random() for i in range(len(X[0]) + 1)])
Loss3 = []
# W[-1] is B
L = np.array( [( Y[j] - W[-1] - np.inner(W[:-1],X[j]) ) for j in range(len(X)) ] ) 
Loss3.append( np.inner(L,L) / len(X) )
print("Average Loss = ", Loss3[0], " before training")

################ start training! ####################
G = np.zeros(len(X[0]) + 1)    # AdaGrad & AdaDelta
D = np.zeros(len(X[0]) + 1)    #           AdaDelta
Delta = np.zeros(len(X[0]) + 1)#           AdaDelta

for ii in range(nEpoch): 
#   compute Delta and change parameters 
    for j in range(len(X[0])):
        g_j = (-2) * math.fsum(L * np.array([X[k][j] for k in range(len(X))])) + 2 * RC *W[j]
#        lr = LR * math.sqrt(D[j] + epsilon)/ math.sqrt(G[j] + epsilon)
#        G[j] = rho * G[j] + (1 - rho) * g_j * g_j
        Delta[j] = - LR * g_j
        W[j] = W[j] + Delta[j]
    g_j = (-2) * (math.fsum(L))
#    lr = LR * math.sqrt(D[-1] + epsilon) / math.sqrt(G[-1] + epsilon)
    Delta[-1] = - LR * g_j
    W[-1] = W[-1] + Delta[-1]
#    G[-1] = rho * G[-1] + (1 - rho) * g_j * g_j
    # compute D
#    for j in range(len(G)):# Adagrad
#        D[j] = rho * D[j] + (1 - rho) * Delta[j] * Delta[j]
    L = np.array( [( Y[j] - W[-1] - np.inner(W[:-1],X[j]) ) for j in range(len(X)) ] ) 
    Loss3.append(np.inner(L,L) / len(X))
    print("Average Loss = %5.2f after epoch %4d " % (Loss3[ii + 1], ii ))

LR          = 0.00000025            # learning rate

##########  change optimizer to AdaDelta ##########
for ii in range(0): 
    #   compute Delta and change parameters 
    for i in range(len(Y)):
        for j in range(len(X[0])):
            l = Y[i] - W[-1] - np.inner(W[:-1],X[i]) 
            g_j = (-2) * l * X[i][j]  + 2 * RC *W[j]
            lr = LR * math.sqrt(D[j] + epsilon)/ math.sqrt(G[j] + epsilon)
            G[j] = rho * G[j] + (1 - rho) * g_j * g_j
            Delta[j] = - lr * g_j
            W[j] = W[j] + Delta[j]
        g_j = (-2) * l
        lr = LR * math.sqrt(D[-1] + epsilon) / math.sqrt(G[-1] + epsilon)
        Delta[-1] = - lr * g_j
        W[-1] = W[-1] + Delta[-1]
        G[-1] = rho * G[-1] + (1 - rho) * g_j * g_j
    #   compute D
        for j in range(len(D)):# Adagrad
            D[j] = rho * D[j] + (1 - rho) * Delta[j] * Delta[j]
    L = np.array( [( Y[j] - W[-1] - np.inner(W[:-1],X[j]) ) for j in range(len(X)) ] ) 
    Loss.append(np.inner(L,L) / len(X))
    print("Average Loss = %5.2f after epoch %4d " % (Loss[ii + 1], ii ))

# Plot the loss and save the model
if(PLOT_STORE):
    plt.ylim(0,100000)
    plt.plot(Loss,'b-' ,lw = 2)
    plt.plot(Loss2,'r-' , lw = 2)
    plt.plot(Loss3,'g-' , lw = 2)
    plt.xlabel("epoch")
    plt.ylabel("Average Loss")
    plt.savefig(PlotName, bbox_inches='tight')
'''
    model = {'W' : W[:-1].tolist() , 'B': W[-1]}
    with open(ModelName, 'w') as f:
        print(json.dumps(model), file=f)
'''
# data processing
'''
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
'''

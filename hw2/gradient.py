from __future__ import print_function
from PIL import Image
import numpy as np
import random

train = np.genfromtxt('train.csv', delimiter = ',', dtype = np.string_)
train = train[1:,3:]
day = train.shape[0]//18

y = []
for j in range(12):
    y += train[20*18*j+9,9:].astype(float).tolist()
    for i in range(1,20):
	    y += train[(20*j+i)*18+9,:].astype(float).tolist()

data = []
for i in range (7):
	data.append([])
for i in range (day):
	data[0] += train[i*18+5,:].astype(float).tolist()
	data[1] += train[i*18+6,:].astype(float).tolist()
	data[2] += train[i*18+7,:].astype(float).tolist()
	data[3] += train[i*18+8,:].astype(float).tolist()
	data[4] += train[i*18+9,:].astype(float).tolist()
	data[5] += train[i*18+12,:].astype(float).tolist()
	data[6] += train[i*18+13,:].astype(float).tolist()

x = []
for i in range(12):
    for j in range(471):
    	X = []
    	X += data[0][i*480+j+4:i*480+j+9]#5
    	X += data[1][i*480+j+4:i*480+j+9]#10
    	X += data[2][i*480+j+6:i*480+j+9]#13
    	X += data[3][i*480+j:i*480+j+9]#22
    	X += data[4][i*480+j:i*480+j+9]#31
    	X += data[5][i*480+j+6:i*480+j+9]#34
    	X += data[6][i*480+j+5:i*480+j+9]#38
    	x.append(X)
x = np.array(x)

par = x.shape[1]+1
G = np.zeros(par, dtype = float)
D = np.zeros(par, dtype = float)
parameter = np.zeros(par, dtype = float)
parameter[par-1] = -2
parameter[30] = 0.9
parameter[28] = -0.5
parameter[27] = 0.5
parameter[25] = -0.2
parameter[24] = 0.2
parameter[12] = 0.1
parameter[4] = 0.3
pointnum = 1
it = 200000//pointnum
avgloss = 0
lamda = 0.005
rho = 0.9
epsilon = 0.00001
for i in range (it):
	block = random.randrange(len(y)-pointnum)
	partial = np.zeros(par)
	for parano in range (par):
		for point in range (pointnum):
			if parano != par-1:
				partial[parano] += (y[block+point] - parameter[par-1] - np.inner(parameter[0:par-1], x[block+point])) * (-2*x[block+point, parano]) + 2*lamda*parameter[parano]
			else:
				partial[parano] -= y[block+point] - parameter[par-1] - np.inner(parameter[0:par-1], x[block+point])
	eta = 1- (1 - 0.001)/it*i
	G = rho*G + (1-rho)*np.square(partial)
	deplacement = -eta*partial*np.sqrt(D+epsilon)
	deplacement /= np.sqrt(G+epsilon)
	D = rho*D + (1-rho)*np.square(deplacement)
	parameter += deplacement
	#print(parameter[par-1])

	loss = 0
	for point in range (pointnum):
		element = y[block+point] - parameter[par-1] - np.inner(parameter[0:par-1], x[block+point])
		loss += element*element
	loss += lamda*np.inner(parameter[0:par-1], parameter[0:par-1])
	avgloss += loss
#print(parameter)
print(avgloss/it/pointnum)


'''
for i in range (20):
	element = parameter[162] + np.inner(parameter[0:162], x[len(y)-20+i])
	print('element',element)
	element = y[len(y)-20+i] - element
	loss = element*element
	print('loss',loss)
'''

test = np.genfromtxt('test_X.csv', delimiter = ',', dtype = np.string_)
test = test[:,2:]
num = test.shape[0]//18
answer = np.zeros(num)
ans = np.genfromtxt('answer.csv', dtype = float)

loss = 0
for i in range (num):
	tstdata = []
	tstdata += test[i*18+5,4:9].astype(float).tolist()
	tstdata += test[i*18+6,4:9].astype(float).tolist()
	tstdata += test[i*18+7,6:9].astype(float).tolist()
	tstdata += test[i*18+8,:].astype(float).tolist()
	tstdata += test[i*18+9,:].astype(float).tolist()
	tstdata += test[i*18+12,6:9].astype(float).tolist()
	tstdata += test[i*18+13,5:9].astype(float).tolist()
	
	answer[i] = parameter[par-1] + np.inner(parameter[0:par-1], tstdata)

loss = 0
for i in range(120,240):
	loss += (answer[i]-ans[i])*(answer[i]-ans[i])
print(loss)

f = open('data/' + str(int(loss)) + '.csv','w')
f.write('id,value\n')
for i in range(num):
	f.write('id_' + str(i) + ',' + str(answer[i]) + '\n')
f.close()
for i in range(120):
	loss += (answer[i]-ans[i])*(answer[i]-ans[i])
print(loss)
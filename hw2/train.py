from lib.util import * 
# settings
config = Config(0.000000027, 5000, 10**(-8), 100)
save = sys.argv[2]
# can the following 7 lines be wrapped
(X,Y,Ave,SD) = dataProcessingHW2(sys.argv[1])
'''
X1 = X[:len(X)//3]
X2 = X[len(X)//3:len(X)//3*2]
X3 = X[len(X)//3*2:]
Y1 = Y[:len(Y)//3]
Y2 = Y[len(Y)//3:len(Y)//3*2]
Y3 = Y[len(Y)//3*2:]
'''
(W,B,Loss) = train(X,Y, 200)
'''
plotLoss(Loss, save + '.png')
'''

model = {'W':W.tolist(), 'B':B, 'Ave':Ave.tolist(), 'SD':SD.tolist()}
with open(save, 'w') as f:
    print(json.dumps(model), file=f)


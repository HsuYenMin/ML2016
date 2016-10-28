from lib.DNN import * 

'''
# settings
config = Config(0.000000027, 5000, 10**(-8), 100)
# can the following 7 lines be wrapped
(X,Y) = dataProcessingHW2()
W = array([rd.random() for i in range(len(X[0]))])
B = rd.random()
Loss = []
L = computeLoss(W,B,X,Y)
Loss.append( L / len(X) )
print("Average Loss = ", Loss[0], " before training")
DeltaW = zeros(len(X[0]))
DeltaB = 0
LR = 0.000000027 # without normalization
LR = 0.0000027   # with normalization
for ii in range(200): 
#   compute Delta and change parameters 
    pL = partialLoss(W,B,X,Y) 
    DeltaW = - LR * pL[0]
    W = W + DeltaW
    DeltaB = - LR * pL[1]
    B = B + DeltaB
    L = computeLoss(W,B,X,Y)
    Loss.append(L / len(X))
    print("Average Loss = %5.6f after epoch %4d " % (Loss[ii + 1], ii ))
plotLoss(Loss, '02_4.png')
'''

from .Import import *
class Config:
    def __init__(self, LR, nEpoch, ep, RC, batch = 0):
        self.LearningRate = LR
        self.Epoch = nEpoch
        self.Epsilon = ep
        self.RegularizationConstant = RC 
        self.Batch = batch 

def dataProcessingHW2(fname = 'spam_data/spam_train.csv'):
    X = []
    Y = []
    nFeatures = 9
    mysteriousList = [19,21,5,51,14,17,15,55,23] # We get this with getAttributes() and getCorrCoef()
    ms = mysteriousList[:nFeatures]
    # data processing
    with codecs.open(fname,"r", encoding='big5', errors='ignore') as f:
        table = [line.strip().split(',') for line in f]
    for i in range(len(table)):
        table[i] = table[i][1:]
    if nFeatures == 0:
        ms = arange(len(table[0]) - 1).tolist()
    # generating X and Y
    for i in range(0,len(table)):
        x = []
        for index in ms:
            x.append(float(table[i][index]))
        X.append(array(x))
        Y.append(float(table[i][-1]))
    Ave = []
    SD = []
    attri = getAttributesHW2()
    for index in ms:
        Ave.append(average(attri[index]))
        SD.append(std(attri[index]))
    Ave = array(Ave)
    SD = array(SD)
    for i in range(len(X)):
        X[i] = (X[i] - Ave) / SD
    return (X,Y,Ave,SD)

def testDataProcessHW2(Ave, SD, fname='spam_data/spam_test.csv'):
    Xtest = []
    nFeatures = 9
    mysteriousList = [19,21,5,51,14,17,15,55,23] # We get this with getAttributes() and getCorrCoef()
    ms = mysteriousList[:nFeatures]
    with codecs.open(fname,"r", encoding='big5', errors='ignore') as f:
        testTable = [line.strip().split(',') for line in f]
    for i in range(len(testTable)):
        testTable[i] = testTable[i][1:]
    for i in range(0,len(testTable)):
        x = []
        for index in ms:
            x.append(float(testTable[i][index]))
        Xtest.append(array(x)) 
    for i in range(len(Xtest)):
        Xtest[i] = (Xtest[i] - Ave) / SD
    return Xtest

def getAttributesHW2():
    # data processing
    with codecs.open('spam_data/spam_train.csv',"r", encoding='big5', errors='ignore') as f:
       table = [line.strip().split(',') for line in f]
    for i in range(len(table)):
        table[i] = table[i][1:]
    attributes = []
    for i in range(1, len(table[0]), 1):
        a = []
        for j in range(len(table)):
            a.append(float(table[j][i]))
        attributes.append(array(a))
    return attributes

def getValue(item):
    return fabs(item[1])

def getCorrCoef(attributes):
    colist = []
    for i in range(len(attributes)):
        corr = corrcoef(attributes[i],attributes[-1])
        colist.append(corr[0,1])
    index =  []
    for i in range(len(attributes)):
        index.append(i)
    li = list(zip(index,colist))
    usefulData = sorted(li, key = getValue)
    return usefulData.inverse()

def plotLoss(Loss, PlotName):
    plt.ylim(min(Loss),max(Loss))
    plt.plot(Loss, lw = 2)
    plt.xlabel("epoch")
    plt.ylabel("Average Loss")
    plt.savefig(PlotName, bbox_inches='tight')

def Sigmoid(z):
    if z > -700:
        a = 1/(1 + exp(-z))
    else:
        a = 0
    return a

def computeLoss(W, B, X, Y):
    loss = 0
    for i in range(len(X)):
        fxn = Sigmoid(inner(W,X[i]) + B)
        if fxn == 0:
            fxn = 1e-10
        if fxn == 1:
            fxn -= 1e-10
        loss += -(Y[i] * log(fxn) + (1 - Y[i]) * log(1 - fxn))
    return loss

def partialLoss(W, B, X, Y):
    PartialLoss = []
    for i in range(len(W)):
        Sumyn_fwb = []
        xilist = []
        for j in range(len(X)):
            xilist.append(X[j][i])
        xilist = array(xilist)
        for j in range(len(X)):
            fxn = Sigmoid(inner(W,X[j]) + B)
            Sumyn_fwb.append((Y[j] - fxn))
        Sumyn_fwb = array(Sumyn_fwb)
        PartialLoss.append(-inner(Sumyn_fwb,xilist))
    Sumyn_fwb = 0
    for i in range(len(X)):
        fxn = Sigmoid(inner(W,X[i]) + B)
        Sumyn_fwb += -(Y[i] - fxn)
    return (array(PartialLoss), Sumyn_fwb)

def train(X, Y, Epoch):
    W = array([random.random() for i in range(len(X[0]))])
    B = random.random()
    Loss = []
    L = computeLoss(W,B,X,Y)
    Loss.append( L / len(X) )
    print("Average Loss = ", Loss[0], " before training")
    DeltaW = zeros(len(X[0]))
    DeltaB = 0
    LR = 0.000000027 # without normalization
    LR = 0.00000047   # with normalization
    for ii in range(Epoch): 
        #   compute Delta and change parameters 
        pL = partialLoss(W,B,X,Y) 
        DeltaW = - LR * pL[0]
        W = W + DeltaW
        DeltaB = - LR * pL[1]
        B = B + DeltaB
        L = computeLoss(W,B,X,Y)
        Loss.append(L / len(X))
        print("Average Loss = %5.6f after epoch %4d " % (Loss[ii + 1], ii ))
        if ii>100 and Loss[-1]>Loss[-2]:
            break
    return (W,B,Loss)

def trainAdaDelta(X, Y, Epoch):
    epsilon     = 10**(-8)              # AdaGrad & AdaDelta
    LR          = 0.000025                # learning rate
    rho         = 0.15                  #           AdaDelta
    RC          = 100                   # regularization 
    G = zeros(len(X[0]))                # AdaGrad & AdaDelta
    D = zeros(len(X[0]))                #           AdaDelta
    GB = zeros(1)
    DB = zeros(1)
    W = array([random.random() for i in range(len(X[0]))])
    B = random.random()
    Loss = []
    L = computeLoss(W,B,X,Y)
    Loss.append( L / len(X) )
    print("Average Loss = ", Loss[0], " before training")
    DeltaW = zeros(len(X[0]))
    DeltaB = 0
    for ii in range(Epoch): 
        #   compute Delta and change parameters 
        pL = partialLoss(W,B,X,Y)
        lr = LR * sqrt(D + epsilon) / sqrt(G + epsilon)
        D = rho * D + (1 - rho) * square(DeltaW)
        G = rho * G + (1 - rho) * square(pL[0])
        DeltaW = - lr * pL[0]
        W = W + DeltaW 

        lr = LR * math.sqrt(DB + epsilon) / math.sqrt(GB + epsilon)
        DeltaB = - lr * pL[1]
        B = B + DeltaB
        DB = rho * DB + (1 - rho) * square(DB)
        GB = rho * GB + (1 - rho) * square(pL[1])
        L = computeLoss(W,B,X,Y)
        Loss.append( L / len(X))
        print("Average Loss = %5.5f after epoch %4d " % (Loss[ii + 1], ii ))
    return (W,B,Loss)

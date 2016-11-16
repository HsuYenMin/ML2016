'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

#from __future__ import print_function
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import numpy as np

batch_size = 100 
nb_classes = 10
nb_epoch = 80
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# path to the model file with weights.
model_path = 'semisupervised.h5'

# the data, shuffled and split between train and test sets
label = pickle.load(open('./data/all_label.p','rb'))
label = np.array(label)
X_train = label.reshape(5000,3,32,32)
Y_train = []
for i in range(10):
	for j in range(500):
		Y_train.append(i)
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, 10)
X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
X_train /= 255
# X_test /= 255
model_path = 'supervised_model.h5'
model = load_model(model_path)

print('Start to load unlabeled data...')
unlabel = pickle.load(open('./data/all_unlabel.p','rb'))
unlabel = np.reshape(unlabel,(45000,3,32,32))
unlabel = unlabel.astype('float32')
unlabel /= 255
print('unlabeled data loaded')
for i in range(3):
    prediction = model.predict_proba(unlabel, batch_size=1000)
    check = np.zeros(len(unlabel),np.dtype(bool))
    for j in range(len(prediction)):
        if np.max(prediction[j]) > 0.6:
            check[j] = True
    Xlen = len(X_train)
    Xshape = (Xlen + np.sum(check),3,32,32)
    Yshape = (Xlen + np.sum(check),10)
    unlabelshape = (len(check) - np.sum(check),3,32,32)
    newX_train = np.zeros(Xshape, np.dtype('float32'))
    newY_train = np.zeros(Yshape)
    newUnlabel = np.zeros(unlabelshape, np.dtype('float32'))
    for j in range(Xlen):
        newX_train[j] = X_train[j]
        newY_train[j] = Y_train[j]
    count = Xlen
    countUnlabel = 0
    for j in range(len(check)):
        if check[j] == True:
            newX_train[count] = unlabel[j]
            newY_train[count][np.argmax(prediction[j])] = 1
            count += 1
        else:
            newUnlabel[countUnlabel] = unlabel[j]
            countUnlabel += 1
    X_train = newX_train
    Y_train = newY_train
    unlabel = newUnlabel
    print('add', np.sum(check),'data to training set.')
    model.fit(X_train,Y_train,batch_size= 100, nb_epoch = 3, validation_split=0.01)
'''
for OAQ in range(3):
    prediction = model.predict_proba(unlabel, batch_size=100)
    i = 0
    count = 0
    while i < len(prediction):
        if np.max(i) > 0.6:
            picClass = np.argmax(i)
            y = np.zeros(10)
            y[picClass] = 1
            X_train = np.append(X_train,unlabel[i])
            Y_train = np.append(Y_train,y)
            unlabel = np.delete(unlabel,i,0)
            prediction = np.delete(prediction,i,0)
            count += 1
        else:
            i += 1
    print("add ", count , " data to training data")
    model.fit(X_train,Y_train,batch_size= 100, nb_epoch = 3, validation_split=0.01)
'''
# save model
model_path = 'semisupervised.h5'
model.save(model_path)

from keras.preprocessing.image import ImageDataGenerator
import pickle
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import json 
import csv
import numpy as np
model_path = 'm02.h5'
model = load_model(model_path)
test = pickle.load(open('./data/test.p','rb'))
test = np.reshape(test['data'],(10000,3,32,32))
test = test.astype('float32')
# X_test = X_test.astype('float32')
test /= 255
# X_test /= 255
prediction =  model.predict_classes(test, batch_size=100, verbose=1)

with open( 'p02_mac.csv', 'w') as csvfile:
    fieldnames = ['ID','class']
    w = csv.DictWriter(csvfile,fieldnames=fieldnames)
    w.writeheader()
    for i in range(len(prediction)):
        w.writerow({'ID':str(i), 'class': prediction[i]})



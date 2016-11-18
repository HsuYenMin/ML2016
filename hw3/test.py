from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import pickle
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import sys
import json 
import csv
import numpy as np
model_path = 'Encoder.h5'
encoder = load_model(model_path)
model_path = sys.argv[2]
model = load_model(model_path)

test = pickle.load(open(sys.argv[1] + 'test.p','rb'))
test = np.reshape(test['data'],(10000,3,32,32))
test = test.astype('float32')
# X_test = X_test.astype('float32')
test /= 255
# X_test /= 255
encoded_img = encoder.predict(test, batch_size=1000, verbose=1)
encoded_img = encoded_img.reshape(10000,128)
prediction = model.predict_classes(encoded_img,batch_size=1000, verbose=1)
with open( sys.argv[3], 'w') as csvfile:
    fieldnames = ['ID','class']
    w = csv.DictWriter(csvfile,fieldnames=fieldnames)
    w.writeheader()
    for i in range(len(prediction)):
        w.writerow({'ID':str(i), 'class': prediction[i]})



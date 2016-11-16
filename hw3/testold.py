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

# model = load_model('cnn_model.json')
with open('cnn_model.json', 'r') as f:
    json_string = f.read()
model = model_from_json(json_string)
model.load_weights('cnn_weights.h5')
test = pickle.load(open('./data/test.p','rb'))
test = np.reshape(test['data'],(10000,3,32,32))
prediction =  model.predict_classes(test, batch_size=32, verbose=1)

with open( 'prediction.csv', 'w') as csvfile:
    fieldnames = ['ID','class']
    w = csv.DictWriter(csvfile,fieldnames=fieldnames)
    w.writeheader()
    for i in range(len(prediction)):
        w.writerow({'ID':str(i), 'class': prediction[i]})



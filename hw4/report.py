from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
import logging
from optparse import OptionParser
import sys
from time import time, sleep
import numpy as np
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.models import Model
import pickle
import csv
data = [None]* 20
count = [0] * 20
numberOfData = [0] * 20
with open(sys.argv[1] + 'label_StackOverflow.txt','r') as label:    
    for row in label:
        numberOfData[int(row)-1] += 1
for idx, row in enumerate(numberOfData):
    data[idx] = [None] * row
Label = open(sys.argv[1] + 'label_StackOverflow.txt','r').readlines()
title = open(sys.argv[1] + 'title_Stackoverflow.txt','r').readlines()

for idx, tag in enumerate(Label):
    data[int(tag)-1][count[int(tag)-1]] = title[idx]
    count[int(tag)-1] += 1

vectorizer = TfidfVectorizer(max_df=0.5, max_features=10,
                             min_df=2, stop_words='english',
                             use_idf=True)
'''
X = vectorizer.fit_transform(data)
svd = TruncatedSVD(n_components=22)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = X.toarray()
X = lsa.fit_transform(X)
'''
'''
input_img = Input(shape=(X.shape[1],))
encoded = Dense(22, activation='relu')(input_img)
encoded = Dense(18, activation='relu')(encoded)
encoded = Dense(14, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)

decoded = Dense(14, activation='relu')(encoded)
decoded = Dense(18, activation='relu')(encoded)
decoded = Dense(22, activation='relu')(decoded)
decoded = Dense(X.shape[1], activation='sigmoid')(decoded)

encoder = Model(input_img, output=encoded)
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')
autoencoder.fit(X, X,
                nb_epoch=120,
                batch_size=256,
                shuffle=True,
                validation_split=0.01)
encoded_txt = encoder.predict(X)
'''
'''
encoded_txt = X
km = KMeans(n_clusters=33, init='k-means++', max_iter=100, n_init=20)
km.fit(encoded_txt)
prediction = km.predict(encoded_txt)

with open(sys.argv[2],'w') as output:
    fn = ['ID','Ans']
    w = csv.DictWriter(output,fn)
    w.writeheader()
    with open(sys.argv[1] + 'check_index.csv', 'r') as check:
        r = csv.DictReader(check)
        for idx,row in enumerate(r):
            if prediction[int(row['x_ID'])] == prediction[int(row['y_ID'])] :
                w.writerow({'ID': str(idx), 'Ans': '1'})
            else:
                w.writerow({'ID': str(idx), 'Ans': '0'})
'''

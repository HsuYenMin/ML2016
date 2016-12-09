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
data = [None]* 20000
with open(sys.argv[1] + 'title_Stackoverflow.txt','r') as f:
    for idx, row in enumerate(f):
        data[idx] = row
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                             min_df=2, stop_words='english',
                             use_idf=True)
X = vectorizer.fit_transform(data)
svd = TruncatedSVD(n_components=22)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = X.toarray()
# X = lsa.fit_transform(X)
input_img = Input(shape=(X.shape[1],))
encoded = Dense(2000, activation='relu')(input_img)
encoded = Dense(1000, activation='relu')(encoded)
encoded = Dense(500, activation='relu')(encoded)
encoded = Dense(250, activation='relu')(encoded)
encoded = Dense(100, activation='relu')(encoded)
encoded = Dense(50, activation='relu')(encoded)
encoded = Dense(25, activation='relu')(encoded)

decoded = Dense(50, activation='relu')(encoded)
decoded = Dense(100, activation='relu')(decoded)
decoded = Dense(250, activation='relu')(decoded)
decoded = Dense(500, activation='relu')(decoded)
decoded = Dense(1000, activation='relu')(decoded)
decoded = Dense(2000, activation='relu')(decoded)
decoded = Dense(X.shape[1], activation='sigmoid')(decoded)

encoder = Model(input_img, output=encoded)
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')
autoencoder.fit(X, X,
                nb_epoch=1,
                batch_size=256,
                shuffle=True,
                validation_split=0.01)
encoded_txt = encoder.predict(X)
X = encoded_txt
km = KMeans(n_clusters=33, init='k-means++', max_iter=100, n_init=20)
km.fit(encoded_txt)
prediction = km.predict(encoded_txt)
'''
with open(sys.argv[1] + 'label_StackOverflow.txt','r') as label:    
    for i,row in enumerate(label):
        prediction[i] = int(row) - 1 
'''

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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
# code.interact(local=locals())
# plt.figure(figsize = (10,10))
# fig = plt.scatter(X[:,1], X[:,2], c=colors).get_figure()
sample_num=10000
from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0)
t0 = time()
vis_data = model.fit_transform(X[:sample_num])
print("TSNE done in %fs" % (time() - t0)) 
vis_x = vis_data[:,0]
vis_y = vis_data[:,1]
fig=plt.scatter(vis_x, vis_y, c=prediction[:sample_num], cmap=plt.cm.get_cmap('jet',33))

plt.colorbar(ticks=range(33))
plt.clim(-0.5,33.5)
plt.savefig('label.png')
plt.show()

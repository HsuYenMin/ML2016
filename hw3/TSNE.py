from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
from sklearn.manifold import TSNE
'''
label = pickle.load(open('./data/all_label.p','rb'))
label = np.array(label)
X_train = label.reshape(5000,3,32,32)
'''
Y_train = np.zeros((5000), np.dtype(int))
for i in range(10):
	for j in range(500):
		Y_train[i * 500 + j] = i
'''
X_train = X_train.astype('float64')
X_train /= 255
encoder = load_model('ENCODER_50EP.h5')
'''
with open('OAQ.json','r') as f:
    en = f.read()
enco = json.loads(en)
encoded_img = np.array(enco)
model = TSNE(n_components=2, random_state=0)
vis_data = model.fit_transform(encoded_img) 
vis_x = vis_data[:,0]
vis_y = vis_data[:,1]
plt.scatter(vis_x, vis_y, c=Y_train, cmap=plt.cm.get_cmap('jet',10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5,9.5)
plt.savefig('tsne_50ep', bbox_inches='tight')


from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD, Adam
from keras.models import Model
import pickle
import numpy as np

label = pickle.load(open('./data/all_label.p','rb'))
label = np.array(label)
label = label.reshape(5000,3,32,32)
unlabel = pickle.load(open('./data/all_unlabel.p','rb'))
unlabel = np.array(unlabel)
unlabel = np.reshape(unlabel,(45000,3,32,32))
X_train = np.concatenate((label,unlabel))
X_train = X_train.reshape(50000,3,32,32)
X_train = X_train.astype('float32')
X_train /= 255

input_img = Input(shape=(3, 32, 32))

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, output=encoded)
autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, nb_epoch=10, batch_size=128, shuffle=True, validation_split= 0.01)
model_path ='semi2.h5'
autoencoder.save(model_path)
model_path ='encoder.h5'
encoder.save(model_path)

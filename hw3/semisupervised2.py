from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Model, load_model, Sequential 
from keras.utils import np_utils
import sys
import pickle
import numpy as np
nb_classes=10
label = pickle.load(open(sys.argv[1] + 'all_label.p','rb'))
label = np.array(label)
X_train = label.reshape(5000,3,32,32)
Y_train = np.zeros((5000))
for i in range(10):
	for j in range(500):
		Y_train[i * 500 + j] = i
Y_train = np_utils.to_categorical(Y_train, 10)
X_train = X_train.astype('float64')
X_train /= 255
encoder = load_model('Encoder.h5')
encoded_img = encoder.predict(X_train)
encoded_img = encoded_img.reshape(5000,128)


model = Sequential()
model.add(Dense(128, input_shape=(128,)))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(encoded_img, Y_train,
          batch_size=100,
          nb_epoch=200,
          shuffle=True,
          validation_split=0.01)
model_path = sys.argv[2]
model.save(model_path)

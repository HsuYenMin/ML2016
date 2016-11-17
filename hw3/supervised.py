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
from keras.models import Sequential
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



# the data, shuffled and split between train and test sets
label = pickle.load(open('./data/all_label.p','rb'))
label = np.array(label)
X_train = label.reshape(5000,3,32,32)
Y_train = []
for i in range(10):
	for j in range(500):
		Y_train.append(i)


'''
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
'''
Y_train = np_utils.to_categorical(Y_train, 10)

'''
Y_test = np_utils.to_categorical(y_test, nb_classes)
'''

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(256))
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
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
X_train /= 255
# X_test /= 255
# model.fit(X_train,Y_train,batch_size= 100, nb_epoch = 20,validation_split=0.01)
# data split
nVali = 100
Xlen = len(X_train)
Xshape = (Xlen - nVali, 3, 32, 32)
Yshape = (Xlen - nVali, 10)
xs = np.random.permutation(np.arange(5000))[:nVali]
newX_train = np.zeros(Xshape, np.dtype('float32'))
newY_train = np.zeros(Yshape)
valiXShape = (nVali, 3, 32, 32)
valiYShape = (nVali, 10)
ValiX = np.zeros(valiXShape, np.dtype('float32'))
ValiY = np.zeros(valiYShape)
check = np.zeros((5000), np.dtype(bool))
for x in xs:
    check[x] = 1
count = 0
countValData = 0
for j in range(len(check)):
    if check[j] == False:
        newX_train[count] = X_train[j]
        newY_train[count] = Y_train[j]
        count += 1
    else:
        ValiX[countValData] = X_train[j]
        ValiY[countValData] = Y_train[j]
        countValData += 1

X_train = newX_train
Y_train = newY_train
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    history = model.fit_generator(datagen.flow(X_train, Y_train,
                                  batch_size=batch_size),
                                  samples_per_epoch=X_train.shape[0],
                                  nb_epoch=nb_epoch,
                                  validation_data=(ValiX, ValiY))

# save model
# path to the model file with weights.
#model_path = 'supervised_model.h5'
#model.save(model_path)

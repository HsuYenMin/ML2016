from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import math
import sys
from tqdm import tqdm
from keras.models import Sequential, load_model, Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Dropout, Flatten, MaxoutDense, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical


[ data_label, data_unlabel, test, class_label ] = pickle.load( open( "data.p", "rb" ) )

'''
inputM = Input( shape = ( 3, 32, 32 ) )
M = Convolution2D( 16, 5, 5, border_mode = 'same', input_shape = ( 3, 32, 32 ) )( inputM )
M = ELU()( M )
M = MaxPooling2D(( 2, 2 ))( M )
M = Dropout( 0.1 )( M )
M = Convolution2D( 128, 5, 5, border_mode = 'same' )( M )
M = ELU()( M )
M = MaxPooling2D(( 2, 2 ))( M )
M = Dropout( 0.1 )( M )
M = Flatten()( M )
M = Dense( output_dim = 640, activation = 'sigmoid' )( M )
M = Dense( output_dim = 160, activation = 'sigmoid' )( M )
M = Dense( output_dim = 40, activation = 'sigmoid' )( M )
M = MaxoutDense( output_dim = 10 )( M )
M = Activation( 'softmax' )( M )
model = Model( inputM, M )
model.compile( loss = 'categorical_crossentropy', optimizer = 'Adadelta', metrics = [ 'categorical_accuracy' ] )
model.fit( data_label, class_label, batch_size = 50, nb_epoch = 300, validation_split = 0 )
print( model.evaluate( data_label, class_label ) )
model.save( "data/model" )


model = load_model( "data/model" )

for i in range( 2 ):
    result = model.predict( data_unlabel )

    confident = np.zeros( result.shape[ 0 ], dtype = bool )
    for j in range( result.shape[ 0 ] ):
        if np.max( result[ j ] ) > 0.8:
            confident[ j ] = True
    confident_num = np.sum( confident )
    print( confident_num )
    data_label = np.append( data_label, np.empty(( confident_num, 3, 32, 32 )), axis = 0 )
    class_label = np.append( class_label, np.zeros(( confident_num, 10 )), axis = 0 )
    temp = np.empty(( confident.shape[ 0 ] - confident_num, 3, 32, 32 ))
    k = 0
    l = 0
    for j in range( confident.shape[ 0 ] ):
        if confident[ j ] == True:
            data_label[ data_label.shape[ 0 ] - confident_num + k ] = data_unlabel[ j ]
            class_label[ class_label.shape[ 0 ] - confident_num + k ][ np.argmax( result[ j ] ) ] = 1
            k += 1
        else:
            temp[ l ] = data_unlabel[ j ]
            l += 1
    data_unlabel = temp.copy()

    model.fit( data_label, class_label, batch_size = 100, nb_epoch = 50, validation_split = 0.01 )
    print( model.evaluate( data_label, class_label ) )

model.save( "data/model1" )
'''


def make_trainable( val ):
    discriminator.trainable = val
    for l in discriminator.layers:
        l.trainable = val
    if val == True:
        discriminator.compile( loss = 'binary_crossentropy', optimizer = opt, metrics = [ 'binary_accuracy' ] )
    else:
        GAN.compile( loss = 'binary_crossentropy', optimizer = opt, metrics = [ 'binary_accuracy' ] )

opt = 'Adadelta'

#Build Generative Model
inputG = Input( shape = ( 100, ) )
G = Dense( 128 * 4 * 4, init = 'he_normal' )( inputG )
G = Activation( 'relu' )( G )
G = Reshape( [ 128, 4, 4 ] )( G )
G = UpSampling2D( size = ( 2, 2 ) )( G )
G = Convolution2D( 64 , 2, 2, border_mode = 'same', init = 'he_normal' )( G )
G = Activation( 'relu' )( G )
G = UpSampling2D( size = ( 2, 2 ) )( G )
G = Convolution2D( 32, 5, 5, border_mode = 'same', init = 'he_normal' )( G )
G = Activation( 'relu' )( G )
G = UpSampling2D( size = ( 2, 2 ) )( G )
G = Convolution2D( 3, 5, 5, border_mode = 'same', init = 'he_normal' )( G )
G = Activation( 'tanh' )( G )
generator = Model( inputG, G )


# Build Discriminative model ...
inputD = Input( shape = ( 3, 32, 32 ) )
D = Convolution2D( 16, 3, 3, border_mode = 'same', activation = 'relu' )( inputD )
D = LeakyReLU( 0.2 )( D )
D = MaxPooling2D(( 2, 2 ))( D )
D = Dropout( 0.25 )( D )
D = Convolution2D( 64, 3, 3, border_mode = 'same', activation = 'relu' )( D )
D = LeakyReLU( 0.2 )( D )
D = MaxPooling2D(( 2, 2 ))( D )
D = Dropout( 0.25 )( D )
D = Flatten()( D )
D = Dense( 128 )( D )
D = LeakyReLU( 0.2 )( D )
D = Dropout( 0.25 )( D )
D = Dense( 16 )( D )
D = LeakyReLU( 0.2 )( D )
D = Dropout( 0.25 )( D )
D = Dense( 1, activation = 'sigmoid' )( D )
discriminator = Model( inputD, D )
discriminator.compile( loss = 'binary_crossentropy', optimizer = opt, metrics = [ 'binary_accuracy' ] )
    

# Build stacked GAN model
inputGAN = Input( shape = ( 100, ) )
GAN = generator( inputGAN )
GAN = discriminator( GAN )
GAN = Model( inputGAN, GAN )
GAN.compile( loss = 'binary_crossentropy', optimizer = opt, metrics = [ 'binary_accuracy' ] )

# Pretraining disc
noise = np.random.uniform( 0, 1, size = ( data_label.shape[ 0 ], 100 ) )
data_generated = generator.predict( noise )

data_train = np.concatenate(( data_label, data_generated ))
fake_train = np.zeros(( data_train.shape[ 0 ] ))
fake_train[ :data_label.shape[ 0 ] ] = 1

d_loss = discriminator.fit( data_train, fake_train, nb_epoch = 2, batch_size = 100 )


data_shuffle = data_label.copy()
np.random.shuffle( data_shuffle )

# training loop
def trainGAN( nb_epoch, batch_size ):
    for i in range( nb_epoch ):
        t = tqdm( range( data_shuffle.shape[ 0 ] // batch_size ), ncols = 100 )
        for j in t:
            # Generate data
            data_batch = data_shuffle[ batch_size * j : batch_size * ( j + 1 ) ]
            noise = np.random.uniform( 0, 1, size = ( batch_size, 100 ) )
            data_generated = generator.predict( noise )

            #Train Disciminator
            data_train = np.concatenate(( data_batch, data_generated ))
            fake_train = np.zeros(( 2 * batch_size ))
            fake_train[ :batch_size ] = 1

            make_trainable( True )
            d_loss = discriminator.train_on_batch( data_train, fake_train )

            #Train GAN
            noise = np.random.uniform( 0, 1, size = ( 2 * batch_size, 100 ) )
            fake_train = np.ones(( 2 * batch_size ))

            make_trainable( False )
            g_loss = GAN.train_on_batch( noise, fake_train )

            t.set_description( "d_accu: %.5f" % d_loss[ 1 ] + " g_accu: %.5f" % g_loss[ 1 ] )

trainGAN( 5, 500 )


def plot_gen( num = 1 ):
    noise = np.random.uniform( 0, 1, size = [ num, 100 ] )
    generated_images = generator.predict( noise )

    for i in generated_images:
        i = np.reshape( i, ( 3, 1024 ) )
        i = i.transpose()
        i = np.reshape( i, ( 32, 32, 3) )
        plt.imshow( i )
        plt.show()

plot_gen( 5 )


answer = Model.predict_classes( test )


f = open( '1','w' )
f.write( 'ID,class\n' )
for i in range( len( answer ) ):
    f.write( str( i ) + ',' + str( answer[ i ] ) + '\n' )
f.close()

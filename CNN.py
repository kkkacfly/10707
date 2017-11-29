'''
Date: 11/27/2017
Objective: 10707 project, CNN part
'''
from __future__ import print_function
import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from BilinearUpSampling import *


# need to resize
input_shape = (224, 224, 3)


# build model
def get_model(input_shape):
    inputs = Input(input_shape)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='b1_1')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='b1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='MaxPool_1')(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='b2_1')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='b2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='MaxPool_2')(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='b3_1')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='b3_2')(conv3)

    atru1 = AtrousConv2D(256, (3,3), padding='same', name='atru1')(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='b4_1')(atru1)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='b4_2')(conv4)

    atru2 = AtrousConv2D(512, (3,3), padding='same', name='atru2')(conv4)
    atru3 = AtrousConv2D(512, (3,3), padding='same', name='atru3')(atru2)
    atru4 = AtrousConv2D(512, (3,3), padding='same', name='atru4')(atru3)

    # Hyper-Atrous combination
    hyper_atrous = concatenate([pool2, atru1, atru2, atru3, atru4], axis=3, name='Hyper-Atrous')

    conv5 = Conv2D(512, (1, 1), activation='relu', padding='same', name='b5_1')(hyper_atrous)

    # deconv
    deconv1 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', name='deconv1')(conv5)
    deconv2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name='deconv2')(deconv1)

    conv6 = Conv2D(1, (1, 1), activation='relu', padding='same', name='b6_1')(deconv2)

    model = Model(inputs=[inputs], outputs=[conv6])

    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy')

    return model





# load data
p = ''
X_train = np.load(p+'X_train.npy')
Y_train = np.load(p+'train_density.npy')

X_test = np.load(p+'X_test.npy')
Y_test = np.load(p+'test_density.npy')

# add dimension
Y_train = np.expand_dims(Y_train, axis=3)
Y_test = np.expand_dims(Y_test, axis=3)


print(X_train.shape)
print(Y_train.shape)


model = get_model(input_shape)
sgd = SGD(lr=0.01, momentum=0.5, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae'])
print(model.summary())
hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=16, verbose=2)




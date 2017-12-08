from __future__ import print_function
import numpy as np
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Conv2D, MaxPooling2D, AtrousConv2D, concatenate, Conv2DTranspose, Lambda, Reshape, Dense, Add, Flatten, Merge
import keras.backend as K
from keras import regularizers
from util import *


# build model
def get_model(input_shape, weight_decay):
    inputs = Input(input_shape, name='inputs')
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay), name='b1_1')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay), name='b1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='MaxPool_1')(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay), name='b2_1')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay), name='b2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='MaxPool_2')(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay), name='b3_1')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay), name='b3_2')(conv3)

    atru1 = AtrousConv2D(256, (3,3), activation='relu', padding='same', name='atru1')(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='b4_1')(atru1)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='b4_2')(conv4)

    atru2 = AtrousConv2D(512, (3,3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay), name='atru2')(conv4)
    atru3 = AtrousConv2D(512, (3,3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay), name='atru3')(atru2)
    atru4 = AtrousConv2D(512, (3,3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay), name='atru4')(atru3)

    # Hyper-Atrous combination
    hyper_atrous = concatenate([pool2, atru1, atru2, atru3, atru4], axis=3, name='Hyper-Atrous')

    conv5 = Conv2D(512, (1, 1), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(weight_decay), name='b5_1')(hyper_atrous)

    # deconv
    deconv1 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', name='deconv1')(conv5)
    deconv2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name='deconv2')(deconv1)

    # density output
    dnsty_output = Conv2D(1, (1, 1), activation='linear', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(0.001), padding='same', name='dnsty_output')(deconv2)

    #count_output
    count_flatten = Flatten()(dnsty_output)
    count_sum = Lambda(sum_flatten_layer)(count_flatten)
    count_flatten = Dense(100, activation='linear', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(0.001))(count_flatten)
    count_fc = Dense(1,  activation='linear', kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(0.001), name='count_prediction')(count_flatten)
    #count_output = Add(name='count_output')([count_sum, count_fc])
    count_output = Merge(name='count_output', mode='sum')([count_sum, count_fc])

    model = Model(inputs=[inputs], outputs=[dnsty_output, count_output])

    return model


# need to resize
input_shape = (224, 224, 3)
weight_decay = 1e-5

# load data
p = 'data/'
X_train = np.load(p+'X_train.npy')
Y_train_density = np.load(p+'train_density.npy')
Y_train = np.load(p+'Y_train.npy').astype(np.float32)


# cropped training data
#X_train_bright = np.load(p+'X_train_brightness.npy')
#Y_train_density_crop = np.load(p+'train_density_crop.npy')


# vstack data
#X_train = np.vstack((X_train, X_train_bright))
#Y_train_density = np.vstack((Y_train_density, Y_train_density))

# load testing data
X_test = np.load(p+'X_test.npy')
Y_test_density = np.load(p+'test_density.npy')
Y_test = np.load(p+'Y_test.npy').astype(np.float32)


# add dimension
Y_train_density = np.expand_dims(Y_train_density, axis=3)
Y_test_density = np.expand_dims(Y_test_density, axis=3)
Y_train = np.expand_dims(Y_train, axis=1)
Y_test = np.expand_dims(Y_test, axis=1)

print('dimension added')

# data augmentation: flip
X_train = h_flip(X_train)
Y_train_density = h_flip(Y_train_density)
Y_train = np.vstack((Y_train, Y_train))

print('data augmentation complete')


adam = Adam(lr=0.0001)

model = get_model(input_shape, weight_decay)
model.compile(loss={'dnsty_output': mse_loss, 'count_output': mse_loss_count},
              loss_weights={'dnsty_output': 1., 'count_output': 0.001},
              optimizer=adam, metrics=['mae'])

#model = load_model('CNN_model_br.h5', custom_objects={'mse_loss': mse_loss, 'mse_loss_count': mse_loss_count})

print(model.summary())

hist = model.fit({'inputs': X_train}, {'dnsty_output': Y_train_density, 'count_output': Y_train},
                 validation_data=({'inputs': X_test}, {'dnsty_output': Y_test_density, 'count_output': Y_test}),
                 epochs=100, batch_size=16, verbose=2)


# save loss
np.save('train_loss_br_ft_1.npy', hist.history)

# save model
model.save('CNN_model_br.h5')

pred = model.predict(X_test)
np.save('./result/pred_br_density.npy', pred[0])

pred = model.predict(X_train)
np.save('./result/pred_br_train_density.npy', pred[0])

# save weights
model.save_weights('br_weight.h5')


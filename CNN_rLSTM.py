'''
Date: 11/30/2017
Objective: build & train model
'''
from __future__ import print_function
import numpy as np
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
#from BilinearUpSampling import *
from keras.layers import Input, Conv2D, MaxPooling2D, AtrousConv2D, concatenate, Conv2DTranspose, Lambda, Reshape, Dense, Add, Flatten, TimeDistributed, LSTM, Merge
import keras.backend as K
from keras import regularizers
import h5py


# sum over tensor (with reshape)
def sum_tensor_block(tensor):
    tensor = Reshape((224,224))(tensor)
    tensor = K.sum(tensor, axis=1)
    return K.sum(tensor, axis=1, keepdims=True)


# sum over flattened layer
def sum_flatten_layer(tensor):
    tensor = K.sum(tensor, axis=1, keepdims=True)
    return tensor


# mse loss for density
def mse_loss(y_true, y_pred):
    diff = K.square(y_true-y_pred)
    t_sum = K.sum(diff, axis=1)
    t_sum = K.sum(t_sum, axis=1)
    return K.mean(t_sum)


# mse loss for count
def mse_loss_count(y_true, y_pred):
    diff = K.square(y_true - y_pred)
    return K.mean(diff)

# data augmentation
def data_augment():
    pass


# build model
'''
Input:
    - input_shape: (None, time_step, height, width, channel)
    - weight_decay: L2 regularizer for Conv layer (feature extraction)
    - weight_decay_reg: L2 regularizer for prediction layer
Output:
    - keras model
'''
def build_model(input_shape, weight_decay, weight_decay_reg):
    inputs = Input(input_shape, name='inputs')

    # conv block 1
    conv1 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same',
                                   kernel_regularizer=regularizers.l2(weight_decay)), name='b1_1')(inputs)
    conv1 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same',
                                   kernel_regularizer=regularizers.l2(weight_decay)), name='b1_2')(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), name='MaxPool_1'))(conv1)

    # conv block 2
    conv2 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same',
                                   kernel_regularizer=regularizers.l2(weight_decay)), name='b2_1')(pool1)
    conv2 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same',
                                   kernel_regularizer=regularizers.l2(weight_decay)), name='b2_2')(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), name='MaxPool_2'))(conv2)

    # conv block 3
    conv3 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same',
                                   kernel_regularizer=regularizers.l2(weight_decay)), name='b3_1')(pool2)
    conv3 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same',
                                   kernel_regularizer=regularizers.l2(weight_decay)), name='b3_2')(conv3)

    atru1 = TimeDistributed(AtrousConv2D(256, (3, 3), activation='relu', padding='same',
                                         kernel_regularizer=regularizers.l2(weight_decay)), name='atru1')(conv3)

    # conv block 4
    conv4 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same',
                                   kernel_regularizer=regularizers.l2(weight_decay)), name='b4_1')(atru1)
    conv4 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same',
                                   kernel_regularizer=regularizers.l2(weight_decay)), name='b4_2')(conv4)

    # atrous block
    atru2 = TimeDistributed(AtrousConv2D(512, (3, 3), activation='relu', padding='same',
                                         kernel_regularizer=regularizers.l2(weight_decay)), name='atru2')(conv4)
    atru3 = TimeDistributed(AtrousConv2D(512, (3, 3), activation='relu', padding='same',
                                         kernel_regularizer=regularizers.l2(weight_decay)), name='atru3')(atru2)
    atru4 = TimeDistributed(AtrousConv2D(512, (3, 3), activation='relu', padding='same',
                                         kernel_regularizer=regularizers.l2(weight_decay)), name='atru4')(atru3)

    # Hyper-Atrous combination
    #hyper_atrous = TimeDistributed((concatenate([pool2, atru1, atru2, atru3, atru4], axis=4, name='Hyper-Atrous')))
    hyper_atrous = (concatenate([pool2, atru1, atru2, atru3, atru4], axis=4, name='Hyper-Atrous'))

    conv5 = TimeDistributed(Conv2D(512, (1, 1), activation='relu', padding='same',
                                   kernel_regularizer=regularizers.l2(weight_decay)), name='b5_1')(hyper_atrous)

    # deconv
    deconv1 = TimeDistributed(Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same'), name='deconv1')(conv5)
    deconv2 = TimeDistributed(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'), name='deconv2')(deconv1)

    # density output
    dnsty_output = TimeDistributed(Conv2D(1, (1, 1), activation='linear', kernel_initializer='he_normal',
                                   kernel_regularizer=regularizers.l2(weight_decay_reg), padding='same'), name='dnsty_output')(deconv2)

    # count_output
    count_flatten = TimeDistributed(Flatten(), name='count_flatten')(dnsty_output)
    count_sum = TimeDistributed(Lambda(sum_flatten_layer), name='count_sum')(count_flatten)
    lstm1 = LSTM(100, return_sequences=True, name='lstm_1')(count_flatten)
    lstm2 = LSTM(100, return_sequences=True, name='lstm_2')(lstm1)
    lstm3 = LSTM(100, return_sequences=True, name='lstm_3')(lstm2)
    count_flatten = TimeDistributed(Dense(100, activation='relu', kernel_initializer='he_normal',
                                          kernel_regularizer=regularizers.l2(weight_decay_reg)), name='fully_connected')(lstm3)
    count_fc = TimeDistributed(Dense(1, activation='linear', kernel_initializer='he_normal',
                                     kernel_regularizer=regularizers.l2(weight_decay_reg)), name='count_prediction')(count_flatten)
    count_output = Merge(name='count_output', mode='sum')([count_sum, count_fc])

    model = Model(inputs=[inputs], outputs=[dnsty_output, count_output])

    return model


'''
Objective: compile the built model

Input: 
    - model: return from function 'build_model'
    - mse_loss: self defined loss for density
    - mse_loss_count: self defined loss for counts
    - metrics: metrics used for evaluation
    - optimizer: adam
    - lr_rate: initial learning rate
    - weights: balance between density loss and count loss
    
Output:
    - compiled model
'''
def model_compile(model, mse_loss, mse_loss_count, metrics, optimizer, lr_rate, weights):
    if optimizer == 'adam':
        adam = Adam(lr=lr_rate)
    model.compile(loss={'dnsty_output': mse_loss, 'count_output': mse_loss_count},
                  loss_weights={'dnsty_output': weights[0], 'count_output': weights[1]},
                  optimizer=adam, metrics=[metrics])
    print('model has been compiled')
    return model


'''
Objective: single input (raw image), multiple output(density, count) training 

Input: 
    - model: compiled model
    
Output:
    - 
     
'''
# train model
def model_fit(X_train, Y_train_density, Y_train_count, model, time_step, batch_size, epochs):
    train_n = X_train.shape[0]
    print(train_n)
    train_seq_rang = range(0, train_n-time_step)
    for epoch in range(epochs):
        print(epoch)
        for i in train_seq_rang:
            batch_size_curr = min(batch_size, train_n-i-time_step+1)
            # initial current training/testing set
            X_train_curr = np.zeros((batch_size_curr, time_step, 224, 224, 3))
            Y_train_density_curr = np.zeros((batch_size_curr, time_step, 224, 224, 1))
            Y_train_count_curr = np.zeros((batch_size_curr, time_step, 1))
            for j in range(batch_size_curr):
                X_train_curr[j] = X_train[i+j:i+j+time_step]
                Y_train_density_curr[j] = Y_train_density[i+j:i+j+time_step]
                Y_train_count_curr[j] = Y_train_count[i+j:i+j+time_step]
            # fit model
            hist = model.fit({'inputs': X_train_curr},
                             {'dnsty_output': Y_train_density_curr, 'count_output': Y_train_count_curr},
                             epochs=1, batch_size=batch_size_curr, verbose=2)

    return model, hist


# convert input with time step
def get_sequential_data(X_train, Y_train_density, Y_train_count, time_step):
    train_n = X_train.shape[0]
    new_n = train_n - time_step
    train_seq_rang = range(0, train_n - time_step)
    # initial sequence
    X_train_s = np.zeros((new_n, time_step, 224, 224, 3))
    Y_train_density_s = np.zeros((new_n, time_step, 224, 224, 1))
    Y_train_count_s = np.zeros((new_n, time_step, 1))
    for i in train_seq_rang:
        print(i)
        X_train_s[i] = X_train[i:i+time_step]
        Y_train_density_s[i] = Y_train_density[i:i+time_step]
        Y_train_count_s[i] = Y_train_count[i:i+time_step]
    return X_train_s, Y_train_density_s, Y_train_count_s



# main

# need to resize
time_step = 5
input_shape = (5, 224, 224, 3)
weight_decay = 1e-6
weight_decay_reg = 1e-3
batch_size = 4
epochs = 100
lr = 1e-4
weights = (1.0, 0.001)


# load data
#p = '/media/inferlabnas/homes/wflores/Data/10707/data/'
p = 'data/'
X_train = np.load(p+'X_train.npy')
Y_train_density = np.load(p+'train_density.npy')
Y_train_count = np.load(p+'Y_train.npy').astype(np.float32)

X_test = np.load(p+'X_test.npy')
Y_test_density = np.load(p+'test_density.npy')
Y_test_count = np.load(p+'Y_test.npy').astype(np.float32)


# add dimension
Y_train_density = np.expand_dims(Y_train_density, axis=-1)
Y_test_density = np.expand_dims(Y_test_density, axis=-1)
Y_train_count = np.expand_dims(Y_train_count, axis=-1)
Y_test_count = np.expand_dims(Y_test_count, axis=-1)



# get sequential data
X_train, Y_train_density, Y_train_count = get_sequential_data(X_train, Y_train_density, Y_train_count, time_step)

X_test, Y_test_density, Y_test_count = get_sequential_data(X_test, Y_test_density, Y_test_count, time_step)

model = build_model(input_shape, weight_decay, weight_decay_reg)

#model = load_model('FCN_model.h5', custom_objects={'mse_loss': mse_loss, 'mse_loss_count': mse_loss_count})

print(model.summary())

model_compile(model, mse_loss, mse_loss_count, 'mae', 'adam', lr, weights)

hist = model.fit({'inputs': X_train}, {'dnsty_output': Y_train_density, 'count_output': Y_train_count},
                 validation_data=({'inputs': X_test}, {'dnsty_output': Y_test_density, 'count_output': Y_test_count}),
                 epochs=50, batch_size=4, verbose=2)


# save loss
np.save('train_loss_fcn_rlstm_ft_1.npy', hist.history)

# save model
model.save('FCN_model.h5')

pred = model.predict(X_test)
np.save('./result/pred_fcn_rlstm_density_ft_1.npy', pred[0])

# save weights
model.save_weights('fcn_weight.h5')

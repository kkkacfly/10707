from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.optimizers import SGD, Adam
#from BilinearUpSampling import *
from keras.layers import Input, Conv2D, MaxPooling2D, AtrousConv2D, concatenate, Conv2DTranspose, Lambda, Reshape, Dense, Add, Flatten
import keras.backend as K
from keras import regularizers
import h5py


# sum over tensor
def sum_tensor(tensor):
    #tensor = Reshape((224,224))(tensor)
    tensor = K.sum(tensor, axis=1)
    return K.sum(tensor, axis=1, keepdims=True)


# sum over tensor (with reshape)
def sum_tensor_block(tensor):
    tensor = Reshape((224,224))(tensor)
    tensor = K.sum(tensor, axis=1)
    return K.sum(tensor, axis=1, keepdims=True)


# mae loss function
def mae_loss(y_pred, y_true):
    tensor_1 = sum_tensor(y_pred)
    tensor_2 = sum_tensor(y_true)
    diff = tensor_1 - tensor_2
    return K.mean(K.abs(diff))


# mae loss for density
def mae_loss_density(y_true, y_pred):
    y_true_sum = K.sum(K.sum(y_true, axis=1), axis=1)
    y_pred_sum = K.sum(K.sum(y_pred, axis=1), axis=1)
    return K.mean(K.abs(y_true_sum-y_pred_sum))


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


# swap axis twice
def swap_tensor_axis(x):
    y = np.moveaxis(x, 1,-1)
    return np.moveaxis(y, 0,-1)


# load & set pretrained weights (vgg16) to model
def set_weight_from_vgg16(weights_path, model):
    f = h5py.File(weights_path)
    loaded_layers = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2']
    assigned_idx = [1, 2, 4, 5, 7, 8]
    n = len(loaded_layers)
    for i in range(n):
        W = f[loaded_layers[i]][loaded_layers[i] + '_W:0']
        b = f[loaded_layers[i]][loaded_layers[i] + '_b:0']
        model.layers[assigned_idx[i]].set_weights([swap_tensor_axis(W), b])


# data agumentation: horizontal flip
def h_flip(X):
    X_fliped = np.flip(X, axis=2)
    return np.vstack((X, X_fliped))


# flip density map (orientation changed after cv2)
def density_map_flip(X):
    flipped = np.flip(X, axis=1)
    return np.flip(flipped, axis=1)


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
                          kernel_regularizer=regularizers.l2(0.01), padding='same', name='dnsty_output')(deconv2)

    # count_output
    count_sum = Lambda(sum_tensor_block)(dnsty_output)
    count_flatten = Flatten()(dnsty_output)
    count_flatten = Dense(100, activation='relu', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(weight_decay), name='fully_connected')(count_flatten)
    count_fc = Dense(1,  activation='linear', kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(0.01), name='count_prediction')(count_flatten)
    count_output = Add(name='count_output')([count_sum, count_fc])


    model = Model(inputs=[inputs], outputs=[dnsty_output, count_output])

    return model


# need to resize
input_shape = (224, 224, 3)
weight_decay = 1e-4

# load data
#p = '/media/inferlabnas/homes/wflores/Data/10707/data/'
p = 'data/'
X_train = np.load(p+'X_train.npy')
Y_train_density = np.load(p+'train_density.npy')
Y_train = np.load(p+'Y_train.npy').astype(np.float32)

X_test = np.load(p+'X_test.npy')
Y_test_density = np.load(p+'test_density.npy')
Y_test = np.load(p+'Y_test.npy').astype(np.float32)


# add dimension
Y_train_density = np.expand_dims(Y_train_density, axis=3)
Y_test_density = np.expand_dims(Y_test_density, axis=3)
Y_train = np.expand_dims(Y_train, axis=1)
Y_test = np.expand_dims(Y_test, axis=1)


# data augmentation
X_train = h_flip(X_train)
Y_train_density = h_flip(Y_train_density)
Y_train = np.vstack((Y_train, Y_train))



model = get_model(input_shape, weight_decay)

# load pretrained weights
#weights_path = 'pretrained_model/vgg16.h5'
#set_weight_from_vgg16(weights_path, model)

#sgd = SGD(lr=0.0001, momentum=0.5, nesterov=False)
adam = Adam(lr=0.0001)
print(model.summary())

#model.compile(loss={'dnsty_output': mse_loss}, optimizer=adam,  metrics=[mae_loss_density])
#hist = model.fit({'inputs': X_train}, {'dnsty_output': Y_train_density},
#                 validation_data=({'inputs': X_test}, {'dnsty_output': Y_test_density}),
#                 epochs=100, batch_size=16, verbose=2)


model.compile(loss={'dnsty_output': mse_loss, 'count_output': mse_loss_count},
              loss_weights={'dnsty_output': 1., 'count_output': 0.01},
              optimizer=adam, metrics=['mae'])
hist = model.fit({'inputs': X_train}, {'dnsty_output': Y_train_density, 'count_output': Y_train},
                 validation_data=({'inputs': X_test}, {'dnsty_output': Y_test_density, 'count_output': Y_test}),
                 epochs=100, batch_size=16, verbose=2)

'''
# save model
model.save('CNN_model_5.h5')

pred = model.predict(X_test)
np.save('./result/pred_5_density.npy', pred)
'''


# save model
model.save('CNN_model_6.h5')

pred = model.predict(X_test)
np.save('./result/pred_6.npy', pred[1])
np.save('./result/pred_6_density.npy', pred[0])


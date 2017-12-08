'''
loss & operation function
'''
import numpy as np
import keras.backend as K
from keras.layers import Reshape
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


# sum over flattened layer
def sum_flatten_layer(tensor):
    tensor = K.sum(tensor, axis=1, keepdims=True)
    return tensor


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



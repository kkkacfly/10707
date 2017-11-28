import numpy as np
from keras import metrics
import keras.backend as K
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.layers import Dense, SimpleRNN, LSTM, merge, Input, Flatten, Lambda, Add
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD

#X_train = np.ones((20, 480*640))
#Y_train = np.ones(20)
#X_valid = np.ones((20, 480*640))
#Y_valid = np.ones(20)

# laod data
p = '/media/inferlabnas/homes/wflores/Data/data/'
X_train = np.load(p+'train_density.npy')
X_valid = np.load(p+'test_density.npy')
Y_train = np.load(p+'Y_train.npy')
Y_valid = np.load(p+'Y_test.npy')

print(X_train.shape)

#n = X_train.shape[0]
#m = X_valid.shape[0]
# reshape X_train, X_valid
# x: (n, 480,640)
#X_train = np.reshape(X_train, (n, 480*640))
#X_valid = np.reshape(X_valid, (m, 480*640))
# add dimension
X_train = np.expand_dims(X_train, axis=1)
X_valid = np.expand_dims(X_valid, axis=1)


# parameter
x_shape = (480, 640)
input_size = (1, x_shape[0]*x_shape[1])
batch_size = 16
hidden_num = 100


def get_sum(x):
    return K.sum(x,axis=1)

def get_sum_2(x):
    return K.sum(x, axis=0)

inputs = Input(shape=input_size)
x = LSTM(hidden_num, return_sequences=True, input_shape=input_size)(inputs)
x = LSTM(hidden_num, return_sequences=True)(x)
x = LSTM(hidden_num)(x)
#y = Lambda((get_sum),output_shape=(1,))(inputs) 
x = Dense(1, activation='linear')(x)
#print(x._keras_shape)
#print(y._keras_shape)
#z = Add()([x, y])
model = Model(inputs, x, name='rLSTM')
sgd = SGD(lr=0.001, momentum=0.5, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae'])
print(model.summary())
hist = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=100, batch_size=batch_size, verbose=2)

train_err = hist.history['loss']
valid_err = hist.history['val_loss']
#np.save(p+'hist', hist)
np.save('train_loss', train_err)
np.save('val_loss', valid_err)




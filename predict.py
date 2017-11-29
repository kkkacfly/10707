import numpy as np


# get mean absolute error
def get_mae_error(y_true, y_pred):
    n = y_pred.shape[0]
    mae = 0
    for i in range(n):
        mae += np.abs(np.sum(y_true[i] - y_pred[i]))
    return mae/n


# main
p = ''


Y_true = np.load(p+'test_density.npy')
Y_pred = np.load(p+'Y_pred.npy')
n = Y_pred.shape[0]
# reduce dimension
Y_pred = np.reshape(Y_pred, (n, 224, 224))

print(get_mae_error(Y_true, Y_pred))

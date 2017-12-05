'''
preprocessing to get:
    - raw image: meet VGG 16 requirements
    - density map
'''
from preprocessing_function import *

from p_test import *

### main

p = ''

'''
# density map: TRANCOS dataset

dim = (480, 640)
dim_resize = (224, 224)
rang = [(7, 477), (3, 633)]
resize_const = 0.16883

gt_density = get_gt_density(p+'train_gt_density.mat', dim, dim_resize, rang, resize_const)
np.save(p+'data/train_density.npy', gt_density)
gt_density = get_gt_density(p+'test_gt_density.mat', dim, dim_resize, rang, resize_const)
np.save(p+'data/test_density.npy', gt_density)
'''


'''
# raw input image: TRANCOS dataset

dim_resize = (224, 224)
global_mean = np.array([150, 152, 152])
array_set = img_2_array(p+'images', 'jpg', p+'masks', 'mat', 1244, dim_resize, global_mean)
X_train = array_set[0:823]
X_test = array_set[823:]
print X_train.shape
print X_test.shape
np.save(p+'data/X_train.npy', X_train)
np.save(p+'data/X_test.npy', X_test)

'''

'''
# ground truth counts
Y_train, Y_test = get_gt_count('ground_truth_count.txt', 823, 421)
np.save('data/Y_train.npy', Y_train)
np.save('data/Y_test.npy', Y_test)
'''

'''
# ground truth counts (sum over density map)
Y_train = get_gt_from_density(p+'train_gt_density.mat')
Y_test = get_gt_from_density(p+'test_gt_density.mat')
np.save('data/Y_train_fd.npy', Y_train)
np.save('data/Y_test_fd.npy', Y_test)
'''




'''


# raw input image: TRANCOS dataset
# adjust brightness

dim_resize = (224, 224)
global_mean = np.array([150, 152, 152])
array_set = random_brightness_contrast(p+'images', 'jpg', p+'masks', 'mat', 1244, dim_resize, p+'TRANCOS_mean.jpg', 'contrast')
X_train = array_set[0:823]
X_test = array_set[823:]
print X_train.shape
print X_test.shape
np.save(p+'data/X_train_brightness.npy', X_train)
np.save(p+'data/X_test_brightness.npy', X_test)

'''


# random crop: rgb
rand_arr = np.vstack((np.random.randint(0,100,1244), np.random.randint(0,130,1244)))
rand_arr = rand_arr.T
'''
dim_resize = (224, 224)
global_mean = np.array([150, 152, 152])
array_set = random_crop_rgb(p+'images', 'jpg', p+'masks', 'mat', 1244, dim_resize, global_mean, rand_arr)
X_train = array_set[0:823]
X_test = array_set[823:]
np.save(p+'data/X_train.npy', X_train)
np.save(p+'data/X_test.npy', X_test)
'''

# random crop: density
dim = (480, 640)
dim_resize = (224, 224)
rang = [(7, 477), (3, 633)]
resize_const = 0.2589

gt_density = random_crop_density(p+'train_gt_density.mat', dim, dim_resize, rang, resize_const, rand_arr[0:823])
np.save(p+'data/train_density.npy', gt_density)
gt_density = random_crop_density(p+'test_gt_density.mat', dim, dim_resize, rang, resize_const, rand_arr[823:])
np.save(p+'data/test_density.npy', gt_density)

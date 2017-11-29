'''
preprocessing to get:
    - raw image: meet VGG 16 requirements
    - density map
'''
from preprocessing_function import *



### main

p = ''


# density map: TRANCOS dataset

dim = (480, 640)
dim_resize = (224, 224)
rang = [(7, 477), (3, 633)]
resize_const = 0.16883

gt_density = get_gt_density(p+'train_gt_density.mat', dim, dim_resize, rang, resize_const)
np.save('data/train_density.npy', gt_density)
gt_density = get_gt_density(p+'test_gt_density.mat', dim, dim_resize, rang, resize_const)
np.save('data/test_density.npy', gt_density)



# raw input image: TRANCOS dataset

dim_resize = (224, 224)
global_mean = np.array([150, 152, 152])
array_set = img_2_array(p+'images', 'jpg', 'masks', 'mat', 1244, dim_resize, global_mean)
X_train = array_set[0:823]
X_test = array_set[823:]
print X_train.shape
print X_test.shape
np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy', X_test)



'''
# ground truth counts
Y_train, Y_test = get_gt_count('ground_truth_count.txt', 823, 421)
np.save('data/Y_train.npy', Y_train)
np.save('data/Y_test.npy', Y_test)
'''

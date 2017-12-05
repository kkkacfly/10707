from __future__ import print_function
from PIL import Image, ImageEnhance, ImageChops
from scipy.io import loadmat
import glob
import numpy as np
import cv2


'''
Objective: get density map from TRANCOS
    - padding ground truth to 480*640
    - reshape to 224*224
Input:
    - dim: original dimension (image)
    - dim_resize: (224, 224)
    - rang: padding constant
    - resize_const: normalization factor (default: 0.16883)

'''
def get_gt_density(filename, dim, dim_resize, rang, resize_const):
    gt = loadmat(filename).get('gtDensities')
    n = gt.shape[0]
    # initial array
    gt_density = np.zeros((n, dim_resize[0], dim_resize[1]))

    for i in range(gt_density.shape[0]):
        temp_gt_matrix = np.zeros(dim)
        # padding
        temp_gt_matrix[rang[0][0]-1:rang[0][1], rang[1][0]-1:rang[1][1]] = gt[i][0]
        # resize
        gt_density[i] = cv2.resize(temp_gt_matrix, dim_resize)/resize_const
    return gt_density.astype(dtype=np.float32)



'''
Objective:
    - convert RGB image to matrix (n, 224, 224, 3)
    - add mask
    - subtract global mean
    - reshape to 224*224
Input:
    - dim: (224, 224)
    - global_mean: np.array([150, 152, 152])

'''
def img_2_array(img_path, img_type, mask_path, mask_type, file_num, dim, global_mean):
    keyword = img_path +'/*.' + img_type
    keyword_mask = mask_path +'/*.' + mask_type
    f_list = glob.glob(keyword)
    mask_list = glob.glob(keyword_mask)
    data_array = np.zeros((file_num, dim[0], dim[1], 3))
    for i in range(file_num):
        print(i)
        img = Image.open(f_list[i])
        # subtract mean
        temp_array = np.array(img, dtype=np.float32) - global_mean
        temp_mask = loadmat(mask_list[i]).get('BW')
        # apply mask
        for j in range(3):
            temp_array[:,:,j] = np.multiply(temp_array[:,:,j], temp_mask)
        # resize
        temp_array = cv2.resize(temp_array, dim)
        # RGB to BGR
        temp_array = temp_array[...,::-1]
        data_array[i,:,:,:] = temp_array

    return data_array


# get ground truth count
def get_gt_count(filename, train_num, test_num):
    value = np.zeros(train_num+test_num)
    with open(filename) as f:
        lines = f.readlines()
    for i in range(train_num+test_num):
        value[i] = float(lines[i].split()[0])
    return value[0:train_num], value[train_num:]


# get ground truth from density map
def get_gt_from_density(filename):
    gt = loadmat(filename).get('gtDensities')
    n = gt.shape[0]
    gt_count = np.zeros(n)
    for i in range(n):
        gt_count[i] = np.sum(gt[i][0])
    return gt_count


# get global mean of dataset (full dataset)
def get_global_mean(img_path, img_type, file_num, raw_dim):
    keyword = img_path +'/*.' + img_type
    f_list = glob.glob(keyword)
    mean_array = np.zeros(3)
    const = raw_dim[0]*raw_dim[1]
    for i in range(file_num):
        #print i
        img = Image.open(f_list[i])
        temp_array = np.array(img, dtype=np.float32)
        for j in range(3):
            mean_array[j] += np.sum(temp_array[:,:,j])/const
    return mean_array/file_num


# --------------------------------------------------------------
# data augmentation


# input: raw image
# crop idx: (100, 130)
def random_crop_rgb(img_path, img_type, mask_path, mask_type, file_num, dim, global_mean, rand_arr):
    # raw input: (480, 640, 3)
    crop_size = (380, 510)
    keyword = img_path +'/*.' + img_type
    keyword_mask = mask_path +'/*.' + mask_type
    f_list = glob.glob(keyword)
    mask_list = glob.glob(keyword_mask)
    data_array = np.zeros((file_num, dim[0], dim[1], 3))
    for i in range(file_num):
        print(i)
        img = Image.open(f_list[i])
        # subtract mean
        temp_array = np.array(img, dtype=np.float32) - global_mean
        temp_mask = loadmat(mask_list[i]).get('BW')
        # apply mask
        for j in range(3):
            temp_array[:,:,j] = np.multiply(temp_array[:,:,j], temp_mask)
        # crop
        temp_array = temp_array[rand_arr[i,0]:rand_arr[i,0]+crop_size[0],rand_arr[i,1]:rand_arr[i,1]+crop_size[1],:]
        # resize
        temp_array = cv2.resize(temp_array, dim)
        # RGB to BGR
        temp_array = temp_array[...,::-1]
        data_array[i,:,:,:] = temp_array

    return data_array


# input density map
# crop to: (380,510)
# crop idx: (100, 130)
# resize_constant: 0.2589
def random_crop_density(filename, dim, dim_resize, rang, resize_const, rand_arr):
    crop_size = (380, 510)
    gt = loadmat(filename).get('gtDensities')
    n = gt.shape[0]
    # initial array
    gt_density = np.zeros((n, dim_resize[0], dim_resize[1]))

    for i in range(gt_density.shape[0]):
        print(i)
        temp_gt_matrix = np.zeros(dim)
        # padding
        temp_gt_matrix[rang[0][0]-1:rang[0][1], rang[1][0]-1:rang[1][1]] = gt[i][0]
        # crop
        cropped_matrix = temp_gt_matrix[rand_arr[i,0]:rand_arr[i,0]+crop_size[0],
                        rand_arr[i,1]:rand_arr[i,1]+crop_size[1]]
        # resize
        gt_density[i] = cv2.resize(cropped_matrix, dim_resize)/resize_const
    return gt_density.astype(dtype=np.float32)


# input: raw image only
def random_brightness_contrast(img_path, img_type, mask_path, mask_type, file_num, dim, img_mean_path, option):
    keyword = img_path +'/*.' + img_type
    keyword_mask = mask_path +'/*.' + mask_type
    f_list = glob.glob(keyword)
    mask_list = glob.glob(keyword_mask)
    img_mean = Image.open(img_mean_path)
    data_array = np.zeros((file_num, dim[0], dim[1], 3))
    for i in range(file_num):
        print(i)
        img = Image.open(f_list[i])
        # subtract mean
        img = ImageChops.subtract(img, img_mean)
        # adjust brightness / contrast
        if option == 'contrast':
            enhancer = ImageEnhance.Contrast(image=img)
            rand_num = np.random.uniform(0.2,0.8)
            img = enhancer.enhance(rand_num)
        elif option == 'brightness':
            enhancer = ImageEnhance.Brightness(image=img)
            rand_num = np.random.uniform(0.2,0.8)
            img = enhancer.enhance(rand_num)
        else:
            raise ValueError('keyword does not match')
        temp_array = np.array(img, dtype=np.float32)
        temp_mask = loadmat(mask_list[i]).get('BW')
        # apply mask
        for j in range(3):
            temp_array[:,:,j] = np.multiply(temp_array[:,:,j], temp_mask)
        # resize
        temp_array = cv2.resize(temp_array, dim)
        # RGB to BGR
        temp_array = temp_array[...,::-1]
        data_array[i,:,:,:] = temp_array

    return data_array

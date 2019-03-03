from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from keras.utils.np_utils import to_categorical

import numpy as np
import argparse
from PIL import Image

import h5py
from glob import glob
import string

import sys
sys.path.append('/Users/sandhya/Documents/repos/cv-translator/')

from label import preprocessImg
import cv2

# LOAD FROM SINGLE DIRECTORY
# SPLIT DATA INTO TRAIN/TEST
# SAVE IN TRAIN/TEST FOLDERS USING THE SAME SUB-DIRECTORY STRUCTURE

# LOAD FROM SINGLE DIRECTORY

train_data_dir = "/Users/sandhya/Documents/repos/cv-translator/asl_dataset/"
list_dirs = glob(train_data_dir+"*/")

image_size = 244
num_channels = 3
all_labels = np.empty(0, dtype=np.float32) 
all_images = np.empty((0, image_size, image_size, num_channels), dtype=np.float32) 
for folder in list_dirs:
    img_out, label_out = preprocessImg(folder, 
                                      size = image_size)
                                      # num_channels = num_channels,
                                      # drop_green=False,
                                      # gray=True)
  
    all_labels = np.concatenate((all_labels, label_out), axis=0)
    all_images = np.concatenate((all_images, img_out), axis=0)


# SPLIT DATA INTO TRAIN/TEST
#convert class letters to integer values

class_index = {}

for pos, letter in enumerate(string.ascii_lowercase):
    class_index[letter] = pos
    
labels_out = np.copy(all_labels)
labels_final = [class_index[value] for value in labels_out]

# stratified train-test split

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=123)

X = all_images.copy()
y = np.copy(labels_final)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

#all_images = np.save("all_images.npy", images)
#all_labels = np.save("all_labels.npy", labels_final)

from keras.utils.np_utils import to_categorical

train_labels = to_categorical(y_train, num_classes=26)
test_labels = to_categorical(y_test, num_classes=26)

train_labels.shape, test_labels.shape

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# SAVE IN TRAIN/TEST FOLDERS USING THE SAME SUB-DIRECTORY STRUCTURE

#recreate labels
pred_index = {}

for pos, letter in enumerate(string.ascii_lowercase):
    pred_index[pos] = letter
    
#labels
save_train_labels = [pred_index[value] for value in y_train]
save_test_labels = [pred_index[value] for value in y_test]

train_output_dir = "/Users/sandhya/Documents/repos/cv-translator/split_data/train/"
validation_output_dir = "/Users/sandhya/Documents/repos/cv-translator/split_data/validation/"


#create subdirectory structure

import os

for letter in string.ascii_lowercase:
    os.makedirs(train_output_dir+letter, exist_ok=True)
    os.makedirs(validation_output_dir+letter, exist_ok=True)


# write TRAIN images to correct subdirectory!

prefix = "/processed_"
suffix = ".jpeg"

count=0
for img in X_train:
    sub_dir = save_train_labels[count]
    outname = prefix + sub_dir + "_%03d%s" % (count, suffix)
    cv2.imwrite(train_output_dir + sub_dir + outname, img)
    count+=1

# write VALIDATION images to correct subdirectory!

prefix = "/processed_"
suffix = ".jpeg"

count=0
for img in X_test:
    sub_dir = save_test_labels[count]
    outname = prefix + sub_dir + "_%03d%s" % (count, suffix)
    cv2.imwrite(validation_output_dir + sub_dir + outname, img)
    count+=1
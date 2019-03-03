# import the necessary packages
import numpy as np
import os
import h5py
from glob import glob
import string
import joblib

from PIL import Image

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet

from keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout, Convolution2D, MaxPooling2D, Activation
from keras.models import Sequential, Model
from keras.callbacks import History, ModelCheckpoint, Callback
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Nadam

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier

#LOAD DATA

vgg_train = joblib.load("/Users/sandhya/Documents/repos/cv-translator/bottleneck/vgg16_train_bottleneck_features.pkl")
vgg_train_y = joblib.load("/Users/sandhya/Documents/repos/cv-translator/bottleneck/vgg16_train_bottleneck_labels.pkl")
vgg_test = joblib.load("/Users/sandhya/Documents/repos/cv-translator/bottleneck/vgg16_test_bottleneck_features.pkl")
vgg_test_y = joblib.load("/Users/sandhya/Documents/repos/cv-translator/bottleneck/vgg16_test_bottleneck_labels.pkl")

vgg = [vgg_train, vgg_train_y, vgg_test, vgg_test_y]

def my_preprocess(x):
    train_y = to_categorical(x[1], num_classes=26)
    test_y = to_categorical(x[3], num_classes=26)
    reshaped_test = x[2].reshape(x[2].shape[0], -1)
    reshaped_train = x[0].reshape(x[0].shape[0], -1)
    return (train_y, test_y, reshaped_train, reshaped_test)


#################
batch_size = 32
nb_classes = 26
nb_epoch = 100
image_size = 244
num_channels = 'rgb'
#################

names = ["vgg"]
c = 0

for net in [vgg]:
            
    train_y, test_y, train_x, test_x = my_preprocess(net)
    filepath = "snapshot_" + names[c] + "_weights.hdf5"
    
    # TO SAVE CONTINUOUS SNAPSHOTS
    save_snapshots = ModelCheckpoint(filepath,
                                     monitor='val_acc',
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max',
                                     verbose=1)
    
    # Save loss history
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.accuracy = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.accuracy.append(logs.get('acc'))

    loss_history = LossHistory()
    callbacks_list = [save_snapshots, loss_history]
    
    # LOAD MODEL
    
    print("loading model...")
    model = Sequential()

    model.add(Dense(256, input_shape=(train_x.shape[1], )))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

    print("model loaded.")
    
    # TRAIN MODEL
    
    print("fitting model...")
    
    neural_net = model.fit(train_x, train_y, batch_size=64, epochs=50, verbose=0, callbacks=callbacks_list, 
                         validation_split=0.0, validation_data=(test_x, test_y), shuffle=True, 
                         class_weight=None, sample_weight=None, initial_epoch=0)
    
    print("done!")
    print("saving logs + weights...")
    
    # SAVE WEIGHTS + LOGS
    model.save_weights("/Users/sandhya/Documents/repos/cv-translator/bottleneck/vgg16_bottleneck_fc_model.npy")
    
    evaluation_cost = neural_net.history['val_loss']
    evaluation_accuracy = neural_net.history['val_acc']
    training_cost = neural_net.history['loss']
    training_accuracy = neural_net.history['acc']

    np.save(names[c] + "_evaluation_cost.npy", evaluation_cost)
    np.save(names[c] + "_evaluation_accuracy.npy", evaluation_accuracy)
    np.save(names[c] + "_training_cost.npy", training_cost)
    np.save(names[c] + "_training_accuracy.npy", training_accuracy)
    c+=1



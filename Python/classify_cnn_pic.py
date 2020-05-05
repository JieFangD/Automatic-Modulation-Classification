# coding: utf-8
import time
import numpy as np
import sys
import random
import itertools
import pydot
import graphviz
import keras.backend as K
from keras.models import Sequential, model_from_json, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten, Reshape, Input, Lambda
from keras.optimizers import SGD, Adam
from keras.utils import np_utils,plot_model
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import scipy.io
from util import *

train_path = sys.argv[1]
test_path = sys.argv[2]
resolution = 36
tmp = train_path.split('_')
tmp = tmp[2].split('.')
SNR = tmp[0]
R = 0   #Read Polar Model
duplicate = 0

def train(x_train, y_train, x_val, y_val, x_test, y_test):
    t = int(2**(int(sys.argv[3])))
    cnn_input = Input(shape=(36,36,1))
    cnn_batch = BatchNormalization()(cnn_input)
    conv1 = Convolution2D(2*t,(3,3),padding='same',activation='relu')(cnn_batch)
    max1 = MaxPooling2D((3,3))(conv1)
    bat1 = BatchNormalization()(max1)
    conv2 = Convolution2D(1*t,(3,3),padding='same',activation='relu')(bat1)
    max2 = MaxPooling2D((3,3))(conv2)
    flat = Flatten()(max2)
    den1 = Dense(2*t,activation='relu')(flat)
    den1 = BatchNormalization()(den1)
    den2 = Dense(1*t,activation='relu')(den1)
    den2 = BatchNormalization()(den2)
    den3 = Dense(4,activation='softmax')(den2)
    model = Model(cnn_input,den3)
    model.summary()
    
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

    earlystopping = EarlyStopping(monitor='val_acc', patience = 8, verbose=0, mode='max')
    checkpoint = ModelCheckpoint(filepath='model.h5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_acc',
                                 mode='max')
    start_time = time.time()
    result = model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=100,epochs=20,shuffle=True,callbacks=[earlystopping,checkpoint],verbose=1)
    print("---Train: %s seconds ---" % (time.time() - start_time))

    #model.summary()
    model.load_weights('model.h5')
    start_time = time.time()
    y = model.predict(x_test)
    print("---Test: %s seconds ---" % (time.time() - start_time))

    scores = model.evaluate(x_test,y_test)
    print('scores: ',scores)
    #plot_fig(result)
    return model

def to2dim(sig):
    out = np.zeros((sig.shape[0],sig.shape[1],2))
    out[:,:,0] = sig.real
    out[:,:,1] = sig.imag
    return out

def main():
    (x_train, y_train) = load_mat(train_path,0)
    (x_train, y_train),(x_val,y_val) = split_data(x_train,y_train,0.2)
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    indices = np.arange(x_val.shape[0])
    np.random.shuffle(indices)
    x_val = x_val[indices]
    y_val = y_val[indices]
    
    x_train = sig2pic(x_train,-3,3,resolution)
    x_val = sig2pic(x_val,-3,3,resolution)

    (x_test, y_test) = load_mat(test_path,0)
    x_test = sig2pic(x_test,-3,3,resolution)
    
    model = train(x_train, y_train, x_val, y_val, x_test, y_test)
    model_json = model.to_json()
    with open("Model/cnn.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("Model/cnn_"+SNR+".h5")
    for i in range(4):
        n = int(x_test.shape[0]/4)
        scores = model.evaluate(x_test[n*i:n*(i+1),:],y_test[n*i:n*(i+1),:])
        print('scores: ',scores)
    
if __name__ == "__main__":
    main()

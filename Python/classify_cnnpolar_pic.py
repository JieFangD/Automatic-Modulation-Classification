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
resolution2 = 36
tmp = train_path.split('_')
tmp = tmp[2].split('.')
SNR = tmp[0]
R = 0   #Read Polar Model
duplicate = 0

def polar(x):
    square = K.square(x)
    abs_sig = K.sqrt(K.sum(square,axis=2,keepdims=True))
    abs_sig = K.reshape(abs_sig,(K.shape(abs_sig)[0],K.shape(abs_sig)[1],))
    div = tf.div(x[:,:,1],x[:,:,0])
    arctan = tf.atan(div)
    return K.concatenate([abs_sig,arctan],axis=1)

def pic2gauss(sig,x0,x1,y0,y1,r0,r1,l):
    linx = np.linspace(x0, x1, r0)
    liny = np.linspace(y0, y1, r1)
    Px = tf.reshape(sig[:,:l],(-1,l,1))
    Py = tf.reshape(sig[:,l:],(-1,l,1))
    tmpx = tf.reshape(tf.square(Px-linx),(-1,l,r0,1))
    tmpy = tf.reshape(tf.square(Py-liny),(-1,l,1,r1))
    tmp = tmpx+tmpy
    out = tf.reduce_sum(tf.exp(-1*tmp/2/0.05),axis=1)
    out = tf.reshape(out,(-1,r0,r1,1))
    return out

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

def fine_tune(x_train, y_train, x_val, y_val, x_test, y_test, model):
    i = 0
    for layer in model.layers:
        layer.trainable = True

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

    earlystopping = EarlyStopping(monitor='val_acc', patience = 30, verbose=0, mode='max')
    checkpoint = ModelCheckpoint(filepath='model.h5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_acc',
                                 mode='max')
    start_time = time.time()
    result = model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=10,epochs=500,shuffle=True,callbacks=[earlystopping,checkpoint],verbose=1)
    print("---Train: %s seconds ---" % (time.time() - start_time))
    model.load_weights('model.h5')
    scores = model.evaluate(x_test,y_test)
    print(scores)
    pre = model.predict(x_test,batch_size=100)
    print(pre)
    prob = np.mean(np.amax(pre,axis=1))
    print(prob)
    return model

def topolar(sig):
    out = np.zeros((sig.shape[0],sig.shape[1]*2))
    for i in range(sig.shape[0]):
        out[i][:sig.shape[1]] = np.abs(sig[i])
        out[i][sig.shape[1]:] = np.arctan(sig[i].imag/sig[i].real)
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
    
    x_train = topolar(x_train)
    x_val = topolar(x_val)
    
    l = x_train.shape[1]
    input_sig = Input(shape=(l,2))
    sig = Lambda(polar,output_shape=[2*l])(input_sig)
    output_pic = Lambda(pic2gauss,arguments={'x0':0,'x1':3,'y0':-1.6,'y1':1.6,'r0':resolution,'r1':resolution2,'l':l})(sig)
    final = Model(input_sig,output_pic)
    start_time = time.time()
    # Use Gaussian distribution
    #x_train = final.predict(x_train,batch_size=100)
    #x_val = final.predict(x_val,batch_size=100)
    x_train = sig2pic1(x_train,0,3,-1.6,1.6,resolution,resolution2)
    x_val = sig2pic1(x_val,0,3,-1.6,1.6,resolution,resolution2)

    print(time.time()-start_time)

    (x_test, y_test) = load_mat(test_path,0)
    x_test = topolar(x_test)
    # Use Gaussian distribution
    #x_test = final.predict(x_test,batch_size=100)
    x_test = sig2pic1(x_test,0,3,-1.6,1.6,resolution,resolution2)
    
    
    model = train(x_train, y_train, x_val, y_val, x_test, y_test)
    model_json = model.to_json()
    with open("Model/cnnpolar.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("Model/cnnpolar_"+SNR+".h5")
    for i in range(4):
        n = int(x_test.shape[0]/4)
        scores = model.evaluate(x_test[n*i:n*(i+1),:],y_test[n*i:n*(i+1),:])
        print('scores: ',scores)
    
if __name__ == "__main__":
    main()

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pydot
import graphviz
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils,plot_model
from keras.models import Sequential, model_from_json
import keras.backend as K
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

def load_mat(path,s):
    Data = scipy.io.loadmat(path)
    x_train = Data['x']
    y_train = np_utils.to_categorical(Data['y'],4)
    '''
    Data1 = scipy.io.loadmat('raw20000_1000_8.mat')
    Data2 = scipy.io.loadmat('raw20000_1000_8_2.mat')
    Data3 = scipy.io.loadmat('raw20000_1000_8_3.mat')
    Data4 = scipy.io.loadmat('raw20000_1000_8_4.mat')
    Data5 = scipy.io.loadmat('raw20000_1000_8_5.mat')
    Data6 = scipy.io.loadmat('raw20000_1000_8_6.mat')
    Data7 = scipy.io.loadmat('raw20000_1000_8_7.mat')
    Data8 = scipy.io.loadmat('raw20000_1000_8_8.mat')
    x_train1 = Data1['x']
    y_train1 = np_utils.to_categorical(Data1['y'],4)
    x_train2 = Data2['x']
    y_train2 = np_utils.to_categorical(Data2['y'],4)
    x_train3 = Data3['x']
    y_train3 = np_utils.to_categorical(Data3['y'],4)
    x_train4 = Data4['x']
    y_train4 = np_utils.to_categorical(Data4['y'],4)
    x_train5 = Data5['x']
    y_train5 = np_utils.to_categorical(Data5['y'],4)
    x_train6 = Data6['x']
    y_train6 = np_utils.to_categorical(Data6['y'],4)
    x_train7 = Data7['x']
    y_train7 = np_utils.to_categorical(Data7['y'],4)
    x_train8 = Data8['x']
    y_train8 = np_utils.to_categorical(Data8['y'],4)
    '''
    #x_train = np.concatenate((x_train1,x_train2),axis=0)
    #x_train = np.concatenate((x_train,Data1['x'],Data2['x'],Data3['x'],Data4['x']),axis=0)
    #y_train = np.concatenate((y_train1,y_train2),axis=0)
    #y_train = np.concatenate((y_train,np_utils.to_categorical(Data1['y'],4),np_utils.to_categorical(Data2['y'],4),np_utils.to_categorical(Data3['y'],4),np_utils.to_categorical(Data4['y'],4)),axis=0)
    if(s):
        v = Data['v']
        return (x_train, y_train, v)
    return (x_train, y_train)

def load_mat1(path):
    Data = scipy.io.loadmat(path)
    x_train = Data['x']
    y_train = Data['y']
    return (x_train, y_train)

def write_loss(result,name):
    scipy.io.savemat(name,{'loss':result.history['val_loss']})

def split_data(X,Y,split_ratio):
    l = int(X.shape[0]/4)
    indices = np.arange(l)
    np.random.shuffle(indices)
    num = int(split_ratio * l)
    indices_train = np.concatenate((indices[num:],indices[num:]+l,indices[num:]+2*l,indices[num:]+3*l))
    indices_test = np.concatenate((indices[:num],indices[:num]+l,indices[:num]+2*l,indices[:num]+3*l))

    X_train = X[indices_train]
    Y_train = Y[indices_train]
    X_val = X[indices_test]
    Y_val = Y[indices_test]
    
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    #print(indices_train)
    #print(indices_test)

    indices = np.arange(X_val.shape[0])
    np.random.shuffle(indices)
    X_val = X_val[indices]
    Y_val = Y_val[indices]
   
    return (X_train,Y_train),(X_val,Y_val)

def split_data2(X,Y,Z,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 

    X_data = X[indices]
    Y_data = Y[indices]
    Z_data = Z[indices]

    num_validation_sample = int(split_ratio * X_data.shape[0] )

    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]
    Z_train = Z_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]
    Z_val = Z_data[:num_validation_sample]

    return (X_train,Y_train,Z_train),(X_val,Y_val,Z_val)

def sig2pic(sig,v0,v1,resolution):
    v = (v1-v0)/resolution
    out = np.zeros((sig.shape[0],resolution,resolution,1))
    for j in range(sig.shape[0]):
        for i in range(sig.shape[1]):
            x = int((sig[j][i].real-v0)/v)
            y = int((sig[j][i].imag-v0)/v)
            if(x > resolution-1):
                x = resolution-1
            elif(x < 0):
                x = 0
            if(y > resolution-1):
                y = resolution-1
            elif(y < 0):
                y = 0
            out[j,x,y,0] = 1 #non accumulated
            #out[j,x,y,0] = out[j,x,y,0] + 1 #accumulated
    return out

def sig2pic1(sig,x0,x1,y0,y1,resolution,resolution2):
    linx = (x1-x0)/(resolution-1)
    liny = (y1-y0)/(resolution2-1)
    out = np.zeros((sig.shape[0],resolution,resolution2,1))
    l = int(sig.shape[1]/2)
    for j in range(sig.shape[0]):
        for i in range(l):
            x = int((sig[j][i]-x0)/linx)
            y = int((sig[j][l+i]-y0)/liny)
            if(x > resolution-1):
                x = resolution-1
            elif(x < 0):
                x = 0
            if(y > resolution2-1):
                y = resolution2-1
            elif(y < 0):
                y = 0
            #out[j,x,y,0] = 1 #non accumulated
            out[j,x,y,0] = out[j,x,y,0] + 1 #accumulated
    return out

def sig2gauss(sig,x0,x1,y0,y1,r0,r1):
    l = int(sig.shape[1]/2)
    linx = np.linspace(x0, x1, r0)
    liny = np.linspace(y0, y1, r1)
    Px = np.reshape(sig[:,:l],(-1,l,1))
    Py = np.reshape(sig[:,l:],(-1,l,1))
    tmpx = np.reshape(np.square(Px-linx),(-1,l,r0,1))
    tmpy = np.reshape(np.square(Py-liny),(-1,l,1,r1))
    out = np.zeros((sig.shape[0],r0,r1,1))
    for i in range(sig.shape[0]):
        tmp = tmpx[i]+tmpy[i]
        tmp = np.sum(np.exp(-1*tmp/2/0.05),axis=0)
        out[i] = np.reshape(tmp,(-1,r0,r1,1))
    return out

def readModel(f1, f2):
    import keras.backend as K
    json_file = open(f1, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json,custom_objects={"backend": K,'tf':tf})
    model.load_weights(f2)
    return model

def plot_fig(result):
    plt.figure
    plt.plot(result.epoch,result.history['acc'],label="acc")
    plt.plot(result.epoch,result.history['val_acc'],label="val_acc")
    plt.scatter(result.epoch,result.history['acc'],marker='*')
    plt.scatter(result.epoch,result.history['val_acc'])
    plt.legend(loc='under right')
    plt.show()

    plt.figure
    plt.plot(result.epoch,result.history['loss'],label="loss")
    plt.plot(result.epoch,result.history['val_loss'],label="val_loss")
    plt.scatter(result.epoch,result.history['loss'],marker='*')
    plt.scatter(result.epoch,result.history['val_loss'],marker='*')
    plt.legend(loc='upper right')
    plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


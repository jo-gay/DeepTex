# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import keras
import h5py
import random
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import optimizers
filedir = 'data/'

# Here we just make sure the image format is as desired. This will make the feature (x)
# data - i.e. the RGB pixel values - for each image have the shape 3x32x32.
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

# This is the main function. You need to write the getModel and fitModel functions to pass to this.
# Call your functions 'myGetModel' and 'myFitModel'.
# The getModel function should accept an object of the CIFAR class, and return a compiled Keras CNN model. 
# In this function you will specify the network structure (including regularization) and the optimizer to 
# be used (and its parameters like learning rate), and run compile the model (in the Keras sense of running 
# model.compile).
# The fitModel function should accect two arguments. The first is the CNN model you return from your getModel 
# function, and the second is the CIFAR classed data object. It will return a trained Keras CNN model, which 
# will then be applied to the test data. In this function you will train the model, using the Keras model.fit 
# function. You will need to specify all parameters of the training algorithm (batch size, etc), and the 
# callbacks you will use (EarlyStopping and ModelCheckpoint). You will need to make sure you save and load 
# into the model the weight values of its best performing epoch.
    
def myGetModel(x_tr,y_tr, x_va,y_va, x_te,y_te):
    num_classes = 2
    cNN = Sequential()
    cNN.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(1,80,80), padding ="same"))
    cNN.add(Dropout(0.5))
    
    cNN.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    cNN.add(MaxPooling2D(pool_size=(2, 2)))
    cNN.add(Dropout(0.25))
    
    cNN.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    cNN.add(Dropout(0.5))
    
    cNN.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    cNN.add(MaxPooling2D(pool_size=(2, 2)))
    cNN.add(Dropout(0.5))
    cNN.add(Flatten())
    
    cNN.add(Dense(1024, activation='relu'))
    cNN.add(Dropout(0.5))
    cNN.add(Dense(num_classes, activation='softmax'))
    
    sgd = optimizers.SGD(lr=0.1, decay=1e-6)
    #adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999)
    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    cNN.compile(loss = keras.losses.categorical_crossentropy ,metrics = ['accuracy'], optimizer = adam)
    return cNN
    
def myFitModel(cNN,x_tr,y_tr, x_va,y_va, x_te,y_te):
    path = "C:/Users/Bulb/Documents/Teknisk Fysik/15hp project/weights-best.hdf5"
    model_checkpoint = ModelCheckpoint(filepath = path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=1)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None)
    cNN.fit(x_tr, y_tr, epochs = 1, batch_size = 100, validation_data = (x_va, y_va),callbacks = [model_checkpoint, early_stopping])
    cNN.load_weights(path)
    return cNN
    
def runImageClassification(getModel=None,fitModel=None,seed=7):
    # Fetch data. You may need to be connected to the internet the first time this is done.
    # After the first time, it should be available in your system. On the off chance this
    # is not the case on your system and you find yourself repeatedly downloading the data, 
    # you should change this code so you can load the data once and pass it to this function. 
    print("Preparing data...")
    class_names=['healthy','cancer']
        
    filedir = 'data/'
    fold1tr=[[3,4,5,6],[37,38,12]]
    fold2tr=[[3,4,5,6],[36,37,38]]
    fold3tr=[[3,4,5,6],[12,36]]
    fold1te=[[7,8],[36]]
    fold2te=[[7,8],[12]]
    fold3te=[[7,8],[37,38]]
        
    # Extracting data from files
    data_fold1_tr=[]
    labels_fold1_tr=[]
    
    for g in fold1tr[0]:
        filename = filedir+'glass'+str(g)+'.hdf5'
        f = h5py.File(filename, 'r')
        
        a_group_key = list(f.keys())[0]
        d=list(f[a_group_key])
        data_fold1_tr += d
        labels_fold1_tr += list(np.zeros(len(d)))
            
    for g in fold1tr[1]:
        filename = filedir+'glass'+str(g)+'.hdf5'
        f = h5py.File(filename, 'r')
        
        a_group_key = list(f.keys())[0]
        d=list(f[a_group_key])
        data_fold1_tr += d
        labels_fold1_tr += list(np.ones(len(d)))
        
    # Shuffling data and partitioning into training, testing, and validation sets    
    nSamples=len(labels_fold1_tr)
    tr_perc=.75
    va_perc=.15
    # te_perc = 0.2
    
    seed = 1
    random.seed(seed)
    random.shuffle(labels_fold1_tr)
    random.seed(seed)
    random.shuffle(data_fold1_tr)
    
    tr=round(nSamples*tr_perc)
    va=round(nSamples*va_perc)
        
    x_tr = np.asarray(data_fold1_tr[0:tr])
    x_va = np.asarray(data_fold1_tr[tr:(tr+va)])
    x_te = np.asarray(data_fold1_tr[(tr+va):nSamples])
    y_tr = np.asarray(labels_fold1_tr[0:tr])
    y_va = np.asarray(labels_fold1_tr[tr:(tr+va)])
    y_te = np.asarray(labels_fold1_tr[(tr+va):nSamples])

    # converting vectors to numpy arrays
    y_tr = np_utils.to_categorical(y_tr, 2)
    y_va = np_utils.to_categorical(y_va, 2)
    y_te = np_utils.to_categorical(y_te, 2)
    x_tr = x_tr.astype('float32')
    x_va = x_va.astype('float32')
    x_te= x_te.astype('float32')
    x_tr /= 255
    x_va /= 255
    x_te /= 255
        
    
    # Create model 
    print("Creating model...")
    model=myGetModel(x_tr,y_tr, x_va,y_va, x_te,y_te)
    
    # Fit model
    print("Fitting model...")
    model=myFitModel(model,x_tr,y_tr, x_va,y_va, x_te,y_te)

    # Evaluate on test data
    print("Evaluating model...")
    score = model.evaluate(x_te, y_te, verbose=0)
    print('Test accuracy:', score[1])






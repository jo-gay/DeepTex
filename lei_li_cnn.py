# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:04:29 2018

@author: Bulb
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
import h5py, sys
import random
import tensorflow as tf
from keras.utils import np_utils
from keras import layers
from keras import models
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Add
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras import optimizers
from keras.engine.topology import Layer
from keras.utils.generic_utils import get_custom_objects
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from first_model_cnn import loadData, doInitTasks



img_height = 80
img_width = 80
img_channels = 1
img_shape=(img_height, img_width, img_channels)

class GateLayer(Activation):
    
    def __init__(self, activation, **kwargs):
        super(GateLayer, self).__init__(activation, **kwargs)
        self.__name__ = 'gate_layer'

#%% MAIN SCRIPT
def main(GPU_NUM):
    
    print('GPU_NUM: ')
    print(GPU_NUM)
    sess = doInitTasks(GPU_NUM)
    with sess.as_default():
        y_te, res = runLeiLi(foldNr=2)

    preds=[int(x>0.5) for x in res]
    #truth=np.argmax(res[0], axis=1)
    print(confusion_matrix(y_te, preds))


def myFitModel(cNN,epochs, x_tr,y_tr, x_va,y_va, batch_size):
    path = "weights-best.hdf5"
    nr_training = len(x_tr)
    #batch_size = batchS
    lr_sched = step_decay_schedule( 0.01, nr_training, batch_size)
    model_checkpoint = ModelCheckpoint(filepath = path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=1)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None)
    cNN.fit(x_tr, y_tr, batch_size=batch_size, epochs = epochs, validation_data = (x_va, y_va),callbacks = [lr_sched, model_checkpoint, early_stopping])
    cNN.load_weights(path)
    return cNN
   
def step_decay_schedule(initial_lr, nr_training, batch_size):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    
    def schedule(epoch):
        decay = 1
        if(epoch*(nr_training/batch_size) >= 3000):
            decay = decay /10
        if(epoch*(nr_training/batch_size) >= 6000):
            decay = decay /10
        print('learning rate: %f' %(initial_lr*decay))    
        return initial_lr * decay
    return LearningRateScheduler(schedule)


#%% LBP-CNN designed by Lei Li et al. 
def getCNN(x):

    def getWeights(j, nb_channels, filterNo):
        
        if(False):
            print("input channels: ")
            print(nb_channels)
            
        weights = np.array([[0,0,0],[0,-1,0],[0,0,0]])
        if(j==0):
            weights[0][0] = 1
        if(j==1):   
            weights[0][1] = 1
        if(j==2):
            weights[0][2] = 1
        if(j==3): 
            weights[1][0] = 1
        if(j==4):
            weights[1][2] = 1
        if(j==5):
            weights[2][0] = 1
        if(j==6):
            weights[2][1] = 1
        if(j==7):    
            weights[2][2] = 1  
        
        weightTensor = np.zeros((nb_channels,3,3))
        weightTensor[filterNo,] = weights
        #print("weigthTensor:" )
        #print(weightTensor)
        #print("_____________________")
        bias=np.zeros(1)   # one filter  
        weightTensor=weightTensor.reshape(3,3,nb_channels,1) 
        return np.asarray([weightTensor, bias])
        
    def lb_convolution(y, j, nb_channels):
        
        groups = []
        for x in range(nb_channels):
            groups.append(layers.Conv2D(filters = 1, kernel_size=(3,3), strides=(1,1), 
                              weights=getWeights(j, nb_channels, x), 
                              padding='same', trainable=False)(y))
        #added_layers = layers.add(groups)    
        if(nb_channels > 1):
            y = layers.concatenate(groups)
        else:
            y = groups[0]
    
        y = layers.BatchNormalization()(y)
        y = Activation('sigmoid')(y)
        #y.add(Lambda(lambda y: y * 2**j))
        return y
        
    def LBP_Module(y, nb_channels):
        """
        returns a LBP convolutional stack, consisting of eight LBP convolutional
        layers in parallel
        """       
        #print("in LBP_MODULE:convolution: y")
        #print(y)
        #print(type(y))
        groups = []
        for j in range(8):
            groups.append(lb_convolution(y,j, nb_channels))
        
        y = layers.Add()(groups)
        return y    
        # the grouped convolutional layer concatenates them as the outputs of the layer 
        
        
    def Gate_Activation(y,filterNo,nb_channels):
        def gate_layer(y):
            lower = filterNo
            upper = filterNo+1
        
            # sets values outside interval (lower, upper) to zero
            
            y = keras.backend.switch(y > (y*0 +lower), y, y*0)
            y = keras.backend.switch(y < (y*0 +upper), y, y*0)
            
            # gate function
            y = keras.backend.switch((y-lower)< 0.5, 4*(y-lower), -4*(y-upper))
            return y 
        get_custom_objects().update({'gate_layer': GateLayer(gate_layer)}) 
        return (Activation('gate_layer')(y))  

    ### PART 1 - CONVOLUTIONAL LAYERS ###
    initial_channels = 8   
    output_channels = 16
    x = layers.Conv2D(initial_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(output_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    
    ### Part 2 - PARALLEL LPB LAYERS ###
    x = LBP_Module(x, output_channels)
    
    ### Part 3 - GATE LAYER ###
    groups = []
    for j in range(8):
          groups.append(Gate_Activation(x,j,output_channels))
    x = layers.concatenate(groups)
     
    x = layers.AveragePooling2D(pool_size=(40, 40))(x)
    ### DENSE LAYERS ###
    x = layers.Flatten()(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = Activation('relu')(x)
    x = layers.Dense(1, activation = 'sigmoid')(x)
    #x = layers.Dense(1, activation = 'softmax')(x) # original settings
    return x

#%% Fetches and compiles the model with the chosen optimizer.     
def getLeiLi():
    image_tensor = layers.Input(shape=img_shape)
    network_output = getCNN(image_tensor)
      
    model = models.Model(inputs=[image_tensor], outputs=[network_output])
    print(model.summary())
#    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, decay=0.0001)
    #adam = optimizers.Adam(lr=0.0001)
    model.compile(loss = keras.losses.binary_crossentropy, metrics = ['accuracy'], optimizer = 'adam')
    #model.compile(loss = keras.losses.categorical_crossentropy ,metrics = ['accuracy'], optimizer = adam)
    return model    

def runLeiLi(foldNr=2, eval_model=0, seed=None, batch_size = 100, num_epochs = 10):
    
    
    print("Preprocessing data...")
    x_tr, x_va, x_te, y_tr, y_va, y_te = loadData(foldNr=foldNr, valPercent=0.2, 
                                                  augment=True, seed=seed, 
                                                  changeOrder=True, normalize=True)
   
    # Create model 
    print("Creating model...")
    model=getLeiLi()
    
    if(eval_model is 1):
        print("evaluating model on existing weights.")
        results = model.predict(x_te)
        score = model.evaluate(x_te, y_te, verbose=0)
        print('Test accuracy:', score[1])
            
    # Fit model
    print("Fitting model...")
    epochs = num_epochs
    model=myFitModel(model,epochs,x_tr,y_tr,x_va,y_va, batch_size)

    # Evaluate on test data
    print("Evaluating model...")
    path_best = "weights-best.hdf5"
    model.load_weights(path_best)
    results = model.predict(x_te)
    score = model.evaluate(x_te, y_te, verbose=0)
    print('Test accuracy:', score[1])
    
    print('results data type:')
    print(type(results))

    return y_te, results 


if __name__ == "__main__":
    if len(sys.argv) > 1:
        GPU_NUM = int(sys.argv[1])
        print("Using GPU %d"%GPU_NUM)
    else:
        print("Using CPU only")
        GPU_NUM=None

    main(GPU_NUM)
    
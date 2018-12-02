# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:04:29 2018

@author: Bulb
"""

import numpy as np
import keras
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



img_height = 80
img_width = 80
img_channels = 1
img_shape=(img_height, img_width, img_channels)

class GateLayer(Activation):
    
    def __init__(self, activation, **kwargs):
        super(GateLayer, self).__init__(activation, **kwargs)
        self.__name__ = 'gate_layer'


def myFitModel(cNN,epochs, x_tr,y_tr, x_va,y_va, batch_size):
    path = "weights-best.hdf5"
    lr_sched = proportional_decay_schedule( 0.01, 0.8, 1)
    model_checkpoint = ModelCheckpoint(filepath = path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=1)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None)
    mHist=cNN.fit(x_tr, y_tr, batch_size=batch_size, epochs = epochs, validation_data = (x_va, y_va),callbacks = [lr_sched, model_checkpoint, early_stopping])
    cNN.load_weights(path)
    return cNN, mHist
   
def proportional_decay_schedule(initial_lr, proportion=0.5, perepochs=1):
    '''
    Wrapper function to create a LearningRateScheduler with geometric decay schedule.
    Every perepochs epochs, multiply learning rate by proportion.
    '''
    def schedule(epoch):
        lr = initial_lr * proportion ** (epoch//perepochs)
        print('learning rate: %f' %(lr))
        return lr

    return LearningRateScheduler(schedule)


#%% LBP-CNN designed by Lei Li et al. 
def getCNN(x):

    def getWeights(j, nb_channels, filterNo):
        
        if(False):
            print("input channels: ")
            print(nb_channels)
            
#        weights = np.array([[0,0,0],[0,-1,0],[0,0,0]])
#        if(j==0):
#            weights[0][0] = 1
#        if(j==1):   
#            weights[0][1] = 1
#        if(j==2):
#            weights[0][2] = 1
#        if(j==3): 
#            weights[1][0] = 1
#        if(j==4):
#            weights[1][2] = 1
#        if(j==5):
#            weights[2][0] = 1
#        if(j==6):
#            weights[2][1] = 1
#        if(j==7):    
#            weights[2][2] = 1  
        
#        weightTensor = np.zeros((nb_channels,3,3))
#        weightTensor[filterNo,] = weights
        weightTensor = np.zeros((3,3,nb_channels,1))
        row=j//3
        col=j-3*row
        weightTensor[row][col][filterNo][0]=1
        weightTensor[1][1][filterNo][0]=-1
#        print("weigthTensor:" )
#        print(weightTensor)
#        print("_____________________")
#        weightTensor=weightTensor.reshape(3,3,nb_channels,1) 
        bias=np.zeros(1)   # one filter  
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
    initial_channels = 32 #32 default
    output_channels = 64 #64 default
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
    #x = layers.Dropout(0.5)(x) ##DROPOUT ADDED TO ADDRESS OVERFITTING
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
    sgd = optimizers.sgd(lr=0.001, decay=0.0005, momentum=0.9)
    model.compile(loss = keras.losses.binary_crossentropy, metrics = ['accuracy'], optimizer = sgd)
    #model.compile(loss = keras.losses.categorical_crossentropy ,metrics = ['accuracy'], optimizer = adam)
    return model    

def runLeiLi(x_tr, x_va, x_te, y_tr, y_va, y_te,
             num_epochs = 10, batch_size = 100, eval_model=0):
    
    # Create model 
    print("Creating model...")
    model=getLeiLi()
    
    if(eval_model == 1):
        print("evaluating model on existing weights.")
        results = model.predict(x_te)
        score = model.evaluate(x_te, y_te, verbose=0)
        print('Test accuracy:', score[1])
            
    # Fit model
    print("Fitting model...")
    epochs = num_epochs
    model, mHist = myFitModel(model,epochs,x_tr,y_tr,x_va,y_va, batch_size)

    # Evaluate on test data
    print("Evaluating model...")
    path_best = "weights-best.hdf5"
    model.load_weights(path_best)
    results = model.predict(x_te)
    score = model.evaluate(x_te, y_te, verbose=0)
    print('Test accuracy:', score[1])
    
    preds=[int(x>0.5) for x in results]

    return y_te, preds, mHist 
    
# -*- coding: utf-8 -*-
#%%
import numpy as np
import keras
from keras import layers
from keras import models
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras import optimizers


#%%

"""
Based on code from https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce

Clean and simple Keras implementation of network architectures described in:
    - (ResNet-50) [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf).
    - (ResNeXt-50 32x4d) [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf).
    
Python 3.
"""


#
# image dimensions
#
img_height = 80
img_width = 80
img_channels = 1
img_shape=(img_height, img_width, img_channels)

class juefei:
    def __init__(self, fastMode=False, nLayers=10, sparsity=0.9, intermed_channels=128, k=512):
        self.kSparsity=sparsity
        self.nLayers=nLayers
        if(fastMode):
            self.intermed_channels=8
        else:
            self.intermed_channels=intermed_channels
        if(fastMode):
            self.k=4
        else:
            self.k=k

    
    def add_common_layers(self, y, leaky=True):
        y = layers.BatchNormalization()(y)
        if(leaky):
            y = layers.LeakyReLU()(y)
        else:
            y = layers.ReLU()(y)

        return y


    #Choose weights for fixed binary filters using predefined sparsity level    
    def getLBCNNWeights(self, size, count, channels=1):
        bias=np.zeros(count)
        numElements=size[0]*size[1]*count*channels
        weight=np.zeros(numElements)
        index=np.random.choice(numElements, int(self.kSparsity*numElements))
        for i in index:
            weight[i]=np.random.binomial(1, 0.5)*2-1
        weight=weight.reshape(size[0],size[1],channels,count)
        return np.asarray([weight,bias])


    #Function to create LBCNN layer, similar to standard CNN but fixed binary filters
    def lb_convolution(self, y, nb_filters, nb_channels, _strides):
        kernel_size=3
        y = layers.ZeroPadding2D(padding=(kernel_size//2, kernel_size//2))(y)
        return layers.Conv2D(nb_filters, kernel_size=(kernel_size, kernel_size), 
                             strides=_strides, weights=self.getLBCNNWeights(
                                     (kernel_size, kernel_size), nb_filters, nb_channels), 
                             padding='valid', trainable=False)(y)

    #Function to define LBCNN block with residual connection, to emulate functionality
    #of resnet-binary-felix.lua:createModel.basicBlock
    def LBCNN_res_block(self, y, nb_filters, nb_channels, _strides=(1, 1), leaky=True):
        """
        Our network consists of a stack of LBCNN blocks, each with a residual connection.
        An LBCNN block is a 3x3 convolutional layer with nb_filters channels, with non-trainable weights,
        followed by a 1x1 convolutional layer with a single channel which creates a linear combination
        of the outputs of the filters in the previous layer.
        """
        shortcut = y

        y = layers.BatchNormalization()(y)
        y = self.lb_convolution(y, nb_filters, nb_channels, _strides=_strides)
        if(leaky):
            y = layers.LeakyReLU()(y)
        else:
            y = layers.ReLU()(y)


        y = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=(1, 1), activation='linear')(y)
        y = layers.add([shortcut, y])

        return y

    def build_juefei(self, x):
        '''Create model, attempting to emulate architecture
           of resnet-binary-felix.lua:createModel as closely as possible
        '''
    
        #Begin with (3x3) conv layer, creating intermediate_channels channels
        x = layers.Conv2D(self.intermed_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = self.add_common_layers(x, False)
            
        #Add LBCNN blocks, each with k channels in their binary filter layer, reducing back 
        #to intermed_channels in the linear combination layer
        for i in range(self.nLayers):
            x = self.LBCNN_res_block(x, self.k, self.intermed_channels, leaky=False)

        #Extra normalisation layer not found in Juefei, but for our data it appears
        #that the model fails without it.
        x = layers.BatchNormalization()(x)
    
        #Average pooling over 5x5 non-overlapping regions
        x = layers.AveragePooling2D(pool_size=(5,5), strides=(5,5), padding='valid')(x)
        #Flatten and feed into a dense layer
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.k)(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.5)(x)
        
        #Output layer
        x = layers.Dense(1, activation='sigmoid')(x)
    
        return x

#%%    
    def build_modified_juefei(self, x):
        '''Build a modified version of juefei-xu, to get best performance with
        our dataset '''
    
        #Begin with (3x3) conv layer, to create intermediate_channels channels
        x = layers.Conv2D(self.intermed_channels, kernel_size=(3, 3), 
                          strides=(1, 1), padding='same')(x)
        x = self.add_common_layers(x, False)
            
        #Add LBCNN blocks, each with k channels in their binary filter layer, reducing back 
        #to intermed_channels in the linear combination layer
        for i in range(self.nLayers):
            x = self.LBCNN_res_block(x, self.k, self.intermed_channels, leaky=False)
    
        x = layers.BatchNormalization()(x)
        
        #Average pooling over 5x5 non-overlapping regions
        x = layers.AveragePooling2D(pool_size=(5,5), strides=(5,5), padding='valid')(x)
        
        #Additional convolutional layers
#        x = layers.Conv2D(self.intermed_channels, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x)
#        x = self.add_common_layers(x, False)
        x = layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x)
        x = self.add_common_layers(x, False)
        #Flatten and feed into a dense layer
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.k)(x) #TRIED: reduce to k/2 since 17 million parameters is a lot
        x = layers.ReLU()(x)
        x = layers.Dropout(0.5)(x)
        
        #Output layer
        x = layers.Dense(1, activation='sigmoid')(x)
    
        return x

            
#%%   

    def getModel(self):
        image_tensor = layers.Input(shape=img_shape)
        network_output = self.build_modified_juefei(x=image_tensor)
          
        model = models.Model(inputs=[image_tensor], outputs=[network_output])
        print(model.summary())
        
        adam = optimizers.Adam()
        model.compile(loss = keras.losses.binary_crossentropy, metrics = ['accuracy'], optimizer = adam)
        return model

#%%
    def step_decay_schedule(self, initial_lr, nr_training, batch_size):
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
    
#%%
    
    def fitModel(self, cNN, epochs, batch_size, x_tr, y_tr, x_va, y_va):
        path = "weights-best-2.hdf5"
        lr_sched = self.step_decay_schedule( 0.001, len(x_tr), batch_size)
        model_checkpoint = ModelCheckpoint(filepath = path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=1)
        early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None)
        mHist=cNN.fit(x_tr, y_tr, batch_size=batch_size, epochs = epochs, validation_data = (x_va, y_va),callbacks = [lr_sched, model_checkpoint, early_stopping])
        cNN.load_weights(path)
        return cNN, mHist
    
#%%
    def classify(self, x_tr, x_va, x_te, y_tr, y_va, y_te, num_epochs=50, batch_size=20, eval_model=0):
        # Create model 
        print("Creating model...")
        model=self.getModel()
        
        if(eval_model == 1):
            print("evaluating model on existing weights.")
            path_best = "weights_01.hdf5"
            model.load_weights(path_best)
            results = model.predict(x_te)
            score = model.evaluate(x_te, y_te, verbose=0)
            print('Test accuracy:', score[1])
            return y_te, results
    
        # Fit model
        print("Fitting model...")
        epochs=num_epochs
        model, mHist = self.fitModel(model,epochs,batch_size,x_tr,y_tr,x_va,y_va)
    
        # Evaluate on test data
        print("Evaluating model...")
        results = model.predict(x_te)
        score = model.evaluate(x_te, y_te, verbose=0)
        print('Test accuracy:', score[1])

        preds=[int(x>0.5) for x in results]
        return y_te, preds, mHist

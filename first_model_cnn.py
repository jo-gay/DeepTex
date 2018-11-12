# -*- coding: utf-8 -*-
#%%
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
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.engine.topology import Layer
filedir = './data/'

#%% MAIN SCRIPT
def main():
     res = runImageClassification()
     preds=np.argmax(res[1], axis=1)
     truth=np.argmax(res[0], axis=1)
     from sklearn.metrics import confusion_matrix
     print(confusion_matrix(truth, preds, sample_weight=None))

#%%
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

class LBCNN(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=False)
        super(LBCNN, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    
def getLBCNNWeights(size, count, kSparsity):
    bias=np.zeros(count)
    numElements=size[0]*size[1]*count
    weight=np.zeros(numElements)
    index=np.random.choice(numElements, int(kSparsity*numElements))
    for i in index:
        weight[i]=np.random.binomial(1, 0.5)*2-1
    weight=weight.reshape(size[0],size[1],1,count)
    return np.asarray([weight,bias])

    
def myGetModel():
    num_classes = 2
    sparsity=0.5
    kernel_size=(3,3)
    k=20
    cNN = Sequential()
    cNN.add(ZeroPadding2D((1,1)))
    cNN.add(Conv2D(k, kernel_size=kernel_size, activation='sigmoid', input_shape=(1,80,80), 
                   padding ="same",trainable = False, weights=np.array(getLBCNNWeights(kernel_size, k, sparsity))))
    cNN.add(Conv2D(1, kernel_size=(1, 1), activation='linear'))
    
    
    for x in range(0,5):
        cNN.add(ZeroPadding2D((1,1)))
        cNN.add(Conv2D(k, kernel_size=kernel_size, activation='sigmoid', trainable = False
                       , weights=np.asarray(getLBCNNWeights(kernel_size, k, sparsity))))
        cNN.add(Conv2D(1, kernel_size=(1, 1), activation='linear'))
    
    for x in range(0,5):
        cNN.add(ZeroPadding2D((1,1)))
        cNN.add(Conv2D(k, kernel_size=kernel_size, activation='sigmoid', trainable = False
                       , weights=np.asarray(getLBCNNWeights(kernel_size, k, sparsity))))
        cNN.add(Conv2D(1, kernel_size=(1, 1), activation='linear'))
        
    for x in range(0,5):
        cNN.add(ZeroPadding2D((1,1)))
        cNN.add(Conv2D(k, kernel_size=kernel_size, activation='sigmoid', trainable = False
                       , weights=np.asarray(getLBCNNWeights(kernel_size, k, sparsity))))
        cNN.add(Conv2D(1, kernel_size=(1, 1), activation='linear'))

    cNN.add(MaxPooling2D(pool_size=(2, 2)))
    
    cNN.add(Flatten())
    cNN.add(Dense(1024, activation='relu'))
    cNN.add(Dense(num_classes, activation='softmax'))
    
    #sgd = optimizers.SGD(lr=0.1, decay=1e-6)
    #adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999)
    #adam = optimizers.adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    cNN.compile(loss = keras.losses.categorical_crossentropy ,metrics = ['accuracy'], optimizer = 'adam')
    return cNN
    
def myFitModel(cNN,epochs, x_tr,y_tr, x_va,y_va):
    path = "weights-best.hdf5"
    model_checkpoint = ModelCheckpoint(filepath = path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=1)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None)
    cNN.fit(x_tr, y_tr, epochs = epochs, batch_size = 100, validation_data = (x_va, y_va),callbacks = [model_checkpoint, early_stopping])
    cNN.load_weights(path)
    return cNN
    
def runImageClassification(getModel=None,fitModel=None,seed=7):
    # Fetch data. You may need to be connected to the internet the first time this is done.
    # After the first time, it should be available in your system. On the off chance this
    # is not the case on your system and you find yourself repeatedly downloading the data, 
    # you should change this code so you can load the data once and pass it to this function. 
    print("Preparing data...")
    class_names=['healthy','cancer']
        
    filedir = './data/'
    #fold1tr=[[3,4,5,6],[37,38,12]]
    fold2tr=[[3,4,5,6],[36,37,38]]
    #fold3tr=[[3,4,5,6],[12,36]]
    #fold1te=[[7,8],[36]]
    fold2te=[[7,8],[12]]
    #fold3te=[[7,8],[37,38]]

    # Extracting data from files
    data_fold2_tr=[]
    labels_fold2_tr=[]
    
    data_fold2_te=[]
    labels_fold2_te=[]
    
    # training and validation data
    #%%
    for g in fold2tr[0]:
        filename = filedir+'glass'+str(g)+'.hdf5'
        f = h5py.File(filename, 'r')
        
        a_group_key = list(f.keys())[0]
        d=list(f[a_group_key])
        data_fold2_tr += d
        labels_fold2_tr += list(np.zeros(len(d)))
            #%%
    # we shuffle the data in preparation for removing healthy samples, in order to get 50/50 healthy/cancer cells
    seed=2
    random.seed(seed)
    random.shuffle(data_fold2_tr)
    nHealthy = len(data_fold2_tr)
#%%    
    for g in fold2tr[1]:
        filename = filedir+'glass'+str(g)+'.hdf5'
        f = h5py.File(filename, 'r')
        
        a_group_key = list(f.keys())[0]
        d=list(f[a_group_key])
        data_fold2_tr += d
        labels_fold2_tr += list(np.ones(len(d)))
        
        # test data
    for g in fold2te[0]:
        filename = filedir+'glass'+str(g)+'.hdf5'
        f = h5py.File(filename, 'r')
        
        a_group_key = list(f.keys())[0]
        d=list(f[a_group_key])
        data_fold2_te += d
        labels_fold2_te += list(np.zeros(len(d)))
            
    for g in fold2te[1]:
        filename = filedir+'glass'+str(g)+'.hdf5'
        f = h5py.File(filename, 'r')
        
        a_group_key = list(f.keys())[0]
        d=list(f[a_group_key])
        data_fold2_te += d
        labels_fold2_te += list(np.ones(len(d)))
       
    #%%    
    # Shuffling data and partitioning into training, testing, and validation sets    
    nSamples=len(labels_fold2_tr)
    tr_perc=.80
    va_perc=.20
    
    #removing healthy cells so that we have 50/50 healthy/sick cells
    data_fold2_tr = data_fold2_tr[(2*nHealthy - nSamples):]
    labels_fold2_tr = labels_fold2_tr[(2*nHealthy - nSamples):]
        
    seed = 1
    random.seed(seed)
    random.shuffle(labels_fold2_tr)
    random.seed(seed)
    random.shuffle(data_fold2_tr)
    
    # we update the number of samples after removing healty cells
    nSamples = len(data_fold2_tr)

    tr=round(nSamples*tr_perc)
        
    x_tr = np.asarray(data_fold2_tr[0:tr])
    x_va = np.asarray(data_fold2_tr[tr:nSamples])
    x_te = np.asarray(data_fold2_te)
    y_tr = np.asarray(labels_fold2_tr[0:tr])
    y_va = np.asarray(labels_fold2_tr[tr:nSamples])
    y_te = np.asarray(labels_fold2_te)

    # converting vectors to numpy arrays
    y_tr = np_utils.to_categorical(y_tr, 2)
    y_va = np_utils.to_categorical(y_va, 2)
    y_te = np_utils.to_categorical(y_te, 2)
    
    #Normalise data by calculating the mean and standard deviation for the 
    #training data, and transform the training and test data using this
    x_tr = x_tr.astype('float32')
    x_va = x_va.astype('float32')
    x_te = x_te.astype('float32')
    
#%%
    for x in range (0,len(x_tr)):
        x_tr[x] = (x_tr[x]-np.mean(x_tr[x]))/ np.std(x_tr[x])
        
    for x in range (0,len(x_va)):
        x_va[x] = (x_va[x]-np.mean(x_va[x]))/ np.std(x_va[x])
        
    for x in range (0,len(x_te)):
        x_te[x] = (x_te[x]-np.mean(x_te[x]))/ np.std(x_te[x])
#    x_tr /= 255
#    x_va /= 255
#    x_te /= 255
        
    #%%
    # Create model 
    print("Creating model...")
    model=myGetModel()
    
    # Fit model
    print("Fitting model...")
    epochs = 15
    model=myFitModel(model,epochs,x_tr,y_tr, x_va,y_va)
    
    results = model.predict(x_te)
    
    # Evaluate on test data
    print("Evaluating model...")
    score = model.evaluate(x_te, y_te, verbose=0)
    print('Test accuracy:', score[1])

    return((y_te, results))

if __name__ == "__main__":
    main()




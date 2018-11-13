# -*- coding: utf-8 -*-
#%%
import numpy as np
import matplotlib.pyplot as plt
import keras
import h5py, sys
import random
import tensorflow as tf
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.engine.topology import Layer
from sklearn.metrics import confusion_matrix

filedir = './data/'

#%% MAIN SCRIPT
def main(GPU_NUM):
    
    sess = doInitTasks(GPU_NUM)
    with sess.as_default():
        res = runImageClassification()

    preds=np.argmax(res[1], axis=1)
    truth=np.argmax(res[0], axis=1)
    
    print(confusion_matrix(truth, preds, sample_weight=None))

#%%
    
def doInitTasks(GPU_NUM):
    
    #Use the command line parameter GPU_NUM to configure which GPU(s) to use
    #Assumes that the machine has totalGPU GPUs (hard-coded as 2) and if a
    #number from 0 to totalGPUs-1 is entered, uses that GPU. 
    #If no command line parameter is supplied, use CPU.
    #Otherwise (e.g. commandline arg = totalGPUs), use all GPUs.
    
    totalGPUs=2
    config = None

    if(GPU_NUM is None):
        config = tf.ConfigProto(device_count = {'GPU':0})
    elif(GPU_NUM < totalGPUs):
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(GPU_NUM)
    else:
        config = tf.ConfigProto()

    sess = tf.Session(config=config)
    K.set_session(sess)
    
    # Here we just make sure the image format is as desired. This will make the feature (x)
    # data - i.e. the RGB pixel values - for each image have the shape 3x32x32.
#    if K.backend()=='tensorflow':
#        K.set_image_dim_ordering("th")
    
    #In this case our data is ordered theano-style, channels first, but we are going to change it
    #to tensorflow style to work around a problem with the batch normalisation 'axis' parameter 
    #when running on GPU. Therefore if theano backend is in use, we need to tell it
    #to use tensorflow order.
    if K.backend()=='theano':
        K.set_image_dim_ordering("tf")
        
    return sess


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

    
def getLBCNNWeights(size, count, kSparsity, channels=1):
    bias=np.zeros(count)
    numElements=size[0]*size[1]*count*channels
    weight=np.zeros(numElements)
    index=np.random.choice(numElements, int(kSparsity*numElements))
    for i in index:
        weight[i]=np.random.binomial(1, 0.5)*2-1
    weight=weight.reshape(size[0],size[1],channels,count)
    return np.asarray([weight,bias])


#%%

"""
Code from https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce

Clean and simple Keras implementation of network architectures described in:
    - (ResNet-50) [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf).
    - (ResNeXt-50 32x4d) [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf).
    
Python 3.
"""

from keras import layers
from keras import models


#
# image dimensions
#

img_height = 80
img_width = 80
img_channels = 1
img_shape=(img_height, img_width, img_channels)

#
# network params
#

cardinality = 1


def residual_network(x):
    """
    ResNeXt by default. For ResNet set `cardinality` = 1 above.
    
    """
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
#        y = layers.LeakyReLU()(y)

        return y
    
    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # modify to use LBCNN instead of normal
        y = lb_convolution(y, 1, nb_channels_out, _strides=_strides)
        #y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='linear')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y

    #Added function to use LBCNN layer instead of standard CNN
    def lb_convolution(y, nb_filters, nb_channels, _strides, sparsity=0.5):
        kernel_size=(3,3)
        return layers.Conv2D(nb_filters, kernel_size=kernel_size, strides=_strides, 
                              weights=getLBCNNWeights(kernel_size, nb_filters, sparsity, nb_channels), 
                              padding='same', trainable=False)(y)

    #Added function to define LBCNN block with residual connection attempting to emulate functionality
    #of resnet-binary-felix.lua:createModel.basicBlock
    def LBCNN_res_block(y, nb_filters, nb_channels, _strides=(1, 1)):
        """
        Our network consists of a stack of LBCNN blocks, each with a residual connection.
        An LBCNN block is a 3x3 convolutional layer with nb_filters channels, with non-trainable weights,
        followed by a 1x1 convolutional layer with a single channel which creates a linear combination
        of the outputs of the filters in the previous layer.
        """
        shortcut = y

        y = layers.BatchNormalization()(y)
        y = lb_convolution(y, nb_filters, nb_channels, _strides=_strides)
        y = layers.ReLU()(y)

        y = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=(1, 1), activation='linear')(y)
        y = layers.add([shortcut, y])

        return y

    #Create model, attempting to emulate functionality
    #of resnet-binary-felix.lua:createModel
    intermed_channels=128

    # conv1
    x = layers.Conv2D(intermed_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = add_common_layers(x)
        
    k=512
    for i in range(10):
        x = LBCNN_res_block(x, k, intermed_channels)

    x = layers.AveragePooling2D(pool_size=(5,5), strides=(5,5), padding="valid")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(k)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(2)(x)

    return x


def getResNetModel():
    image_tensor = layers.Input(shape=img_shape)
    network_output = residual_network(image_tensor)
      
    model = models.Model(inputs=[image_tensor], outputs=[network_output])
    print(model.summary())
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss = keras.losses.categorical_crossentropy ,metrics = ['accuracy'], optimizer = adam)
    return model



#%%
    
def myGetModel():
    num_classes = 2
    sparsity=0.5
    kernel_size=(3,3)
    k=20
#    cNN = keras.layers.Input(shape=(1,80,80))
    cNN = Sequential()
#    cNN.add(ZeroPadding2D((1,1)))
    cNN.add(Conv2D(k, kernel_size=kernel_size, activation='relu', input_shape=img_shape, 
                   padding ="same",trainable = False, weights=np.array(getLBCNNWeights(kernel_size, k, sparsity))))
    cNN.add(BatchNormalization())
    cNN.add(Conv2D(1, kernel_size=(1, 1), activation='linear'))
    cNN.add(BatchNormalization())
    
    for x in range(0,5):
#        cNN.add(ZeroPadding2D((1,1)))
        cNN.add(Conv2D(k, kernel_size=kernel_size, activation='relu', trainable = False, padding="same"
                       , weights=np.asarray(getLBCNNWeights(kernel_size, k, sparsity))))
        cNN.add(BatchNormalization())
        cNN.add(Conv2D(1, kernel_size=(1, 1), activation='linear'))
        cNN.add(BatchNormalization())
    
    for x in range(0,5):
#        cNN.add(ZeroPadding2D((1,1)))
        cNN.add(Conv2D(k, kernel_size=kernel_size, activation='relu', trainable = False, padding="same"
                       , weights=np.asarray(getLBCNNWeights(kernel_size, k, sparsity))))
        cNN.add(BatchNormalization())
        cNN.add(Conv2D(1, kernel_size=(1, 1), activation='linear'))
        cNN.add(BatchNormalization())
        
    for x in range(0,5):
#        cNN.add(ZeroPadding2D((1,1)))
        cNN.add(Conv2D(k, kernel_size=kernel_size, activation='relu', trainable = False, padding="same"
                       , weights=np.asarray(getLBCNNWeights(kernel_size, k, sparsity))))
        cNN.add(BatchNormalization())
        cNN.add(Conv2D(1, kernel_size=(1, 1), activation='linear'))
        cNN.add(BatchNormalization())

    cNN.add(MaxPooling2D(pool_size=(2, 2)))
    
    cNN.add(Flatten())
    cNN.add(Dense(1024, activation='relu'))
    cNN.add(Dense(num_classes, activation='softmax'))
    
    #sgd = optimizers.SGD(lr=0.1, decay=1e-6)
#    adam = optimizers.Adam(lr=0.001)
    #adam = optimizers.adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    cNN.compile(loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'], optimizer = 'adam')
    return cNN
    
def myFitModel(cNN,epochs, x_tr,y_tr, x_va,y_va):
    path = "weights-best.hdf5"
    model_checkpoint = ModelCheckpoint(filepath = path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=1)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None)
    cNN.fit(x_tr, y_tr, batch_size=25, epochs = epochs, validation_data = (x_va, y_va),callbacks = [model_checkpoint, early_stopping])
    cNN.load_weights(path)
    return cNN
   
#%%
def runImageClassification(getModel=None,fitModel=None,seed=7):
    # Fetch data. NB SEED NOT USED
    print("Preparing data...")
    class_names=['healthy','cancer']
        
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
    # Healthy cells:
    for g in fold2tr[0]:
        filename = filedir+'glass'+str(g)+'.hdf5'
        f = h5py.File(filename, 'r')
        
        a_group_key = list(f.keys())[0]
        d=list(f[a_group_key])
        data_fold2_tr += d
        labels_fold2_tr += list(np.zeros(len(d)))

    nHealthy_tr = len(data_fold2_tr)
#%%    
    # cancer cells:
    for g in fold2tr[1]:
        filename = filedir+'glass'+str(g)+'.hdf5'
        f = h5py.File(filename, 'r')
        
        a_group_key = list(f.keys())[0]
        d=list(f[a_group_key])
        data_fold2_tr += d
        labels_fold2_tr += list(np.ones(len(d)))
        
    # test data healthy cells
    for g in fold2te[0]:
        filename = filedir+'glass'+str(g)+'.hdf5'
        f = h5py.File(filename, 'r')
        
        a_group_key = list(f.keys())[0]
        d=list(f[a_group_key])
        data_fold2_te += d
        labels_fold2_te += list(np.zeros(len(d)))
        
        
    # test data cancer cells         
    for g in fold2te[1]:
        filename = filedir+'glass'+str(g)+'.hdf5'
        f = h5py.File(filename, 'r')
        
        a_group_key = list(f.keys())[0]
        d=list(f[a_group_key])
        data_fold2_te += d
        labels_fold2_te += list(np.ones(len(d)))
       
    #%%    
    # Converting training and test data lists to numpy arrays   
    data_fold2_tr =  np.asarray(data_fold2_tr)
    data_fold2_te = np.asarray(data_fold2_te)
    # rotation and mirroring of cancer images
    cancer_data = data_fold2_tr[nHealthy_tr:,]
    cancer_hz_mirror = np.flip(cancer_data, axis =2)
    cancer_ver_mirror = np.flip(cancer_data, axis =3)
    cancer_2x_mirror = np.flip(cancer_ver_mirror, axis = 2)
    
    cancer_data = np.append(cancer_data, cancer_hz_mirror, axis = 0)
    cancer_data = np.append(cancer_data, cancer_ver_mirror, axis = 0)
    cancer_data = np.append(cancer_data, cancer_2x_mirror, axis = 0)
    cancer_data = np.append(cancer_data, np.rot90(cancer_data,k=1,axes=(2,3)), axis = 0)
    
    # rotation and mirroring of healthy images
    healthy_data = data_fold2_tr[0:(len(data_fold2_tr) -nHealthy_tr),]
    healthy_hz_mirror = np.flip(healthy_data, axis =2)
    healthy_ver_mirror = np.flip(healthy_data, axis =3)
    healthy_2x_mirror = np.flip(healthy_ver_mirror, axis = 2)
    
    healthy_data = np.append(healthy_data, healthy_hz_mirror, axis = 0)
    healthy_data = np.append(healthy_data, healthy_ver_mirror, axis = 0)
    healthy_data = np.append(healthy_data, healthy_2x_mirror, axis = 0)
    healthy_data = np.append(healthy_data, np.rot90(healthy_data,k=1,axes=(2,3)), axis = 0)
    
    sick_labels = np.ones(len(cancer_data))
    healthy_labels = np.zeros(len(healthy_data))
    fold2_aug_tr = np.append(healthy_data, cancer_data, axis = 0)
    fold2_aug_tr_labels = np.append(healthy_labels,sick_labels)
    
    # we shuffle the training data and labels
    random.seed(1)
    np.random.shuffle(fold2_aug_tr)
    random.seed(1)
    np.random.shuffle(fold2_aug_tr_labels)

    tr_perc=.80
    
    # we update the number of samples after removing healty cells
    nSamples=len(fold2_aug_tr)
    tr=round(nSamples*tr_perc)
    
    x_tr = fold2_aug_tr[0:tr,]
    x_va = fold2_aug_tr[tr:,]
    x_te = np.asarray(data_fold2_te)
    
    y_tr = fold2_aug_tr_labels[0:tr]
    y_va = fold2_aug_tr_labels[tr:]
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
    
    # change order to channels LAST (tensorflow default)
    x_te = np.moveaxis(x_te,1,3)
    x_va = np.moveaxis(x_va,1,3)
    x_tr = np.moveaxis(x_tr,1,3)
    
#%%
    for x in range (0,len(x_tr)):
        x_tr[x] = (x_tr[x]-np.mean(x_tr[x]))/ np.std(x_tr[x])
        
    for x in range (0,len(x_va)):
        x_va[x] = (x_va[x]-np.mean(x_va[x]))/ np.std(x_va[x])
        
    for x in range (0,len(x_te)):
        x_te[x] = (x_te[x]-np.mean(x_te[x]))/ np.std(x_te[x])
        
        
    #%%
    # Create model 
    print("Creating model...")
#    model=myGetModel()
    model=getResNetModel()
    
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
    if len(sys.argv) > 1:
        GPU_NUM = int(sys.argv[1])
        print("Using GPU %d"%GPU_NUM)
    else:
        print("Using CPU only")
        GPU_NUM=None

    main(GPU_NUM)
    




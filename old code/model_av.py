 #Trains the selected model on all three folds (or just fold 2 if fastMode = True)
 #Complies the the data into one confusion matrix and prints the result  
import mnist_test.py
import numpy as np
import keras
import tensorflow as tf
from keras.utils import np_utils
from keras import backend as K

def main(GPU_NUM, fastMode=False):
        
        confusionMat=[[0,0],[0,0]]
        overallResults={'Precision':0, 'Accuracy':0, 'Recall':0, 'F1-Score':0, 'NumTestPts':0, 'Confusion':[]}
        trainfolds={
                1:{'healthy': [3,4,5,6], 'cancer':[37,38,12]},
                2:{'healthy': [3,4,5,6], 'cancer':[36,37,38]},
                3:{'healthy': [3,4,5,6], 'cancer':[12,36]},
                }
        testfolds={
                1:{'healthy': [7,8], 'cancer':[36]},
                2:{'healthy': [7,8], 'cancer':[12]},
                3:{'healthy': [7,8], 'cancer':[37,38]},
                }
        folds=trainfolds.keys()
        initial_lr = 0.1
        GPU_NUM = 0
    
    
    sess = doInitTasks(GPU_NUM)
    if(fastMode):
        folds=[2,]
        
    with sess.as_default():
        for f in folds:
            print("\nStarting training with fold number %d of %d"%(f, len(folds)))
            y_te, res = mnist_test(foldNr=f, fastMode=fastMode)
            preds=[int(x>0.5) for x in res]
            confusionMat+=confusion_matrix(y_te, preds)
    
        #Now we have the total confusion matrix for all folds. Calculate measures from that.
        overallResults['NumTestPts']=sum(sum(confusionMat))
        overallResults['Accuracy'] = (confusionMat[0][0] + confusionMat[1][1])/overallResults['NumTestPts']
        overallResults['Precision'] = confusionMat[0][0]/(confusionMat[0][0]+confusionMat[0][1])
        overallResults['Recall'] = confusionMat[0][0]/(confusionMat[0][0]+confusionMat[1][0])
        overallResults['F1-Score']=2*overallResults['Precision']*overallResults['Recall'] / (overallResults['Precision'] + overallResults['Recall'])
        overallResults['Confusion']=confusionMat
        
        print("\n******** Combined results over %d folds ************"%len(folds))
        for k, v in overallResults.items():
            if(k == 'NumTestPts'):
                print('%12s: %d'%(k, v))
            elif(k != 'Confusion'):
                print('%12s: %2.4f'%(k, v))
    
        print('Confusion Matrix:')
        for row in confusionMat:
            print('\t '.join(map(str,row)))
        print("********     End of combined results    ************")
        with open('savedResults.txt', 'w') as outfile:
            outfile.write(str(overallResults)+'\n')
            
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

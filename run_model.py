import numpy as np
import h5py, datetime, random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import argparse

#import models that can be run using this script
from juefei import juefei
from marcos import marcos
from lei_li import runLeiLi
from lbphist import lbphist

availModels={0: 'Juefei-Xu', 1: 'Marcos', 2: 'LiLei', 3: 'LBPhist'}

'''DEFINE GPU SETUP - this is the total number of gpus in system.
Specify which gpu to use on command line (or None)
'''
totalGPUs=2

'''DEFINE PARAMETERS RELATING TO DATA'''
#Data is stored in <filedir><fname><k>.hdf5 for k in trainfolds.keys and testfolds.keys
filedir = './data/'
fname='glass'
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

def loadData(foldNr=2, valPercent=0.2, augment=False, seed=None, changeOrder=True, normalize=True, fastMode=False):
    '''For a given fold, load the training and testing data. Limit the number
    of healthy cells in the training data so that there are the same number of
    healthy cells as cancer cells - cells are chosen at random and seed can be
    specified.
    Choose a validation set as a random subset of valPercent of the training data
    Augment training data (not test data) with mirrored and rotated versions, if augment
    Change order from channels first to channels last if changeOrder
    Normalize each image (mean 0, std 1) if normalize. 
    '''
    if(not seed is None):
        random.seed(seed)

    trainGlasses=trainfolds[foldNr]
    testGlasses=testfolds[foldNr]
    
    healthyData=[]
    cancerData=[]
    for g in trainGlasses['cancer']:
        filename = filedir+fname+str(g)+'.hdf5'
        f = h5py.File(filename, 'r')
        
        a_group_key = list(f.keys())[0]
        d=list(f[a_group_key])
        cancerData += d
        #print("Glass %d, tumour, %d images"%(g,len(d)))
    for g in trainGlasses['healthy']:
        filename = filedir+'glass'+str(g)+'.hdf5'
        f = h5py.File(filename, 'r')
        
        a_group_key = list(f.keys())[0]
        d=list(f[a_group_key])
        healthyData += d
        #print("Glass %d, healthy, %d images"%(g,len(d)))
    
    if(len(cancerData)<len(healthyData)):
        healthyDataIdx = np.random.choice(len(healthyData), size=len(cancerData), replace=False)
        healthyData=np.asarray(healthyData)[healthyDataIdx]
    elif(len(healthyData)<len(cancerData)):
        cancerDataIdx = np.random.choice(len(cancerData), size=len(healthyData), replace=False)
        cancerData=np.asarray(cancerData)[cancerDataIdx]
    
    trainData=np.append(healthyData, cancerData, axis=0).astype(np.float32)
    trainLabels=np.append(np.zeros(len(healthyData)), np.ones(len(cancerData)))

    #Split the training data into training and validation BEFORE augmentation so
    #that the validation set contains cells that have not been seen before.
    x_tr, x_va, y_tr, y_va = train_test_split(trainData, trainLabels, test_size=valPercent, random_state=seed)

    if(augment):
        '''Augment the training data and validation data with mirrored and rotated versions. 
        '''
        h_mirror = np.flip(x_tr, axis=2)
        v_mirror = np.flip(x_tr, axis=3)
        b_mirror = np.flip(v_mirror, axis=2)
        
        x_tr = np.append(x_tr, h_mirror, axis = 0)
        x_tr = np.append(x_tr, v_mirror, axis = 0)
        x_tr = np.append(x_tr, b_mirror, axis = 0)
        
        x_tr = np.append(x_tr, np.rot90(x_tr,k=1,axes=(2,3)), axis = 0)
        y_tr = np.tile(y_tr, 8)

        h_mirror = np.flip(x_va, axis=2)
        v_mirror = np.flip(x_va, axis=3)
        b_mirror = np.flip(v_mirror, axis=2)
        
        x_va = np.append(x_va, h_mirror, axis = 0)
        x_va = np.append(x_va, v_mirror, axis = 0)
        x_va = np.append(x_va, b_mirror, axis = 0)
        
        x_va = np.append(x_va, np.rot90(x_va,k=1,axes=(2,3)), axis = 0)
        y_va = np.tile(y_va, 8)


    healthyData=[]
    cancerData=[]
    for g in testGlasses['cancer']:
        filename = filedir+'glass'+str(g)+'.hdf5'
        f = h5py.File(filename, 'r')
        
        a_group_key = list(f.keys())[0]
        d=list(f[a_group_key])
        cancerData += d
        #print("Glass %d, tumour, %d images"%(g,len(d)))

    for g in testGlasses['healthy']:
        filename = filedir+'glass'+str(g)+'.hdf5'
        f = h5py.File(filename, 'r')
        
        a_group_key = list(f.keys())[0]
        d=list(f[a_group_key])
        healthyData += d
        #print("Glass %d, healthy, %d images"%(g,len(d)))
        
    x_te=np.append(healthyData, cancerData, axis=0).astype(np.float32)
    y_te=np.append(np.zeros(len(healthyData)), np.ones(len(cancerData)))

    if(normalize):
        # standardize so that each image has mean 0 and std 1
        x_tr = np.asarray([(x_tr[x]-np.mean(x_tr[x]))/np.std(x_tr[x]) for x in range(len(x_tr))])
        x_va = np.asarray([(x_va[x]-np.mean(x_va[x]))/np.std(x_va[x]) for x in range(len(x_va))])
        x_te = np.asarray([(x_te[x]-np.mean(x_te[x]))/np.std(x_te[x]) for x in range(len(x_te))])
        
    if(changeOrder):
        # change order to channels LAST (tensorflow default)
        x_tr = np.moveaxis(x_tr,1,3)
        x_va = np.moveaxis(x_va,1,3)
        x_te = np.moveaxis(x_te,1,3)
        
    if(fastMode):
        subsample=np.random.choice(len(x_tr), size=min(len(x_tr), 1000), replace=False)
        x_tr = x_tr[subsample]
        y_tr = y_tr[subsample]
        subsample=np.random.choice(len(x_va), size=min(len(x_va), 200), replace=False)
        x_va = x_va[subsample]
        y_va = y_va[subsample]
        subsample=np.random.choice(len(x_te), size=min(len(x_te), 200), replace=False)
        x_te = x_te[subsample]
        y_te = y_te[subsample]

            
    return x_tr, x_va, x_te, y_tr, y_va, y_te
    

def initKerasSession(GPU_NUM, totalGPUs=2, imageOrdering='tf'):
    '''
    Get a Keras session, configuring which GPU(s) to use.
    Assumes that the machine has totalGPU GPUs (default 2) and if a
    number from 0 to totalGPUs-1 is entered, uses that GPU. 
    If GPU_NUM is None, use CPU only.
    Otherwise (e.g. commandline arg = totalGPUs), use all GPUs.
    Also set image ordering in Keras to match data.
    '''
    import tensorflow as tf
    from keras import backend as K

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
    
    #Make sure the keras backend image ordering is the same as the image ordering 
    #for our data. If data is ordered theano-style, i.e. channels first, and backend
    #is tensorflow, tell Keras that we are using theano ordering, and vice versa.
    if imageOrdering=='tf' and K.backend()=='theano':
        K.set_image_dim_ordering('tf')
    elif imageOrdering=='th' and K.backend()=='tensorflow':
        K.set_image_dim_ordering('th')
        
    return sess


def appendFoldResults(f, resultsFile, cm, mHist=None, v_acc=None, t_acc=None):
    ''' Append results for fold f to resultsFile, including confusion matrix cm
        and any of the following if available: 
            model History (from Keras) mHist, 
            validation accuracy v_acc, 
            test accuracy t_acc
    '''
    with open(resultsFile, 'a') as outfile:
        if mHist is not None:
            outfile.write('Training history for fold %d:\n'%f)
            for k,v in mHist.history.items():
                outfile.write('%16s: %s\n'%(k, str(v)))
        outfile.write('Confusion matrix for fold %d:\n'%f)
        for row in cm:
            outfile.write('\t'+'\t '.join(map(str,row)))
            outfile.write('\n')
        if v_acc is not None:
            outfile.write('Validation accuracy for fold %d: %2.5f\n'%(f, v_acc))
        if t_acc is None:
            t_acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
        outfile.write('Test accuracy for fold %d: %2.5f\n'%(f, t_acc))
            


def readCommandLine(totalGPUs=2):
    ''' Read command line parameters. Defaults are also set here
    '''
    dnow=datetime.datetime.now()
    resultsFile = 'savedResults%02d%02d.txt'%(dnow.hour, dnow.minute)

    parser = argparse.ArgumentParser(description='Run one of several available models on specified data')
    parser.add_argument('-g', '--gpu', dest='GPU_NUM', type=int, default=None,
                   help='GPU number 0-%d to run on; Omit for CPU; %d for all'%(totalGPUs-1, totalGPUs))
    parser.add_argument('--model','-m', type=int, default=0, dest='model_ver',
                   help='Model number: %s'%str(availModels))
    parser.add_argument('--fast', '-f', dest='fastMode', action='store_const',
                   const=True, default=False,
                   help='Run in fast mode with very limited training dataset, number of epochs, etc')
    parser.add_argument('--outfile','-o', type=str, default=resultsFile, dest='resultsFile',
                   help='Filename to write results to (will be overwritten)')
    parser.add_argument('--epochs','-e', type=int, default=90, dest='Nepochs',
                   help='Number of epochs to run for')
    parser.add_argument('--batchsz','-b', type=int, default=200, dest='batch_size',
                   help='Set batch size')

    args = parser.parse_args()
    
    #limit batch size depending on model
    if args.model_ver == 0 or args.GPU_NUM is None: #model 0 crashes with 50
        max_bs = 20
    elif args.model_ver == 1: #model 1 crashes with 70 (if 64 filters in each layer)
        max_bs = 200
    elif args.model_ver == 2: #model 2 crashes with 100(if filters increased)
        max_bs = 200 
    if(args.batch_size > max_bs):
        args.batch_size = max_bs
    
    print(args)
    return args


if __name__ == '__main__':
    ''' Main function reads command line arguments, iterates through folds
    and for each fold, trains and evaluates the chosen model.
    Results are output to screen and to a file.
    '''

    args=readCommandLine(totalGPUs)
    
    folds = list(trainfolds.keys())
    if(args.fastMode):
        folds=[2,]
        args.Nepochs=2

    valAcc=0
    confusionMat=[[0,0],[0,0]]
    overallResults={'Precision':0, 'Accuracy':0, 'Recall':0, 'F1-Score':0, 'ValidationAcc':0, 'NumTestPts':0, 'Confusion':[]}
    
    print('Running model %s %s using %d epochs, batch size %d, on %s'%
                  (availModels[args.model_ver], "in fastmode" if args.fastMode else "", 
                   args.Nepochs, args.batch_size, "GPU %d"%args.GPU_NUM if args.GPU_NUM is not None else "CPU"))
    print("\n Saving results to", args.resultsFile)
    with open(args.resultsFile, 'w') as outfile:
        outfile.write('Running model %s %s using %d epochs, batch size %d, on %s\n'%
                      (availModels[args.model_ver], "in fastmode" if args.fastMode else "", 
                       args.Nepochs, args.batch_size, "GPU %d"%args.GPU_NUM if args.GPU_NUM is not None else "CPU"))
        outfile.write('Start time: '+str(datetime.datetime.now())+'\n')

    for idx, f in enumerate(folds):
        print("\nStarting training with fold %d (%d of %d)"%(f, idx+1, len(folds)))
        
        print("Preprocessing data...")
        x_tr, x_va, x_te, y_tr, y_va, y_te = loadData(foldNr=f, valPercent=0.2, 
                                                  augment=True, seed=None, 
                                                  changeOrder=not(args.model_ver==1), 
                                                  normalize=True, 
                                                  fastMode=args.fastMode)
        #### Juefei ####
        if(args.model_ver == 0):
            classifier=juefei(args.fastMode, nLayers=20, sparsity=0.9)
            sess = initKerasSession(args.GPU_NUM)
            with sess.as_default():
                y_te, preds, mHist = classifier.classify(x_tr, x_va, x_te, y_tr, y_va, y_te, 
                                                   num_epochs=args.Nepochs, batch_size=args.batch_size)
                valAcc += max(mHist.history['val_acc'])
                for k,v in mHist.history.items():
                    print(k, v)
                appendFoldResults(f, args.resultsFile, confusion_matrix(y_te, preds), mHist=mHist)

        #### Marcos ####
        elif(args.model_ver == 1):
            classifier=marcos(args.GPU_NUM, args.fastMode)
            y_te, preds, mHist = classifier.classify(x_tr, x_va, x_te, y_tr, y_va, y_te, 
                                              num_epochs = args.Nepochs, batch_size = args.batch_size)
            valAcc += max(mHist.history['val_acc'])
            for k,v in mHist.history.items():
                print(k, v)
            appendFoldResults(f, args.resultsFile, confusion_matrix(y_te, preds), mHist)

        #### LeiLi ####
        elif(args.model_ver == 2):
            sess = initKerasSession(args.GPU_NUM)
            with sess.as_default():
                y_te, preds, mHist = runLeiLi(x_tr, x_va, x_te, y_tr, y_va, y_te, 
                                              num_epochs = args.Nepochs, batch_size = args.batch_size)
                valAcc += max(mHist.history['val_acc'])
                for k,v in mHist.history.items():
                    print(k, v)
                appendFoldResults(f, args.resultsFile, confusion_matrix(y_te, preds), mHist=mHist)
            
        #### LBP Histograms ####
        elif(args.model_ver == 3):
            classifier=lbphist()
            sess = initKerasSession(args.GPU_NUM)
            with sess.as_default():
                y_te, preds, mHist = classifier.classify(x_tr, x_va, x_te, y_tr, y_va, y_te, 
                                                  num_epochs = args.Nepochs, batch_size = args.batch_size)
                valAcc += max(mHist.history['val_acc'])
                for k,v in mHist.history.items():
                    print(k, v)
                appendFoldResults(f, args.resultsFile, confusion_matrix(y_te, preds), mHist=mHist)
        
        else:
            print("model %d not defined"%args.model_ver)
            exit()
        
        #Add the results for this fold to the overall confusion matrix
        confusionMat+=confusion_matrix(y_te, preds)

    #Now we have the total confusion matrix for all folds. Calculate measures from that.
    overallResults['NumTestPts']=sum(sum(confusionMat))
    overallResults['Accuracy'] = (confusionMat[0][0] + confusionMat[1][1])/overallResults['NumTestPts']
    overallResults['Precision'] = confusionMat[0][0]/(confusionMat[0][0]+confusionMat[0][1])
    overallResults['Recall'] = confusionMat[0][0]/(confusionMat[0][0]+confusionMat[1][0])
    overallResults['F1-Score']=2*overallResults['Precision']*overallResults['Recall'] / (overallResults['Precision'] + overallResults['Recall'])
    overallResults['Confusion']=confusionMat
    overallResults['ValidationAcc']=valAcc/len(folds)

    #Write the results to the screen
    print("\n******** Combined results over %d folds ************"%len(folds))
    for k, v in overallResults.items():
        if(k == 'NumTestPts'):
            print('%16s: %d'%(k, v))
        elif(k != 'Confusion'):
            print('%16s: %2.4f'%(k, v))

    print('Confusion Matrix:')
    for row in confusionMat:
        print('\t'+'\t '.join(map(str,row)))
    print("********     End of combined results    ************")

    #Write the results to file
    with open(args.resultsFile, 'a') as outfile:
        outfile.write("\n******** Combined results over %d folds ************\n"%len(folds))
        for k, v in overallResults.items():
            if(k == 'NumTestPts'):
                outfile.write('%16s: %d\n'%(k, v))
            elif(k != 'Confusion'):
                outfile.write('%16s: %2.4f\n'%(k, v))
        outfile.write('Confusion Matrix:\n')
        for row in confusionMat:
            outfile.write('\t'+'\t '.join(map(str,row)))
            outfile.write('\n')
        outfile.write('\nEnd time: '+str(datetime.datetime.now())+'\n')

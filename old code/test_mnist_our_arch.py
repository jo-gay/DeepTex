#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:38:46 2018

@author: jo
"""
import numpy as np
from sklearn.metrics import confusion_matrix
import datetime
from marcos import marcos

def loadMnistRot():
    def load_and_make_list(mode):
        data = np.load('RotEqNet/mnist/mnist_rot/' + mode + '_data.npy')
        lbls = np.load('RotEqNet/mnist/mnist_rot/' + mode + '_label.npy')
        data = np.split(data, data.shape[2],2)
        lbls = np.split(lbls, lbls.shape[0],0)

        return zip(data,lbls)

    train = load_and_make_list('train')
    val = load_and_make_list('val')
    test = load_and_make_list('test')
    return train, val, test


if __name__ == '__main__':
    GPU_NUM=0
    fastMode=False
    Nepochs=50
    batch_size=600
    numClasses=10
    dnow=datetime.datetime.now()
    resultsFile = 'savedResults%02d%02d.txt'%(dnow.hour, dnow.minute)
    #Load datasets
    train_set, val_set,  test_set = loadMnistRot()
    train_set=list(map(list, train_set)) #Turn into a list (JCG)
    val_set=list(map(list, val_set)) #Turn into a list (JCG)
    test_set=list(map(list, test_set)) #Turn into a list (JCG)
    
    classifier=marcos(GPU_NUM, fastMode, train_set[0][0].shape, numClasses)
    y_te, preds, v_acc, t_acc = classifier.classifyPt2(train_set, val_set, 
                                             test_set, num_epochs = Nepochs, 
                                             batch_size = batch_size)

    with open(resultsFile, 'w') as outfile:
        outfile.write('Validation accuracy: %2.5f\n'%(v_acc))
        outfile.write('Test accuracy: %2.5f\n'%(t_acc))
        outfile.write('Confusion matrix:\n')
        for row in confusion_matrix(y_te, preds):
            outfile.write('\t'+'\t '.join(map(str,row)))
            outfile.write('\n')
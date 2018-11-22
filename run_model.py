#import torch
#import  torch.nn as nn
from mnist_test import run
from first_model_cnn import doInitTasks
from first_model_cnn import runImageClassification
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
#def main(GPU_NUM=None, fastMode=False, model_ver = 0, Nepochs = 1):
    
    ## SELECTED NETWORK ##
    model_ver = 1
    #
    # 0 - Juefei
    # 1 - Marcos
    #
    ######################
    
    
    ### PARAMETERS ###
    GPU_NUM= 1
    Nepochs = 1
    batch_size = 100
    fastMode=False
    ##################
    
    
    folds = [1,2,3]
    confusionMat=[[0,0],[0,0]]
    overallResults={'Precision':0, 'Accuracy':0, 'Recall':0, 'F1-Score':0, 'NumTestPts':0, 'Confusion':[]}
    
    print("model_ver:")
    print(model_ver)
    if(fastMode):
        folds=[2,]
    
    for f in folds:
        print("\nStarting training with fold number %d of %d"%(f, len(folds)))
        
        #### Juefei ####
        if(model_ver == 0):
            sess = doInitTasks(GPU_NUM)
            with sess.as_default():
                y_te, res = runImageClassification(foldNr=f, num_epochs = Nepochs, batch_size = batch_size)
                preds=[int(x>0.5) for x in res]
        #### Marcos ####
        if(model_ver == 1):
            y_te, preds = run(gpu_no = GPU_NUM, fold_nr=f, num_epochs = Nepochs, batch_size = batch_size)
        
        if(False):
            print("shape res: ")
            print(res.shape)
            print("shape y_te: ")
            print(y_te.shape)
            
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

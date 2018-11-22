 #Trains the selected model on all three folds (or just fold 2 if fastMode = True)
 #Complies the the data into one confusion matrix and prints the result  
import mnist_test.py

confusionMat=[[0,0],[0,0]]
overallResults={'Precision':0, 'Accuracy':0, 'Recall':0, 'F1-Score':0, 'NumTestPts':0, 'Confusion':[]}
folds=trainfolds.keys()

fastMode = False
initial_lr = 0.1

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
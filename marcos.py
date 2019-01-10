from __future__ import division, print_function
import torch
import  torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from torch.autograd import Variable
import random

from marcos_layers_2D import *
from marcos_utils import rotate_im, random_rotation

"""
Adapted from mnist_test.py by Anders U. Waldeland
https://github.com/COGMAR/RotEqNet

A reproduction of the MNIST-classification network described in:
Rotation equivariant vector field networks (ICCV 2017)
Diego Marcos, Michele Volpi, Nikos Komodakis, Devis Tuia
https://arxiv.org/abs/1612.09346
https://github.com/dmarcosg/RotEqNet
"""

useOriginalParams=True

#dummy class to emulate Keras model history record
class trainingHistory(object):
    pass

# Define network
class Net(nn.Module):
    def __init__(self, imsz=80, nClasses=2):
        super(Net, self).__init__()

        if imsz==80: #our data
            if useOriginalParams: #Original settings
                self.main = nn.Sequential(
    
                    RotConv(1, 6, [9, 9], 1, 9 // 2, n_angles=17, mode=1),
                    VectorMaxPool(2),
                    VectorBatchNorm(6),
        
                    RotConv(6, 16, [9, 9], 1, 9 // 2, n_angles=17, mode=2),
                    VectorMaxPool(2),
                    VectorBatchNorm(16),
        
                    RotConv(16, 32, [9, 9], 1, 1, n_angles=17, mode=2),
                    VectorMaxPool(2),
                    VectorBatchNorm(32),
                    
                    RotConv(32, 32, [9, 9], 1, 1, n_angles=17, mode=2),
                    Vector2Magnitude(),
        
                    nn.Dropout2d(0.7), #added dropout layer because of overfitting
                    nn.Conv2d(32, 128, 1),  # FC1
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Dropout2d(0.7),
                    nn.Conv2d(128, nClasses, 1),  # FC2
                )
            else: #tuned to oral cancer dataset
                self.main = nn.Sequential(
                    RotConv(1, 16, [9, 9], 1, 0, n_angles=17, mode=1), #padding was 9//2
                    VectorMaxPool(2), #size 36 now
                    VectorBatchNorm(16),
        
                    RotConv(16, 32, [9, 9], 1, 0, n_angles=17, mode=2), #padding was 9//2
                    VectorMaxPool(2), #size 14 now
                    VectorBatchNorm(32),
        
                    RotConv(32, 64, [9, 9], 1, 9 // 2, n_angles=17, mode=2), #padding was 1
                    VectorMaxPool(2), #size 7 now 
                    VectorBatchNorm(64),
                    
                    RotConv(64, 64, [9, 9], 1, 1, n_angles=17, mode=2), #padding was 1
                    Vector2Magnitude(), #size 1 now (7-6)
        
                    nn.Conv2d(64, 128, 1),  # FC1
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Dropout2d(0.7),
                    nn.Conv2d(64, nClasses, 1),  # FC2
        
                )
        else: #mnist, size = 28
            self.main = nn.Sequential(
    
                RotConv(1, 6, [9, 9], 1, 9 // 2, n_angles=17, mode=1),
                VectorMaxPool(2), #size 14 now
                VectorBatchNorm(6),
    
                RotConv(6, 16, [9, 9], 1, 9 // 2, n_angles=17, mode=2),
                VectorMaxPool(2), #size 7 now
                VectorBatchNorm(16),
                
                RotConv(16, 32, [9, 9], 1, 1, n_angles=17, mode=2),
                Vector2Magnitude(), # size 1 now (7-6)
    
                nn.Conv2d(32, 128, 1),  # FC1
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Dropout2d(0.7),
                nn.Conv2d(128, nClasses, 1),  # FC2
    
            )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size()[0], x.size()[1])
        return x


class marcos:
    def __init__(self, gpu_no=0, fastMode=False, shape=(1,80,80), numClasses=2):
        '''Set up parameters for the model
        '''
        self.fastMode = fastMode
        self.gpu_no=gpu_no
        self.use_train_time_augmentation=False
        self.use_test_time_augmentation=False
        self.imshape=shape
        self.numClasses=numClasses
        
    def test(self, model, dataset, mode, criterion, returnval = 0):
        """ Return test acuracy for a given dataset
            If returnval != 0, return labels for test set, predictions, and accuracy.
            Labels are returned because the given test dataset will be truncated to the number
            of whole batches available.
        """
        model.eval()

        true = []
        pred = []
        for batch_no in range(len(dataset) // self.batch_size):
            data, labels = self.getBatch(dataset, mode)

            #Run same sample with different orientations through network and average output
            if self.use_test_time_augmentation and mode == 'test':
                data = data.cpu()
                original_data = data.clone().data.cpu().numpy()

                out = None
                rotations = [0,15,30,45, 60, 75, 90]

                for rotation in rotations:

                    for i in range(self.batch_size):
                        im = original_data[i,:,:,:].squeeze()
                        im = rotate_im(im, rotation, self.imshape[1])
                        if self.imshape[0]>self.imshape[-1]:
                            im = im.reshape([1, self.imshape[-1], self.imshape[1], self.imshape[0]])
                        else:
                            im = im.reshape([1, self.imshape[0], self.imshape[1], self.imshape[2]])
                        im = torch.FloatTensor(im)
                        data[i,:,:,:] = im

                    if type(self.gpu_no) == int:
                        data = data.cuda(self.gpu_no)

                    if out is None:
                        out = F.softmax(model(data))
                    else:
                        out += F.softmax(model(data))

                out /= len(rotations)

            #Only run once
            else:
                out = F.softmax(model(data),dim=1)

            #loss = criterion(out, labels)
            _, c = torch.max(out, 1)
            true.append(labels.data.cpu().numpy())
            pred.append(c.data.cpu().numpy())
        true = np.concatenate(true, 0)
        pred = np.concatenate(pred, 0)
        if mode == 'test':
            print("Predicted:", pred, "\nTrue:", true)
        acc = np.average(pred == true)
        if(returnval ==  0):
            return acc
        else:
            print('Test accuracy:', acc)
            return true, pred, acc

    def getBatch(self, dataset, mode):
        """ Collect a batch of samples from list """

        # Make batch
        data = []
        labels = []
        for sample_no in range(self.batch_size):
            tmp = dataset.pop()  # Get top element and remove from list
            img = tmp[0].astype('float32').squeeze()

            # Train-time random rotation
            if mode == 'train' and self.use_train_time_augmentation:
                img = random_rotation(img, self.imshape[1])

            data.append(np.expand_dims(np.expand_dims(img, 0), 0))
            labels.append(tmp[1].squeeze())
        data = np.concatenate(data, 0)
        labels = np.array(labels, 'int32')

        data = Variable(torch.from_numpy(data))
        labels = Variable(torch.from_numpy(labels).long())

        if type(self.gpu_no) == int:
            data = data.cuda(self.gpu_no)
            labels = labels.cuda(self.gpu_no)

        return data, labels

    def adjust_learning_rate(self, optimizer, epoch):
        """Gradually decay learning rate"""
        if epoch == 20:
            self.lr = self.start_lr / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
        if epoch == 40:
            self.lr = self.start_lr / 100
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
        if epoch == 60:
            self.lr = self.start_lr / 1000
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
    
    
    def classify(self, x_tr, x_va, x_te, y_tr, y_va, y_te, num_epochs=2, batch_size=128):
        '''classify function takes training, validation and test datasets and returns
        true labels for test dataset (same as y_te but maybe reordered and/or truncated), 
        predicted labels for test dataset, test accuracy. 
        Model is trained on x_tr/y_tr for num_epochs epocs and with batch size batch_size.
        Model weights are saved each time validation accuracy (as calculated using x_va/y_va)
        increases and after all epochs are complete, are reloaded. This reloaded model is
        used for testing.
        '''
        train_set = list(zip(x_tr, y_tr))
        val_set = list(zip(x_va, y_va))
        test_set = list(zip(x_te, y_te)) 
        
        return self.classifyPt2(train_set, val_set, test_set, num_epochs, batch_size)
        
    def classifyPt2(self, train_set, val_set, test_set, num_epochs=50, batch_size=128):
        '''Split classify function into separate parts to better handle the case where 
        the data is already in the zipped format required
        '''
        bestModelFname='best_model.pt'
        #Setup net, loss function, optimizer and hyper parameters
        net = Net(train_set[0][0].shape[1], self.numClasses)
        if type(self.gpu_no) == int:
            net.cuda(self.gpu_no)
        criterion = nn.CrossEntropyLoss()

        if not useOriginalParams: #Current best setup using this implementation - error rate of 1.2%
            self.start_lr = 0.01
            self.batch_size = batch_size
            optimizer = optim.Adam(net.parameters(), lr=self.start_lr)  # , weight_decay=0.01)
            self.use_test_time_augmentation = False #was True
            self.use_train_time_augmentation = False
        else: #From paper using MATLAB implementation - reported error rate of 1.4%
            self.start_lr = 0.1 #was 0.1
            self.batch_size = batch_size #was 200
            optimizer = optim.SGD(net.parameters(), lr=self.start_lr, weight_decay=0.01) #was weight_decay=0.01
            self.use_test_time_augmentation = False #was True
            self.use_train_time_augmentation = False #was True

        self.lr = self.start_lr
        
        best_acc = 0
        mHist=trainingHistory()
        mHist.history={'lr':[], 'val_acc':[]}
        for epoch_no in range(num_epochs):
    
            #Random order for each epoch
            train_set_for_epoch = train_set[:] #Make a copy
            random.shuffle(train_set_for_epoch) #Shuffle the copy
    
            #Training
            net.train()
            for batch_no in range(len(train_set)//self.batch_size):
    
                # Train
                optimizer.zero_grad()
    
                data, labels = self.getBatch(train_set_for_epoch, 'train')
                out = net( data )
                loss = criterion( out,labels )
                _, c = torch.max(out, 1)
                loss.backward()
    
                optimizer.step()
    
                #Print training-acc
                if batch_no%10 == 0:
                    print('Train', 'epoch:', epoch_no, ' batch:', batch_no, ' loss:', loss.data.cpu().numpy(), ' acc:', np.average((c == labels).data.cpu().numpy()))

            del data, labels, out, train_set_for_epoch #make some room
    
            #Validation
            acc = self.test(net, val_set[:], 'val', criterion)
            print('Val',  'epoch:', epoch_no,  ' acc:', acc)
    
            #Save model if better than previous
            if acc > best_acc:
                torch.save(net.state_dict(), bestModelFname)
                best_acc = acc
                print('Model saved')
            
            mHist.history['lr'].append(self.lr)
            mHist.history['val_acc'].append(acc)
            self.adjust_learning_rate(optimizer, epoch_no)
    
        # Finally test on test-set with the best model
        net.load_state_dict(torch.load(bestModelFname))
        
        true, pred, acc = self.test(net, test_set[:], 'test', criterion, returnval = 1)
    
        return true, pred, mHist
        

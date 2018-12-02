from mahotas.features import lbp
import numpy as np
from keras import layers, models, optimizers, losses
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler


class lbphist:
    '''Build a neural network which takes LBP histograms as input
    '''
    def __init__(self, R=10, P=4, uniform=True):
        self.R = R
        self.P = P
        self.uniform = uniform


#%%    
    def transform_data_to_LBPs(self, x):
        '''Given some input images, transform them to histograms of LBP patterns
        Combine non-uniform patterns if self.uniform is true.
        '''
        #Mahotas returns all 36 rotation invariant patterns. We want to only use the uniform ones
        #and take the non-uniform ones as a single data point.

        #Take pattern histograms and combine patterns as proposed in Ojala et al 2002
        #Uniform patterns 0-P map to 0-P and all others map to P+1.
        #Hist is an array of histograms, each element is an array of pattern counts for a particular image
        def combinePatterns(hist):
            ret=[list(hist[i][0:(self.P+1)])+[sum(hist[i][(self.P+1):])] for i in range(len(hist))]
            return ret

        nSamples=len(x)
        nPixels=1
        
        shape=x[0].shape
        for a in shape:
            nPixels *= a 
        
        lbp_x=[]
        for i in range(nSamples):
            lbp_x.append(lbp(x[i].reshape(shape[:-1]), self.R, self.P, ignore_zeros=False)/nPixels)

        if(self.uniform):
            lbp_x=combinePatterns(lbp_x)
        
        nBins=len(lbp_x[0])
        lbp_x=np.array(lbp_x).reshape((nSamples,nBins))
        return lbp_x

    def build_lbp_hists_nn(self, x):

        x = layers.BatchNormalization()(x)
        x = layers.Dense(160, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(320, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(640, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(640, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(160, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        #Output layer
        x = layers.Dense(1, activation='sigmoid')(x)
    
        return x
    
    def getModel(self, lbp_shape):
        image_tensor = layers.Input(shape=lbp_shape)
        network_output = self.build_lbp_hists_nn(x=image_tensor)
          
        model = models.Model(inputs=[image_tensor], outputs=[network_output])
        print(model.summary())
        
        adam = optimizers.Adam()
        model.compile(loss = losses.binary_crossentropy, metrics = ['accuracy'], optimizer = adam)
        return model

#%%
    def step_decay_schedule(self, initial_lr, nr_training, batch_size):
        '''
        Wrapper function to create a LearningRateScheduler with step decay schedule.
        '''
        
        def schedule(epoch):
            decay = 1
            if(epoch*(nr_training/batch_size) >= 1000):
                decay = decay /10
            if(epoch*(nr_training/batch_size) >= 2000):
                decay = decay /10
            if(epoch*(nr_training/batch_size) >= 3000):
                decay = decay /10
            print('learning rate: %f' %(initial_lr*decay))    
            return initial_lr * decay
        return LearningRateScheduler(schedule)
    

    def fitModel(self, cNN, epochs, batch_size, x_tr, y_tr, x_va, y_va):
        path = "weights-best.hdf5"
        lr_sched = self.step_decay_schedule( 0.01, len(x_tr), batch_size)
        model_checkpoint = ModelCheckpoint(filepath = path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=1)
        early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None)
        mHist=cNN.fit(x_tr, y_tr, batch_size=batch_size, epochs = epochs, validation_data = (x_va, y_va),callbacks = [lr_sched, model_checkpoint, early_stopping])
        cNN.load_weights(path)
        return cNN, mHist

    
    def classify(self, x_tr, x_va, x_te, y_tr, y_va, y_te, num_epochs=50, batch_size=20, eval_model=0):
        print("Transforming input images to LBPs...")
        x_tr=self.transform_data_to_LBPs(x_tr)
        x_va=self.transform_data_to_LBPs(x_va)
        x_te=self.transform_data_to_LBPs(x_te)

        # Create model 
        print("Creating model...")
        model=self.getModel(x_tr[0].shape)
        
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

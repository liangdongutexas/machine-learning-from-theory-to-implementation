#!/usr/bin/env python
# coding: utf-8

# # The following cell is the model class. 
# Its __call__ method returns the predicted FPS according to aformentioned formula.
# Its load_variables method loads previously trained parameters which will be used by the __call__ method to make predictions.

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


class layer():
    def __init__(self,shape=None,activation=None):
        self.shape=shape
        self.activation=activation


# In[3]:


class train_action():
    def __init__(self):
        
        # define all the metric loss functions
        self.losses={'binary_crossentropy':tf.keras.losses.binary_crossentropy,
          'MSE': tf.keras.losses.MSE,'CategoricalCrossentropy':tf.keras.losses.CategoricalCrossentropy,\
                   'categorical_crossentropy':tf.keras.losses.categorical_crossentropy
}
        
    ## uses tensorflow to do backpropagation onece for each epoch.
    def train_one_step(self,X,Y):
        with tf.GradientTape() as tape:
            predict = self.predict(X)
            loss=self.compute_loss(predict, Y)
            optimizer=self.optimizer
            # compute gradient
            grads = tape.gradient(loss, self.trainable_variables)
            # update to weights
            optimizer.apply_gradients(zip(grads, self.trainable_variables))   
        
     
    def compute_loss(self,predict, Y):       
        return self.losses[self.loss](predict,Y)
    
    def save_model(self,filename):
        stored_variables=np.array([i.numpy() for i in self.trainable_variables])
        np.save(filename, stored_variables,allow_pickle=True, fix_imports=True)
        print('model has been saved')


# In[4]:


## shape is a list of tuples.
## The i_th element represents the tensor structure of the i_th layer.
## The 0_th element represents the shape of the input.
## The -1_th element represents the shape of the output.
class sequential(train_action):
    
    def __init__(self,layers):       
        super().__init__()
        
        ## attribute cannot be a function but can be a dictionary of functions.
        self.activations={'relu':tf.keras.activations.relu,'elu':tf.keras.activations.elu,            'sigmoid':tf.keras.activations.sigmoid, 'selu': tf.keras.activations.selu,
                        'softmax':tf.keras.activations.softmax}

        
        self.layers=layers
        self.weights=[]
        self.bias=[]
        self.trainable_variables=[]

        for i in range(len(layers)-1):
            weight=tf.Variable(tf.random.truncated_normal(shape=self.layers[i].shape+self.layers[i+1].shape))
            bias=tf.Variable(tf.zeros(shape=self.layers[i+1].shape))
            self.weights.append(weight) 
            self.bias.append(bias) 
        self.trainable_variables=self.weights+self.bias
        
    def predict(self,X=None):
        X=tf.cast(X,tf.float32)
        Y=X
        for i in range(0,len(self.layers)-1): #i is the layer we are right now and the
                                              # data will propagate to the next layer
                                              # so the we should use the activation function in the next layer 
            
            # the shape of current layer, the first one is the batch size
            backdim=self.layers[i].shape 
            
            # the shape of next layer
            frontdim=self.layers[i+1].shape      
            bl=len(backdim)
            
            
            Y=tf.tensordot(Y,self.weights[i],                           axes=[np.arange(1,bl+1),np.arange(0,bl)])+self.bias[i]
            if self.layers[i+1].activation:
                Y=self.activations[self.layers[i+1].activation](Y)
             


        return Y
    
    def compile(self,loss=None,optimizer=None):
        # loss is string name of the tf.keras.losses functions
        # optimizer is a class for most cases in tensorflow, so the optimizer here is a class
        self.loss=loss
        self.optimizer=optimizer
        
    def train(self,X,Y,epochs=100,epochgroup=10,save_name=None):
        
        for epoch in range(epochs): 
            if epoch%epochgroup==0:
                predict=self.predict(X)
                print('for epoch {}, {} is {}'                  .format(epoch,self.loss,tf.math.reduce_mean(self.compute_loss(predict,Y))))
            self.train_one_step(X,Y)  
        if save_name:
            self.save_model(save_name)
        
    


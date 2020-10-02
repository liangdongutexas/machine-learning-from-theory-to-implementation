#!/usr/bin/env python
# coding: utf-8

# # The following cell is the model class. 
# Its __call__ method returns the predicted FPS according to aformentioned formula.
# Its load_variables method loads previously trained parameters which will be used by the __call__ method to make predictions.

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


class train_action():
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
        
    ## computes the mean squared error of predicted FPS with respect to the real FPS at those tested data point in F.  
    def compute_loss(self,predict, Y):
        loss = self.loss         
        return loss(predict,Y)
    
    def save_model(self,filename):
        stored_variables=np.array([i.numpy() for i in self.trainable_variables])
        np.save(filename, stored_variables,allow_pickle=True, fix_imports=True)
        print('model has been saved')


# In[3]:


## shape is a list of tuples.
## The i_th element represents the tensor structure of the i_th layer.
## The 0_th element represents the shape of the input.
## The -1_th element represents the shape of the output.
class model(train_action):
    def __init__(self,shape):
        self.shape=shape
        self.weights=[]
        self.bias=[]
        self.trainable_variables=[]
        for i in range(len(self.shape)-1):
            layer=tf.Variable(tf.random.truncated_normal(shape=self.shape[i]+self.shape[i+1]))
            bias=tf.Variable(tf.random.truncated_normal(shape=self.shape[i+1]))
            self.weights.append(layer) 
            self.bias.append(bias) 
        self.trainable_variables=self.weights+self.bias
    def predict(self,X=None):
        X=tf.cast(X,tf.float32)
        Y=X
        for i in range(0,len(self.shape)-1):
            backdim=self.shape[i]    # the shape of current layer
            frontdim=self.shape[i+1] # the shape of next layer
            bl=len(backdim)
            try:
                Y=tf.tensordot(Y,self.weights[i],                               axes=[np.arange(1,bl+1),np.arange(0,bl)])+self.bias[i]
                Y=tf.nn.relu(Y)
            except:
                print(Y.shape,self.weights[i].shape,self.bias[i].shape,bl)

        return Y
    def compile(self,loss=None,optimizer=None):
        self.loss=loss
        self.optimizer=optimizer
        
    def train(self,X,Y,epochs=10,save_name=None):
        
        for epoch in range(epochs): 
            predict=self.predict(X)
            print('for epoch {}, MSE is {}'.format(epoch,self.compute_loss(predict,Y)))
            self.train_one_step(X,Y)  
        if save_name:
            self.save_model(save_name)
        
    


# In[ ]:







## (X_train,Y_train),(X_test,Y_test)=tf.keras.datasets.mnist.load_data()




# X_train = tf.cast(X_train, tf.float32) / 255.0
# X_test=tf.cast(X_test, tf.float32) / 255.0




# X_train=X_train[:100]





# shape=[(28,28),(14,14),(7,7),(14,14),(28,28)]
# testmodel=model(shape)





#testmodel.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))





#testmodel.train(X_train,X_train,save_name='weights_bias')


 





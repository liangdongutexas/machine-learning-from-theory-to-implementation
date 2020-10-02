#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np


# The loss function at each node in xgboost reads:
# $$
# -\frac{g^2_t}{2(h_t+\lambda)}+\frac{\gamma}{K+1},
# $$
# where $g_t=\sum_{ \{i|q(\boldsymbol{x}_i)=t\}}\frac{g_i}{N}$ and $h_t=\sum_{ \{i|q(\boldsymbol{x}_i)=t\}}\frac{h_i}{N}$.

# In[19]:


class xgboost_loss():
    def __init__(self,K,gamma,lambd):
        
        self.K=K
        self.gamma=gamma
        self.lambd=lambd
    
    def loss(self,N,Z):
        '''Z: [(g_0,h_0),(g_1,h_1),...,(g_i,h_i)] is the  loss coefficient for each instance in the sample at current tree node;
           N is the total training sample size.
        '''
        g_t,h_t=np.sum(Z,axis=0)/N
        
        loss=-g_t**2/(2*(h_t+self.lambd))+self.gamma/(self.K+1)
        
        return loss


# In[ ]:





#!/usr/bin/env python
# coding: utf-8


# We limit to the case where there is only one categorical outcome with several predictors. We let the user identify whether they want this feature treated as discrete or continuous number. If coninuous we will use binary threshold, else we will split by asking whether it belongs to a particular catogory or not. 



import numpy as np


# In[3]:


# each DecisionNode represents a subspace in the dividing process described before
class DecisionNode():
    """Class that represents a decision node or leaf in the decision tree
    
    Parameters:
    -----------
    disorder: float
        how disordered the subspace represented by the node is
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """
    
    def __init__(self):
        self.disorder=None                 # the disorder of the subspace represented by this node
        self.distribution=distribution     # the distribution of outcome in this subspace
        self.feature_i=None                # the ith feature choosed to further divide the current subspace
        self.threshold=None                # the threshold used to divide the feature i
        self.true_branch=None              # the true subspace of the current space with respect to the division
        self.false_branch=None             # the false subspace of the current space with respect to the division
        self.error=None


# In[4]:


class DecisionTree(object):
    """
    X=np.array(shape=[samplesize,feature])
    Y=np.array(shape=[samplesize])
    feature_name_type=np.array([(name,'c'/'d')...])
    where 'c' respresents continuous variabl and 'd' represents discrete variable
    """
    def __init__(self,max_depth=np.Infinity,min_sample_size=1,disorder_thres=0,loss='entropy'):
        self.loss_function={'entropy':entropy,'gini':gini}
        self.loss=loss
        self.max_depth=max_depth
        self.min_sample_size=min_sample_size
        self.disorder_thres=disorder_thres
        
    def fit(self,X,Y,feature_name_type):
        # attribute self.feature_name_type and self.root related to the fitted data
        # and be renewed if new data is fitted
        
        self.feature_name_type=np.array(feature_name_type)
        self.sample_size=len(X)
        self.outcome=np.unique(Y)
        
        # just to store the root of the tree 
        # consistent check whether the shape of inputs are expected:
        condition1= ((len(np.shape(X))==2)
                    and (len(np.shape(feature_name_type))==2)
                    and (len(np.shape(Y))==1))         
        condition2= ((np.shape(X)[0]==np.shape(Y)[0]) 
                     and (np.shape(X)[1]==np.shape(feature_name_type)[0]))
        
        if condition1 and condition2:
            Y = np.expand_dims(Y, axis=1)
            # Add Y as last column of X so easier to split them together
            XY = np.concatenate((X, Y), axis=1)
            self.root=self.build_tree(XY)
        else:
            raise ValueError('The shapes of inputs are not compatible')
    
    def build_tree(self,XY,depth=0):
        
        # record the curren state of the sample:
        P_Y=distribution(XY[:,-1])                    # distribution of outcome for current node or subspace
        for key in self.outcome:                      # update zero for outcomes that's not in the current XY
            P_Y[key]=P_Y.get(key,0)
        disorder_Y=self.loss_function[self.loss](P_Y) # loss (entropy, gini, mean squared error) for current subspace
        node=DecisionNode()
        node.distribution=P_Y
        node.sample_pro=len(XY)/self.sample_size
        
        # whether the state of current node is suitable for further division:
        # 1. reach maximal depth 2. sample size statitical significant 3. the disorder is high enough
        if depth<=self.max_depth and len(XY)>=self.min_sample_size and disorder_Y>self.disorder_thres:                                    
            disorder_best,feature_best,threshold_best,XY_left,XY_right=self.best_split(XY,self.feature_name_type)    
            if disorder_best/disorder_Y<=0.9:
                node.feature_i=feature_best
                node.threshold=threshold_best
                node.true_branch=self.build_tree(XY_left,depth=depth+1)
                node.false_branch=self.build_tree(XY_right,depth=depth+1)
            else:
                node.error='not divisiable'
                
        return node                   
        
        
    def best_split(self,XY,feature_name_type):  
      
        loss_best=np.Infinity
        feature_best=None
        threshold_best=None
        XY_left=None
        XY_right=None
                
        for feature_i in range(np.shape(XY)[1]-1):                  # O(n) complexity here

            values_feature_i=np.unique(XY[:,feature_i])   

            for threshold in values_feature_i:                   # O(m)
                smaller_equal,larger=binary_split(XY,feature_i,threshold,dtype=feature_name_type[feature_i,1]) # O(m)
                
                try:
                    loss_Left=entropy(distribution(smaller_equal[:,-1]))
                except:
                    loss_Left=0
                # if larger is empty
                try:
                    loss_Right=entropy(distribution(larger[:,-1]))
                except:
                    loss_Right=0
                loss_target=(len(smaller_equal)/len(XY))*loss_Left+(len(larger)/len(XY))*loss_Right

                if loss_target<=loss_best:
                    loss_best=loss_target
                    feature_best=feature_i
                    threshold_best=threshold
                    XY_left=smaller_equal
                    XY_right=larger
                    
        return  loss_best,feature_best,threshold_best,XY_left,XY_right 
    
    def predict(self,X):
        predictions=[]
        pre_distributions=[]
        if len(np.shape(X))==2 and np.shape(X)[1]==len(self.feature_name_type):
            for instance in X:
                currnode=self.root
                while currnode.feature_i: # the current node has branches
                    #print(currnode.distribution)
                    feature_type=self.feature_name_type[currnode.feature_i,1]
                    feature_name=self.feature_name_type[currnode.feature_i,0] 
                    if feature_type=='c':
                        if instance[currnode.feature_i]<=currnode.threshold:
                            #print('{} smaller or equal to {}'.format(feature_name,currnode.threshold))
                            currnode=currnode.true_branch
                        else:
                            #print('{} larger than {}'.format(feature_name,currnode.threshold))
                            currnode=currnode.false_branch                    
                    elif feature_type=='d':
                        if instance[currnode.feature_i]==currnode.threshold:
                            #print('{} is {}'.format(feature_name,currnode.threshold))
                            currnode=currnode.true_branch
                        else:
                            #print('{} is not {}'.format(feature_name,currnode.threshold)) 
                            currnode=currnode.false_branch                   
                    else:
                        raise ValueError('feature dtype should be c for continuous or d for discrete')
                pre_distributions.append(currnode.distribution)
                
            for pre_dis in pre_distributions:
        
                pre=None
                max_prob=max(pre_dis.values())
                for key in pre_dis.keys():
                    if pre_dis[key]==max_prob:
                        pre=key
                                    
                predictions.append(pre)
            # print(pre_distributions)   
            return np.array(predictions)
        else:
            raise ValueError('the shape of X should be (sample,feature)')
            
    def summarize_feature(self,target):
        """give a summary of the key features that lead to the conclusion of a particular outcome
        target:
                The outcome with respect to which the importantce of features is summarized        
        """
        curr_node=self.root
        self.summary={}
        self.update_entropy(curr_node,target)
        self.update_entropy_reduction(curr_node)
        return self.summary        
    
    def update_entropy(self,curr_node,target):
        """use DFS to transverse the trained decision tree"""
        
        curr_node.disorder=entropy(curr_node.distribution,target)

     
        if curr_node.feature_i!=None:      # there is further spliting for this node
            self.update_entropy(curr_node.true_branch,target)
            self.update_entropy(curr_node.false_branch,target)
        else:
            pass
    
    def update_entropy_reduction(self,curr_node):
        nd=lambda node: node.sample_pro*node.disorder
        
        if curr_node.feature_i!=None:   # only those nodes making decision contributes to the disorder reduction
            disorder_reduction=nd(curr_node)-(nd(curr_node.true_branch)+nd(curr_node.false_branch))   
            self.summary[curr_node.feature_i]=self.summary.get(curr_node.feature_i,0)+disorder_reduction
            self.update_entropy_reduction(curr_node.true_branch)
            self.update_entropy_reduction(curr_node.false_branch)  


# In[5]:


def distribution(Y):
    if len(Y)==0:
        return None
    else:
        Y_unique=np.unique(Y)    
        P_Y={i:0 for i in Y_unique} 
        for i in Y:
            P_Y[i]+=1/len(Y)
        return P_Y

def entropy(P_Y):
    if not P_Y:
        return 0
    else:
        entropy=0  
        log2 = lambda x: np.log(x) / np.log(2)
        for i in P_Y.keys():   
            entropy+= -(P_Y[i])*log2(P_Y[i])
        return entropy

def gini(P_Y):
    gini=1
    for i in P_Y.keys():
        gini-=P_Y[i]**2
    return gini

def binary_split(XY, feature_i, threshold,dtype=None):
    """ Divide dataset based on if sample value on feature index is larger than
        the given threshold """
    split_func = None
    if dtype=='c':
        split_func = lambda sample: sample[feature_i] <= threshold
    elif dtype=='d':
        split_func = lambda sample: sample[feature_i] == threshold
    else:
        raise ValueError('The datatype of feature is incorrect. \'c\' for continuous feature                 and \'d\' for discrete feature')

    smaller_equal = np.array([sample for sample in XY if split_func(sample)])
    larger = np.array([sample for sample in XY if not split_func(sample)])

    return smaller_equal,larger


# ## Test <a name="test"></a>

# In[6]:

if __name__=='__main__':
    from sklearn.datasets import load_iris


    # In[7]:


    iris=load_iris()
    X=iris.data[:,2:] # petal length and width
    Y=iris.target
    feature_names=iris.feature_names[2:]
    feature_name_type=[[i,'c'] for i in feature_names]


    # In[8]:


    X.shape
    Y.shape
    feature_name_type


    # In[9]:


    tree=DecisionTree()
    tree.fit(X,Y,feature_name_type)


    # In[10]:


    tree.predict(X[20:100])


    # Then we compare the above results with the prediction made by the standard module.

    # In[11]:


    from sklearn.tree import DecisionTreeClassifier


    # In[12]:


    tree_clf=DecisionTreeClassifier(max_depth=2)
    tree_clf.fit(X,Y)


    # In[13]:


    tree_clf.predict(X[20:100])


    # Ultimately, we can compare on the whole data set:

    # In[14]:


    tree.predict(X[:])==tree_clf.predict(X[:])


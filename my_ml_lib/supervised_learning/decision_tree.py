import numpy as np
from graphviz import Digraph

## define a tree node decorator: takes in a model class and create a child class out of it, then return the child class
#  to create a tree_node class e.g. node=tree_node_decorator(linear_regression)()
def tree_node_decorator(model,hparameter=None,stats=None):
    class node_model(model):
        def __init__(self,hparameter=hparameter,stats=stats):

            if not hparameter:
                super().__init__(stats=stats)              
            else:
                super().__init__(hparameter=hparameter,stats=stats)


            # tree_node: the left or true branch 
            self.true=None 

            # tree_node: the right or false branch
            self.false=None



            # parameters used in the split method: return true if the split criterion suggest going to the true child node otherwise false
            self.feature=None       
            self.dtype=None
            self.threshold=None
            
            # probability that a instance falls into current node
            self.probability=1
            
            # store the index of the subsamples belong to the current node in the total training sample
            self.sample_index=None


        def find_child(self,unit):
            '''for each unit return true or false according to the criteria'''
            if self.dtype=='c':
                return self.true if unit[self.feature] <= self.threshold else self.false

            elif self.dtype=='d':
                 return self.true if unit[self.feature] == self.threshold else self.false

            else:
                raise ValueError('The datatype of feature is incorrect. \'c\' for continuous feature \
                        and \'d\' for discrete feature')
    return node_model


## define a decision tree decorator: takes in a node class and create a child class out of it, then return the child class
#  to create a tree_node class e.g. node=tree_node_decorator(linear_regression)()

def decision_tree_decorator(tree_node,max_depth=np.Infinity,min_sample_size=2,gamma=0):
    class decision_tree():
        """
        X=np.array(shape=[samplesize,feature])
        Y=np.array(shape=[samplesize])
        feature_name_dtype=np.array([(name,'c'/'d')...])
        where 'c' respresents continuous variabl and 'd' represents discrete variable
        """
        def __init__(self,max_depth=max_depth,min_sample_size=min_sample_size,gamma=gamma):
            self.max_depth=max_depth
            self.min_sample_size=1
            self.min_sample_size=min_sample_size
            self.gamma=gamma
                                 


            # quantities used in building tree
            self.root=None
            self.feature_name_dtype=None
            self.node_hparameter=None

        def fit(self,X=[],Y=[],XY_mul=[],feature_name_dtype=None):
            '''
            feature_name_dtype=[feature_name,'c' for ordinal feature and 'd' for categorical feature]
            hparameter: hyperparameters for the model of on the tree node 
            '''
            self.feature_name_dtype=np.array(feature_name_dtype)
            
            if len(X)!=0 and len(Y)!=0:
        
                # just to store the root of the tree 
                # consistent check whether the shape of inputs are expected:
                condition1= ((len(np.shape(X))==2)
                            and (len(np.shape(self.feature_name_dtype))==2))         
                condition2= ((np.shape(X)[0]==np.shape(Y)[0]) 
                             and (np.shape(X)[1]==np.shape(self.feature_name_dtype)[0]))

                if condition1 and condition2:
                    if len(np.shape(Y))==1: 
                        Y = np.expand_dims(Y, axis=1)

                    # Add Y,g,h as last columns to X so easier to split them together
                    XY = np.concatenate((X, Y), axis=1)

                else:
                    raise ValueError('The shapes of inputs are not compatible')

                # store data 
                self.XY_mul=np.array([sorted(XY,key=lambda unit: unit[i]) for i in range(X.shape[1])])                               
                
            elif len(XY_mul)!=0:
                self.XY_mul=XY_mul
                
            else:
                raise ValueError('No input data is given')
                
            currnode_index=np.ones(self.XY_mul.shape[:-1],dtype=bool)
                                       
            self.root=self.build_tree(currnode_index=currnode_index)

            # pruning trees
            self.pruning(self.root)

            # release memory
            self.XY_mul=0


        def build_tree(self,currnode_index=True,depth=1,stats=None,probability=1):   
            
            
            # pick the units out of the total samples that correspond to current node determined by the currnode_index
            XY_mul=self.XY_mul[currnode_index].reshape(len(self.feature_name_dtype),-1,len(self.feature_name_dtype)+1)
            
            parent_condition=((depth<self.max_depth) and (XY_mul.shape[1]>=2*self.min_sample_size))
            
            leaf_condition=((depth==self.max_depth) or (XY_mul.shape[1]<2*self.min_sample_size))
            
            null_condtion=((depth>self.max_depth) or (XY_mul.shape[1]<self.min_sample_size))
            
            if null_condtion:
                return None
            
            elif leaf_condition:
                node=tree_node(stats=stats)
                node.fit(XY_mul[0,:,:-1],XY_mul[0,:,-1:])             
                node.probability=probability
                node.sample_index=currnode_index
                
                return node
                    
            elif parent_condition:    
                # this node has the potential to be parent
                node=tree_node(stats=stats)                    
                node.probability=probability
                node.sample_index=currnode_index

                best=[np.Infinity,0,0]  ##[best split loss, feature_i whic gives best split, value of feature_i gives best split]
                best_split_stats=[[],[]]     ##[left split stats,right split stats]
                best_split_left_size=0
                best_split_right_size=0

                for feature_i in range(len(XY_mul)):
                    if self.feature_name_dtype[feature_i][1]=='c':
                        b_split_left_stats,b_split_right_stats,b_split_left_size,b_split_right_size,b_split_value,b_loss\
                        =node.online_fit(XY_mul[feature_i,:,:-1],XY_mul[feature_i,:,-1:],feature_i,feature_name_dtype=self.feature_name_dtype)
                    elif self.feature_name_dtype[feature_i][1]=='d':
                        b_split_left_stats,b_split_right_stats,b_split_left_size,b_split_right_size,b_split_value,b_loss\
                        =node.batch_fit(XY_mul[feature_i,:,:-1],XY_mul[feature_i,:,-1:],feature_i,feature_name_dtype=self.feature_name_dtype)

                    if b_loss<best[0]:
                        best[0]=b_loss
                        best[1]=feature_i
                        best[2]=b_split_value

                        best_split_stats[0]=b_split_left_stats
                        best_split_stats[1]=b_split_right_stats
                        
                        best_split_left_size=b_split_left_size
                        best_split_right_size=b_split_right_size
                        
                # whether disorder reduction is significant enough to support the division
                condition=(best_split_left_size>=self.min_sample_size) and (best_split_right_size>=self.min_sample_size)
                
                if condition: 
                # this is truly a parent node
                    if self.feature_name_dtype[best[1],1]=='c':
                        node.dtype='c'        
                        left_index=(self.XY_mul[:,:,best[1]]<=best[2]) & (currnode_index)
                        right_index=(self.XY_mul[:,:,best[1]]>best[2]) & (currnode_index)
                    else:
                        node.dtype='d'
                        left_index=(self.XY_mul[:,:,best[1]]==best[2]) & (currnode_index)
                        right_index=(self.XY_mul[:,:,best[1]]!=best[2]) & (currnode_index)

                    node.feature=best[1]
                    node.threshold=best[2]
                    
                    
                    pro_left=best_split_left_size/(best_split_left_size+best_split_right_size)*node.probability
                    pro_right=best_split_right_size/(best_split_left_size+best_split_right_size)*node.probability
                    
                    ## add child branches
                    node.true=self.build_tree(currnode_index=left_index,depth=depth+1,stats=best_split_stats[0],probability=pro_left)
                    node.false=self.build_tree(currnode_index=right_index,depth=depth+1,stats=best_split_stats[1],probability=pro_right)                       
                
                return node 
            
        def pruning(self,currnode):
            if currnode.true is None:
                return currnode
            else:
                # prune child nodes first
                currnode.true=self.pruning(currnode.true)
                currnode.false=self.pruning(currnode.false)
                
                # whether current node is the parent of two leaf nodes
                if currnode.true.true is None and currnode.false.false is None:
                    loss_leaves=(currnode.true.loss*currnode.true.probability\
                                +currnode.false.loss*currnode.false.probability+2*self.gamma)
                    loss_parent=currnode.loss*currnode.probability+self.gamma
                    
                    # whether loss reduction is too small: need to be removed
                    if (loss_parent!=0) and (loss_leaves/loss_parent>=0.9):
                        currnode.true=None
                        currnode.false=None
                return currnode
            
        def export_graphviz(self):
            
            def get_name(currnode):
                if not (currnode.true is None):                
                    if currnode.dtype=='c':
                        curr_name='{}<={}'.format(self.feature_name_dtype[currnode.feature][0],currnode.threshold)
                    else:
                        curr_name='{} is {}'.format(self.feature_name_dtype[currnode.feature][0],currnode.threshold)
                else:
                    curr_name=currnode.model_description()
                return curr_name
            
            dot = Digraph(comment='Decision Tree')
            que=[self.root]
            currnode=que[0]
            parent_name=get_name(currnode)
            dot.node(parent_name)
            
            while que:
                currnode=que[0]
                parent_name=get_name(currnode)
                if not (que[0].true is None):
                    currnode=que[0].true
                    true_child_name=get_name(currnode)
                    dot.node(true_child_name)
                    dot.edge(parent_name,true_child_name)
                    que.append(currnode)
                    
                    currnode=que[0].false
                    false_child_name=get_name(currnode)                  
                    dot.node(false_child_name)
                    dot.edge(parent_name,false_child_name)
                    que.append(currnode)                            
         
                que.pop(0)
            
            return dot
                                   
        def plot_tree(self):
            self.export_graphviz().render(view=True)
            


        def predict(self,X):
            if self.root==None:
                raise Exception('model not fitted yet')
            else:
                predictions=np.array([])
                if len(np.shape(X))==2 and np.shape(X)[1]==len(self.feature_name_dtype):
                    for instance in X:
                        currnode=self.root
                        while currnode.true!=None: # the current node has branches
                            currnode=currnode.find_child(instance)
                        predictions=np.concatenate((predictions,currnode.predict(instance)),axis=0)  
                    return predictions
                else:
                    raise ValueError('the shape of X should be (sample,feature)')
         
            
    return decision_tree
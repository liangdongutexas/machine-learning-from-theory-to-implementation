import numpy as np
from graphviz import Digraph

def xgboost_regression_decorator(decision_tree,max_num_trees=10):
    class xgboost_regression_tree():
        """
        X=np.array(shape=[samplesize,feature])
        Y=np.array(shape=[samplesize])
        feature_name_type=np.array([(name,'c'/'d')...])
        where 'c' respresents continuous variabl and 'd' represents discrete variable
        """
        def __init__(self,max_num_trees=max_num_trees):
            self.max_num_trees=max_num_trees          


        def fit(self,X,Y,feature_name_dtype):

            # just to store the root of the tree 
            # consistent check whether the shape of inputs are expected:
            condition1= ((len(np.shape(X))==2)
                        and (len(np.shape(feature_name_dtype))==2))         
            condition2= ((np.shape(X)[0]==np.shape(Y)[0]) 
                         and (np.shape(X)[1]==np.shape(feature_name_dtype)[0]))

            if condition1 and condition2:
                if len(np.shape(Y))==1: 
                    Y = np.expand_dims(Y, axis=1)

                # Add Y,g,h as last columns to X so easier to split them together
                XY = np.concatenate((X, Y), axis=1)

            else:
                raise ValueError('The shapes of inputs are not compatible')

            # store data 
            self.XY_mul=np.array([sorted(XY,key=lambda unit: unit[i]) for i in range(X.shape[1])])
            self.trees=[]
            self.feature_name_dtype=feature_name_dtype
            
        
                  
            for i in range(self.max_num_trees):
                tree_i=decision_tree()
                tree_i.gamma=tree_i.gamma/(i+1)
                tree_i.fit(XY_mul=self.XY_mul,feature_name_dtype=feature_name_dtype)
                self.update_Y(tree_i.root)
                self.trees.append(tree_i)
                                        
            # release memory
            self.XY_mul=0


        def predict(self,X):
            result=np.zeros((len(X)))
            for tree_i in self.trees:
                result+=tree_i.predict(X)
                
            return result 
        
        def plot_tree(self):
            dot = Digraph(comment='XGboost Trees')
            for tree_i in self.trees:
                dot.subgraph(tree_i.export_graphviz())
            dot.render(view=True)
        
        def update_Y(self,node):
            if node.true is None:
                y_hat=node.predict(self.XY_mul[node.sample_index][:,:-1])     
                y_hat=np.expand_dims(y_hat,axis=1)
                zeros=np.zeros((len(y_hat),len(self.feature_name_dtype)))
                y_hat=np.concatenate((zeros,y_hat),axis=1)      
                self.XY_mul[node.sample_index]-=y_hat
                node.sampel_index=None
            else:
                node.sampel_index=None
                self.update(node.true)
                self.update(node.false)
                      
    return xgboost_regression_tree
## K-means
The loss function of K-means is:
\begin{align}
J=\sum_{n=1}^{N}\sum_{k=1}^{K}r_{nk}||x_n-u_k||^2,
\end{align}
which is called distortion measure. $r_{nk}$ denotes which cluster the $nth$ data point belongs to. $r_{nk}=1$ if it belongs to the $kth$ cluster otherwise zero.
The K-means algorithm is to find the $\{r,\mu\}$ such that the loss function is minimized.

One possible algorithm to minimize the distortion measure is
1. keep $\mu$ fixed, find the value of ${r_{nk}}$ by assigning the data point to the cluster with least distance.
2. update $\mu_k$. In the two norm case, $\mu_{k}=\frac{\sum_{n}r_{nk}x_n}{\sum_{n}r_{nk}}$.

Next, we realize the algorithm in the following.


```python
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```


```python
class k_means(object):
    def __init__(self,K):
        self.n_cluster=K
    def fit(self,X,epochs):
        X=np.array(X)
        # pick n_cluster data points from X as the starting cluster centers
        self.mu=X[np.random.randint(len(X), size=self.n_cluster)]
        self.r_nk=self.predict(X)
        for epoch in range(epochs):
            # M step
            self.mu=np.divide(np.matmul(np.transpose(self.r_nk),X),\
            np.expand_dims(np.sum(self.r_nk, axis=0),axis=1))
            # E step
            self.r_nk=self.predict(X)

    def predict(self,X):
        X=np.array(X)
        r_nk=np.zeros(shape=(len(X),self.n_cluster))
        for n in range(len(X)):
            d=np.inf
            K=0
            for k in range(self.n_cluster):
                d_new=np.sum(np.power(X[n]-self.mu[k],2))
                if d_new<d:
                    K=k
                    d=d_new
            r_nk[n,K]=1
        return r_nk
```


```python
cluster1=np.random.randn(1000,2)+[0.5,0.5]
cluster2=np.random.randn(1000,2)-[0.5,0.5]
clusters=np.concatenate((cluster1,cluster2),axis=0)
clusters.shape
```




    (2000, 2)




```python
k_clf=k_means(2)
k_clf.fit(clusters,90)
k_clf.mu
k_clf.predict([[0.5,0.5]])
```




    array([[ 0.70628289,  0.77541215],
           [-0.5924069 , -0.73523301]])






    array([[1., 0.]])




```python
k_clf=k_means(2)
k_clf.fit(clusters,90)
k_clf.mu
k_clf.predict([[0.5,0.5]])
```




    array([[-0.5924069 , -0.73523301],
           [ 0.70628289,  0.77541215]])






    array([[0., 1.]])



From the above two examples, we can see the predicted clusters are permuted. This is due to we didn't specify which cluster should be assigned first. So permutation of $\mu$s gives equivalent result.  

## Gaussian Mixture Model
The Gaussian Mixture Model assumed the distribution of data points satisfies the mixture of Gaussians:
\begin{align}
p(x)=\sum_{k=1}^{K}\pi_{k}\mathcal{N}(x|\mu_k,\Sigma_k).
\end{align}
For $p(x)$ to be normalized, we have $\sum_k{\pi_k}=1$.And for $p(x)$ to be positive, we have $\pi_k\geq0$.
Then it is instructive to imagine there is another random variable $Z$ which takes $K$ different values $\{z_k\}$ and has the distribution function $\{\pi_k\}$. $Z$ is also called the latent variable. 

The above expression can be interpreted as
\begin{align}
p(X)=\sum_{Z}p(Z)p(X|Z),
\end{align}
with $p(Z=z_k)=\pi_k$ and $p(x|Z=z_k)=\mathcal{N}(x|\mu_k,\Sigma_k)$.

Given the observed data points $\{x_n\}$, the goal is to maximize the likelihood of observing the data with respect to the unkonwn parameters $\{\pi,\mu,\Sigma\}$ under the constraint $\sum_k{\pi_k}=1$. The logarithm of the likelihood function takes the following form:
\begin{align}
\log p({X}|\pi,\mu,\Sigma)=\sum_{n}\log[\sum_{k}\pi_{k}\mathcal{N}(x_n|\mu_k,\Sigma_k)]+\lambda (\sum_k{\pi_k}-1).
\end{align}
We can then try to find the maximum by taking derivatives of the logarithm of the likelihood function (ll) with respect to each parameters:
\begin{align}
\frac{\partial ll}{\partial \pi_k}=&\sum_{n}\frac{\mathcal{N}(x_n|\mu_k,\Sigma_k)}{\sum_{q}\pi_{q}\mathcal{N}(x_n|\mu_q,\Sigma_q)}+\lambda,\\
\frac{\partial ll}{\partial \mu_k}=&\sum_{n}\frac{\pi_{k} \mathcal{N}(x_n|\mu_k,\Sigma_k)}{\sum_{q}\pi_{q}\mathcal{N}(x_n|\mu_q,\Sigma_q)}(x_n-\mu_{k})\Sigma_{k}^{-1},\\
\frac{\partial ll}{\partial \Sigma_{k}^{-1}}=&\sum_{n}\frac{\pi_{k} \mathcal{N}(x_n|\mu_k,\Sigma_k)}{\sum_{q}\pi_{q}\mathcal{N}(x_n|\mu_q,\Sigma_q)}[-\frac{1}{2}(x_n-\mu_{k})(x_n-\mu_{k})+\frac{1}{2}\Sigma].\\
\end{align} 

$\{\pi,\mu,\Sigma\}$ can be solved by setting the above derivatives to be zero. Noticing that the conditional probability $p(z_k|x)$ is:
\begin{align}
p(z_k|x)=&\frac{p(x|z_k)p(z_k)}{\sum_{q}p(x|z_q)p(z_q)}\\
        =&\frac{\mathcal{N}(x|\mu_{k},\Sigma_{k})\pi_{k}}{\sum_{q}\mathcal{N}(x|\mu_{q},\Sigma_{q})\pi_{q}},
\end{align}
we have:
\begin{align}
\pi_k=&\frac{N_k}{N},\\
\mu_k=&\frac{1}{N_k}\sum_{n}p(z_k|x_n)x_n,\\
\Sigma_{k}=&\frac{1}{N_k}\sum_{n}p(z_k|x_n)(x_n-\mu_{k})(x_n-\mu_{k}),\\
\end{align}
where $N_k=\sum_{n}p(z_k|x_n)$ which has the meaning of number of data points in the sample belonging to the cluster $k$ and $N$ is the sample length with $N=\sum_{k}N_k$. The first expression tells us that the weight of the $k$th cluster is roughly the number of points belonging to that cluster over the total data points. The second expression shows that the center of the $k$th cluster is the probability weighted average of all data points vector. And the last expression shows the variance is the probability weighted average of all data variance. 

The above expression is not an explicit expression of the parameters at maximum of likelihood. Particularly the dependence of $p(z_k|x)$ on the parameters $\{\pi,\mu,\Sigma\}$ makes it hard to solve for an explicit expression. However, the above expression gives an iteration approach:
1. Initialize the parameters $\{\pi,\mu,\Sigma\}$.
2. E step: Calculate the conditional probability $p(z_k|x)$.
3. M step: update the parameters according to the right hand side of the above expressions.
4. Recursively repeat steps 2 and 3 untill some convergence criterion is satisfied.

An issue in the code is how to properly generate a covariance matrix that is (1) symmetric, (2) positive definite, (3) invertible as required in the exponent of Gaussian distribution. 
There is a problem when calculating normal distribution that the normalization factor diverges when the stadard deviation is small.However, we can see that the normal distribution is always multiplied by $\pi$. So the thought is to redefine $\tilde{\pi}_k\equiv\frac{\pi_k}{\sqrt{\det(\Sigma_k)}}$ such that the normalization factor is absorbed. Then the constraint becomes $\sum_k{\tilde{\pi}_k}\sqrt{\det(\Sigma_k)}=1$. Then in terms of the unnormalized normal distribution $\tilde{\mathcal{N}}$, $p(z_k|x)$ still reads:
\begin{align}
p(z_k|x)=&\frac{\tilde{\mathcal{N}}(x|\mu_{k},\Sigma_{k})\tilde{\pi}_{k}}{\sum_{q}\tilde{\mathcal{N}}(x|\mu_{q},\Sigma_{q})\tilde{\pi}_{q}},
\end{align}
Then the new iteration relations become:
\begin{align}
\tilde{\pi}_k=&\frac{N_k}{N\sqrt{\det(\Sigma_k)}},\\
\mu_k=&\frac{1}{N_k}\sum_{n}p(z_k|x_n)x_n,\\
\Sigma_{k}=&\frac{1}{N_k}\sum_{n}p(z_k|x_n)(x_n-\mu_{k})(x_n-\mu_{k}),\\
\end{align}
where $N_k=\sum_{n}p(z_k|x_n)$ which has the meaning of number of data points in the sample belonging to the cluster $k$ and $N$ is the sample length with $N=\sum_{k}N_k$. The first expression tells us that the weight of the $k$th cluster is roughly the number of points belonging to that cluster over the total data points. The second expression shows that the center of the $k$th cluster is the probability weighted average of all data points vector. And the last expression shows the variance is the probability weighted average of all data variance. 

```python
'''random_cov=[]
        for i in range(self.n_cluster):       
            O=rvs(X.shape[1])
            # O=np.random.randn(self.n_cluster,X.shape[1], X.shape[1])
            random_cov.append(np.matmul(O.T,O))
            
        random_cov=np.array(random_cov)
        try:
            self.sigma=random_cov+sample_cov
        except:
            raise ValueError(random_cov.shape,sample_cov.shape)'''
```


```python
class GMM(object):
    def __init__(self,K):
        self.n_cluster=K
        
    def fit(self,X,epochs):
        X=np.array(X)
        # pick n_cluster data points from X as the starting cluster centers
        self.mus=X[np.random.randint(len(X), size=self.n_cluster)]
        
        # generate n_cluster covariance matricies by O^{T}O+(covariance matrix of the sample), where O is a random matrix
        # with each element sampled from standard normal distribution
        sample_cov=np.cov(X.T)  
        self.sigmas=np.array([sample_cov for i in range(self.n_cluster)])

        # generate \pi
        pi=np.random.randn(self.n_cluster)
        pi=np.abs(pi)
        self.pi=pi/np.sum(pi)
                
        for epoch in range(epochs):
        # E process generating p(z|x)
            gm=self.mix_gaussian(X,self.mus,self.sigmas,self.pi)        #shape=[sample,pi_k*Normal_dis_k]
            p_z_given_x=np.divide(gm,np.sum(gm,axis=1)[:,np.newaxis])   #shape=[sample,p(z_k|x_n)]
            N_k=np.sum(p_z_given_x,axis=0)                              #shape=[N_k]
            p_zgx_over_N_k=np.divide(p_z_given_x,N_k)                   #shape=[sample,p(z_k|x_n)/N_k]

        # M process updating self.mu,self.pi,self.sigma
            #update self.pi
            self.pi=np.sum(p_z_given_x,axis=0)/len(X)                 #shape=[pi_k]
                      
            #update self.mu
            self.mus=np.zeros((self.n_cluster,X.shape[1]))
            for p_n,x_n in zip(p_zgx_over_N_k,X):            
                self.mus=np.add(self.mus,np.outer(p_n,x_n))
                
            #use updated mu to update self.sigma
            self.sigmas=np.zeros((self.n_cluster,X.shape[1],X.shape[1]))
            for p_n,x_n in zip(p_zgx_over_N_k,X):
                # sigma_n stores for nth data point [K_cluster,dim(x_n-mu_k),dim(x_n-mu_k)]
                sigma_n=[]
                for p_n_k,mu_k in zip(p_n,self.mus):
                    sigma_n.append(np.outer(np.outer(p_n_k,np.subtract(x_n,mu_k)),np.subtract(x_n,mu_k)))
                self.sigmas=np.add(self.sigmas,np.array(sigma_n))
                
    def predict(self,X):
        gm=self.mix_gaussian(X,self.mus,self.sigmas,self.pi)             #shape=[sample,pi_k*Normal_dis_k]
        p_z_given_x=np.divide(gm,np.sum(gm,axis=1)[:,np.newaxis])      #shape=[sample,p(z_k|x_n)]
        return np.argmax(p_z_given_x,axis=1)
        
    def mix_gaussian(self,X,mus,sigmas,pis):
        #gm has the shape [sample_n,cluster_k], with each element pi_{k}\mathcal{N}(x_n|\mu_k,\Sigma_k)
        X=np.array(X)
        gm=np.array([])
        for mu,sigma,pi in zip(mus,sigmas,pis):
            try:
                power=-np.matmul(np.subtract(mu,X)[:,np.newaxis,:],np.matmul(np.linalg.inv(sigma),np.subtract(mu,X)[:,:,np.newaxis]))             
            except:
                raise ValueError(np.linalg.inv(sigma))
            power=power.reshape(power.shape[0],1)
            try:
                normal=np.exp(power)*pi/np.sqrt((2*np.pi)**X.shape[1]*np.absolute(np.linalg.det(sigma)))
            except:
                raise ValueError(np.linalg.det(sigma))
            if len(gm)==0:
                gm=np.exp(power)*pi
            else:
                gm=np.concatenate((gm,np.exp(power)*pi),axis=1)
        return gm      
```

\begin{align}
p_{nk}=\pi_{k}\mathcal{N}(x_n|\mu_k,\Sigma_k).
\end{align}


```python
#generate two two-dimensional clusters
cluster1=np.random.randn(1000,2)+[1,1]
cluster2=np.random.randn(1000,2)-[1,1]
clusters=np.concatenate((cluster1,cluster2),axis=0)
clusters.shape
```




    (2000, 2)




```python
gmm_clf=GMM(2)
gmm_clf.fit(clusters,epochs=100)
gmm_clf.mus
gmm_clf.predict([[1,1],[-1,-1]])
```




    array([[-1.33197892, -1.25433492],
           [ 0.81082361,  0.73892938]])






    array([1, 0])



## EM in general
The goal is to maximize the likelihood function:
\begin{align}
p(\boldsymbol{X}|\theta)=\sum_{\boldsymbol{Z}}p(\boldsymbol{X},\boldsymbol{Z}|\theta),
\end{align}
which is expressed as the summation over the latent variables. The summation usually causes some difficulty, so we assume $p(\boldsymbol{X},\boldsymbol{Z}|\theta)$ is easy to maximize while $p(\boldsymbol{X}|\theta)$ is not.
We can rewrite it in another form as:
\begin{align}
\log p(\boldsymbol{X}|\theta)=\sum_{\boldsymbol{Z}} q(\boldsymbol{Z}) \log \frac{p(\boldsymbol{X},\boldsymbol{Z}|\theta)}{q(\boldsymbol{Z})}+\sum_{\boldsymbol{Z}} q(\boldsymbol{Z})\log\frac{q(\boldsymbol{Z})}{p(\boldsymbol{Z}|\boldsymbol{X},\theta)}.
\end{align}
The second term is the Kullback-Leibler divergence. For any fixed $q(\boldsymbol{Z})$, we can variate $p(\boldsymbol{Z}|\boldsymbol{X},\theta)$ such that the minimal value is given by $p(\boldsymbol{Z}|\boldsymbol{X},\theta)=q(\boldsymbol{Z})$ and Kullback-Leibler divergence becomes zero. Thus for any $q(\boldsymbol{Z})$ and $p(\boldsymbol{Z}|\boldsymbol{X},\theta)$, $KL(q||p)\geq0$ with zero given when they are equal.

Then the first term gives a lower bond for the expression on the left hand side. In the next, we denote the first term as $\mathcal{L}(q,\theta)$, which is a functional on the distribution $q$ and a function on the parameter $\theta$.
* In the E step, for fixed parameter $\theta_{old}$, we maximize the first term, which is equivalent to minimize the second term and the result is $q(\boldsymbol{Z})=p(\boldsymbol{Z}|\boldsymbol{X},\theta_{old})$. Then the first term becomes $\mathcal{L}(p(\boldsymbol{Z}|\boldsymbol{X},\theta),\theta_{old})=\sum_{\boldsymbol{Z}} p(\boldsymbol{Z}|\boldsymbol{X},\theta_{old}) \log \frac{p(\boldsymbol{X},\boldsymbol{Z}|\theta)}{p(\boldsymbol{Z}|\boldsymbol{X},\theta_{old})}$.
* In the M step, for fixed PDF $q$, we maximize $\mathcal{L}$ by updating $\theta$ and $q$ fixed. This is assumed to be viable.




For a dataset, 
* In the E step, for fixed parameter we calculate 
\begin{align}
q(\{\boldsymbol{Z}\})= & p(\{\boldsymbol{Z}_n\}|\{\boldsymbol{X}_n\},\theta_{old})\\
           \propto & p(\{\boldsymbol{Z}_n\},\{\boldsymbol{X}_n\}|\theta_{old})\\
            \propto& \prod_n p(\boldsymbol{Z}_n,\boldsymbol{X}_n|\theta_{old})\\
                 = & \frac{\prod_n p(\boldsymbol{Z}_n,\boldsymbol{X}_n|\theta_{old})}{\sum_{\{\boldsymbol{Z}_n\}}\prod_n p(\boldsymbol{Z}_n,\boldsymbol{X}_n|\theta_{old})}\\
                 = & \prod_n  \frac{ p(\boldsymbol{Z}_n,\boldsymbol{X}_n|\theta_{old})}{\sum_{\boldsymbol{Z}_n}p(\boldsymbol{Z}_n,\boldsymbol{X}_n|\theta_{old})}
\end{align}

Therefore, it is convenient to define:
$$
q_n(\boldsymbol{Z}_n)\equiv   \frac{ p(\boldsymbol{Z}_n,\boldsymbol{X}_n|\theta_{old})}{\sum_{\boldsymbol{Z}}p(\boldsymbol{Z},\boldsymbol{X}_n|\theta_{old})},
$$
such that: 
$$
q(\{\boldsymbol{Z}\})= \prod_n q_n(\boldsymbol{Z}_n)
$$

Then the complete log likelihood function can be written as:
\begin{align}
E(\log p(\{\boldsymbol{X}\}|\theta))=& \sum_{\{\boldsymbol{Z}\}} q(\{\boldsymbol{Z}\}) \log p(\{\boldsymbol{X}\},\{\boldsymbol{Z}\}|\theta)\\
                                 =& \sum_{\{\boldsymbol{Z}\}} \prod_m q_m(\boldsymbol{Z}_m) \sum_n \log p(\boldsymbol{X}_n,\boldsymbol{Z}_n|\theta)\\
                                 =& \sum_n \sum_{\boldsymbol{Z}}  q_n(\boldsymbol{Z})  \log p(\boldsymbol{X}_n,\boldsymbol{Z}_n|\theta)
\end{align}

In the case of Gaussian Mixture Model, we have
\begin{align}
q_n(\boldsymbol{Z}=\boldsymbol{Z}_k)= &\frac{p(\boldsymbol{Z}=\boldsymbol{Z}_k,\boldsymbol{X}_n)}{\sum_{\boldsymbol{Z}_k}p(\boldsymbol{Z}=\boldsymbol{Z}_k,\boldsymbol{X}_n)}\\
                                    = & \frac{\mathcal{N}(\boldsymbol{X}_n|\mu^{old}_{k},\Sigma^{old}_{k})\pi^{old}_{k}}{\sum_{q}\mathcal{N}(\boldsymbol{X}_n|\mu^{old}_{q},\Sigma^{old}_{q})\pi^{old}_{q}}
\end{align}
which desribes the probability that the given data instance $\boldsymbol{X}_n$ belongs to cluster $k$.


```python

```

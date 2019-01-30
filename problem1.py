import numpy as np
from scipy.stats import multivariate_normal
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 1: EM algorithm for Gaussian Mixture Model (GMM).
    In this problem, you will implement the expectation-maximization problem of a Gaussian Mixture Distribution.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''

#--------------------------
def E_step(X,mu,sigma,PY):
    '''
        E-step: Given the current estimate of model parameters, compute the expected mixture of components on each data point. 
        Input:
            X: the feature matrix of data samples, a numpy matrix of shape n by p
                Here n is the number of samples, p is the number of features
                X[i] is the i-th data sample.
            mu: the list of mean of each Gaussian component, a float matrix of k by p.
                k is the number of components in Gaussian mixture.
                p is the number dimensions in the feature space.
                mu[i] is the mean of the i-th component.
            sigma: the list of co-variance matrix of each Gaussian component, a float tensor of shape k by p by p.
                sigma[i] is the covariance matrix of the i-th component.
            PY: the probability of each component P(Y=i), a float vector of length k.
        Output:
            Y: the estimated label of each data point, a numpy matrix of shape n by k.
                Y[i,j] represents the probability of the i-th data point being generated from j-th Gaussian component.
        Hint: you could use multivariate_normal.pdf() to compute the density funciont of Gaussian ditribution.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n = X.shape[0]
    p = X.shape[1]
    k = mu.shape[0]
    Y = np.mat(np.zeros((n,k)))
    
    for m in range(k):
        Y[:,m] = np.mat(multivariate_normal.pdf(X,mu[m],sigma[m])*PY[m]).T
    
    a = Y.copy()
    for i in range(n):
        for j in range(k):
            Y[i,j] = a[i,j]*1.0/np.sum(a[i])
        
            


    #########################################
    return Y 


#--------------------------
def M_step(X,Y):
    '''
        M-step: Given the current estimate of label distribution, update the parameters of GMM. 
        Input:
            X: the feature matrix of data samples, a numpy matrix of shape n by p
                Here n is the number of samples, p is the number of features
                X[i] is the i-th data sample.
            Y: the estimated label of each data point, a numpy matrix of shape n by k.
                Y[i,j] represents the probability of the i-th data point being generated from j-th Gaussian component.
        Output:
            mu: the list of mean of each Gaussian component, a float matrix of k by p.
                k is the number of components in Gaussian mixture.
                p is the number dimensions in the feature space.
                mu[i] is the mean of the i-th component.
            sigma: the list of co-variance matrix of each Gaussian component, a float tensor of shape k by p by p.
                sigma[i] is the covariance matrix of the i-th component.
            PY: the probability of each component P(Y=i), a float vector of length k.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n = X.shape[0]
    k = Y.shape[1]
    p = X.shape[1]
    mu = np.zeros((k,p))
    for i in range(k):
        for j in range(p):
            mu[i,j] = np.dot(X[:,j],Y[:,i])*1.0/np.sum(Y[:,i])
    
    sigma = np.zeros((k,p,p))
    for i in range(k):
        s = np.zeros((p,p))
        for j in range(n):
            a = X[j]-mu[i]
            s += np.outer(a,a)*Y[j,i]
        sigma[i] = s*1.0/np.sum(Y[:,i])
    
    PY = np.zeros(k)
    for m in range(k):
        PY[m] =np.sum(Y[:,m])*1.0/np.sum(Y)





    #########################################
    return mu,sigma,PY



#--------------------------
def EM(X,k=2,num_iter=10):
    '''
        EM: Given a set of data samples, estimate the parameters and label assignments of GMM. 
        Input:
            X: the feature matrix of data samples, a numpy matrix of shape n by p
                Here n is the number of samples, p is the number of features
                X[i] is the i-th data sample.
            k: the number of components in Gaussian mixture, an integer scalar.
            num_iter: the number EM iterations, an integer scalar.
        Output:
            Y: the estimated label of each data point, a numpy matrix of shape n by k.
                Y[i,j] represents the probability of the i-th data point being generated from j-th Gaussian component.
            mu: the list of mean of each Gaussian component, a float matrix of k by p.
                p is the number dimensions in the feature space.
                mu[i] is the mean of the i-th component.
            sigma: the list of co-variance matrix of each Gaussian component, a float tensor of shape k by p by p.
                sigma[i] is the covariance matrix of the i-th component.
            PY: the probability of each component P(Y=i), a float vector of length k.
    '''
    # initialization (for testing purpose, we use first k samples as the initial value of mu) 
    n,p = X.shape
    mu = X[:k]

    sigma = np.zeros((k,p,p))
    for i in range(k):
        sigma[i] = np.eye(p)
    PY = np.ones(k)/k

    #########################################
    ## INSERT YOUR CODE HERE
    for i in range(num_iter):
        Y = E_step(X,mu,sigma,PY)
        mu,sigma,PY = M_step(X,Y) 
   






    #########################################
    return Y,mu,sigma,PY


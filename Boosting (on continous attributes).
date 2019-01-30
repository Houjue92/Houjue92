import math
import numpy as np
from problem2 import DT,Node
from problem1 import Tree 
#-------------------------------------------------------------------------
'''
    Problem 5: Boosting (on continous attributes). 
               We will implement AdaBoost algorithm in this problem.
    You could test the correctness of your code by typing `nosetests -v test5.py` in the terminal.
'''

#-----------------------------------------------
class DS(DT):
    '''
        Decision Stump (with contineous attributes) for Boosting.
        Decision Stump is also called 1-level decision tree.
        Different from other decision trees, a decision stump can have at most one level of child nodes.
        In order to be used by boosting, here we assume that the data instances are weighted.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y, D):
        '''
            Compute the entropy of the weighted instances.
            Input:
                Y: a list of labels of the instances, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the entropy of the weighted samples, a float scalar
            Hint: you could use np.unique(). 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        dict = {}
        for i in range(len(Y)):
            if Y[i] in dict:
                dict[Y[i]].append(i)
            else:
                dict[Y[i]] = []
                dict[Y[i]].append(i)
        
        y = np.unique(Y)
        l = []
        for j in y:
            l.append(sum(D[dict[j]]))
        
        e = 0
        for k in range(len(l)):
            if l[k] == 0:
                E = 0
            elif l[k] > 0:
                E = -l[k]*(math.log(l[k])/math.log(2))
            e += E
              





        #########################################
        return e 
            
    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X,D):
        '''
            Compute the conditional entropy of y given x on weighted instances
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                ce: the weighted conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        dict = {}
        for i in range(len(X)):
            if X[i] in dict:
                dict[X[i]].append(i)
            else:
                dict[X[i]] = []
                dict[X[i]].append(i)
        
        x = np.unique(X)
        ce = 0
        for j in x:
            p = sum(D[dict[j]])
            if p == 0:
                ce += 0
            else:
                y = Y[dict[j]]
                d = np.array(D[dict[j]])*(sum(D)/p)
                q = DS.entropy(y,d)
                ce += p*q




    
        #########################################
        return ce 

    #--------------------------
    @staticmethod
    def information_gain(Y,X,D):
        '''
            Compute the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                g: the weighted information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g = DS.entropy(Y, D) - DS.conditional_entropy(Y,X,D)



    
        #########################################
        return g

    #--------------------------
    @staticmethod
    def best_threshold(X,Y,D):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. The data instances are weighted. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
            Output:
                th: the best threhold, a float scalar. 
                g: the weighted information gain by using the best threhold, a float scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        cp = DT.cutting_points(X,Y)
        if cp[0] == - float('inf'):
            th = - float('inf')
            g = -1.
        else:
            l =[]
            for i in range(len(cp)):
                t = X.copy()
                for j in range(len(t)):
                    if t[j] >= cp[i]:
                        t[j] = 1
                    else:
                        t[j] = 0
                    
                ig = DS.information_gain(Y,t,D)
                l.append(ig)
        
            g = max(l)
            b = np.argmax(l)
            th = cp[b]





        #########################################
        return th,g 
     
    #--------------------------
    def best_attribute(self,X,Y,D):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float). The data instances are weighted.
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        l1 = []
        l2 = []
        for i in range(X.shape[0]):
            th,g = DS.best_threshold(X[i],Y,D)
            l1.append(th)
            l2.append(g)
        
        i = np.argmax(l2)
        th = l1[i]





    
        #########################################
        return i, th
             
    #--------------------------
    @staticmethod
    def most_common(Y,D):
        '''
            Get the most-common label from the list Y. The instances are weighted.
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
                D: the weights of instances, a numpy float vector of length n
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        dict = {}
        for i in range(len(Y)):
            if Y[i] in dict:
                dict[Y[i]].append(i)
            else:
                dict[Y[i]] = []
                dict[Y[i]].append(i)
        
        l1 = []
        l2 = []
        for key,value in dict.iteritems():
            l1.append(dict[key])
            l2.append(sum(D[dict[key]]))
        i = np.argmax(l2)
        y = [k for (k, v) in dict.iteritems() if v == l1[i]][0]



        #########################################
        return y
 

    #--------------------------
    def build_tree(self, X,Y,D):
        '''
            build decision stump by overwritting the build_tree function in DT class.
            Instead of building tree nodes recursively in DT, here we only build at most one level of children nodes.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Return:
                t: the root node of the decision stump. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Node(X,Y)
        t.p = DS.most_common(Y,D)
        # if Condition 1 or 2 holds, stop splitting 
        if Tree.stop1(Y) == True or Tree.stop2(X) == True:
            t.isleaf = True
            return t 


        # find the best attribute to split
        ds = DS()
        t.i,t.th= ds.best_attribute(X,Y,D)



        # configure each child node
        t.C1,t.C2 = DT.split(t.X,t.Y,t.i,t.th)
        
        x = X[t.i].copy()
        for j in range(X.shape[1]):
            if x[j] >= t.th:
                x[j] = 1.
            else:
                x[j] = 0.
        X1 = X.copy()
        X1[t.i] = x
        dict = {}
        for l in range(X1.shape[1]):
            if X1[t.i,l] in dict:
                dict[X1[t.i,l]].append(l)
            else:
                dict[X1[t.i,l]] = []
                dict[X1[t.i,l]].append(l)
        
        C = {}
        for key, value in dict.iteritems():
            d = D[value]
            C[key] = d 
        t.C1.isleaf = True
        t.C1.p = DS.most_common(t.C1.Y,C[0])
        t.C2.isleaf = True
        t.C2.p = DS.most_common(t.C2.Y,C[1])
        

    
        #########################################
        return t
    
 

#-----------------------------------------------
class AB(DS):
    '''
        AdaBoost algorithm (with contineous attributes).
    '''

    #--------------------------
    @staticmethod
    def weighted_error_rate(Y,Y_,D):
        '''
            Compute the weighted error rate of a decision on a dataset. 
            Input:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the weighted error rate of the decision stump
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        l = []
        for i in range(len(Y)):
            if Y[i] != Y_[i]:
                l.append(i)
        
        e = sum(D[l])
                

        #########################################
        return e

    #--------------------------
    @staticmethod
    def compute_alpha(e):
        '''
            Compute the weight a decision stump based upon weighted error rate.
            Input:
                e: the weighted error rate of a decision stump
            Output:
                a: (alpha) the weight of the decision stump, a float scalar.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        if e == 0.:
            a = 600
        elif e ==1.:
            a = -600
        else:
            a = 0.5*np.log((1-e)/e)




        #########################################
        return a

    #--------------------------
    @staticmethod
    def update_D(D,a,Y,Y_):
        '''
            update the weight the data instances 
            Input:
                D: the current weights of instances, a numpy float vector of length n
                a: (alpha) the weight of the decision stump, a float scalar.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels by the decision stump, a numpy array of length n. Each element can be int/float/string.
            Output:
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        l1 = []
        l2 = []
        for i in range(len(Y)):
            if Y[i] != Y_[i]:
                l1.append(i)
            else:
                l2.append(i)   
        Z = sum(np.array(D[l1])*math.exp(a)) + sum(np.array(D[l2])*math.exp(-a)) 
        for m in l1:
            D[m] = D[m]*math.exp(a)/Z
        for n in l2:
            D[n] = D[n]*math.exp(-a)/Z




        #########################################
        return D

    #--------------------------
    @staticmethod
    def step(X,Y,D):
        '''
            Compute one step of Boosting.  
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the current weights of instances, a numpy float vector of length n
            Output:
                t:  the root node of a decision stump trained in this step
                a: (alpha) the weight of the decision stump, a float scalar.
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        d = DS()
        t = d.build_tree(X,Y,D)
        
        if t.isleaf == True:
            Y1 = []
            for i in range(len(Y)):
                Y1.append(t.p)
            e = AB.weighted_error_rate(Y,Y1,D)
            a = AB.compute_alpha(e)
            D = AB.update_D(D,a,Y,Y1)
        else:
            for i in range(len(t.C1.Y)):
                Y[i] = t.C1.p 
            for j in range(len(t.C2.Y)):
                Y[j] = t.C2.p
            Y_ = np.concatenate((t.C1.Y,t.C2.Y),axis = 0)
            e = AB.weighted_error_rate(Y,Y_,D)
            a = AB.compute_alpha(e)
            D = AB.update_D(D,a,Y,Y_)


        #########################################
        return t,a,D

    
    #--------------------------
    @staticmethod
    def inference(x,T,A):
        '''
            Given a bagging ensemble of decision trees and one data instance, infer the label of the instance. 
            Input:
                x: the attribute vector of a data instance, a numpy vectr of shape p.
                   Each attribute value can be int/float
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                y: the class label, a scalar of int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        l = []
        for t in T:
            while(not t.isleaf):
                if x[t.i] < t.th:
                    t = t.C1
                else:
                    t = t.C2
            l.append(t.p)
        
        dict = {}
        for j in range(len(l)):
            if l[j] in dict:
                dict[l[j]].append(A[j])
            else:
                dict[l[j]] = []
                dict[l[j]].append(A[j])
        
        l1 = []
        C = {}
        for key,value in dict.items():
            w = sum(dict[key])
            l1.append(w)
            C[key] = w
        mw = max(l1)
        vote = l[0]
        for key,value in C.items():
            if C[key] == mw:
                vote = key      
        y = vote

    
        #########################################
        return y
 

    #--------------------------
    @staticmethod
    def predict(X,T,A):
        '''
            Given an AdaBoost and a dataset, predict the labels on the dataset. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        l = []
        for i in range(X.shape[1]):
            x = X[:,i]
            y = AB.inference(x,T,A)
            l.append(y)
        Y = np.array(l)



 
        #########################################
        return Y 
 

    #--------------------------
    @staticmethod
    def train(X,Y,n_tree=10):
        '''
            train adaboost.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                n_tree: the number of trees in the ensemble, an integer scalar
            Output:
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE


        # initialize weight as 1/n
        D = np.array([1./len(Y)]*len(Y))

        # iteratively build decision stumps
        ab = AB()
        l1 = []
        l2 = []
        for i in range(n_tree):
            t,a,D = ab.step(X,Y,D)
            l1.append(t)
            l2.append(a)
        T = l1
        A = np.array(l2)


        #########################################
        return T, A
   



 

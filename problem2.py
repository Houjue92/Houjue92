import math
import csv
import numpy as np
from problem1 import Tree
#-------------------------------------------------------------------------
'''
    Problem 2: Decision Tree (with continous attributes)
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''

#--------------------------
class Node:
    '''
        Decision Tree Node (with continous attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            th: the threshold on the attribute, a float scalar.
            C1: the child node for values smaller than threshold
            C2: the child node for values larger than threshold
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X=None,Y=None, i=None,th=None,C1=None, C2=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.th = th 
        self.C1= C1
        self.C2= C2
        self.isleaf = isleaf
        self.p = p


#-----------------------------------------------
class DT(Tree):
    '''
        Decision Tree (with contineous attributes)
        Hint: DT is a subclass of Tree class in problem1. So you can reuse and overwrite the code in problem 1.
    '''

    #--------------------------
    @staticmethod
    def cutting_points(X,Y):
        '''
            Find all possible cutting points in the continous attribute of X. 
            (1) sort unique attribute values in X, like, x1, x2, ..., xn
            (2) consider splitting points of form (xi + x(i+1))/2 
            (3) only consider splitting between instances of different classes
            (4) if there is no candidate cutting point above, use -inf as a default cutting point.
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                cp: the list of  potential cutting points, a float numpy vector. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        dict = {}
        for i in range(len(X)):
            if X[i] in dict:
                dict[X[i]].append(Y[i])
            else:
                dict[X[i]] = []
                dict[X[i]].append(Y[i])
                
            
        x = np.sort(np.unique(X))
        cp = []
        for i in range(len(x)-1):
            c = float(x[i] + x[i+1])/2
            s1 = set(dict[x[i]])
            s2 = set(dict[x[i+1]])
            if len(s1) == len(s2) and len(s1) ==2:
                cp.append(c)
            if len(s1) == 1 and len(s2) ==1 and  list(s1)[0] != list(s2)[0] :
                cp.append(c)
            if len(s1) != len(s2):
                cp.append(c) 
        cp = np.array(cp)
        if len(cp) == 0:
            cp = np.array([- float('inf')])
            





        #########################################
        return cp
    
    #--------------------------
    @staticmethod
    def best_threshold(X,Y):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                th: the best threhold, a float scalar. 
                g: the information gain by using the best threhold, a float scalar. 
            Hint: you can reuse your code in problem 1.
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
                    
                ig = Tree.information_gain(Y,t)
                l.append(ig)
        
            g = max(l)
            b = np.argmax(l)
            th = cp[b]
        
       
        



        #########################################
        return th,g 
    
    
    #--------------------------
    def best_attribute(self,X,Y):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float).
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        l1 = []
        l2 = []
        for i in range(X.shape[0]):
            th,g = DT.best_threshold(X[i],Y)
            l1.append(th)
            l2.append(g)
        
        i = np.argmax(l2)
        th = l1[i]




 
        #########################################
        return i, th
    


        
    #--------------------------
    @staticmethod
    def split(X,Y,i,th):
        '''
            Split the node based upon the i-th attribute and its threshold.
            (1) split the matrix X based upon the values in i-th attribute and threshold
            (2) split the labels Y 
            (3) build children nodes by assigning a submatrix of X and Y to each node
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C1: the child node for values smaller than threshold
                C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        #th,g = DT.best_threshold(X[i],Y)
        t = X[i].copy()
        for j in range(X.shape[1]):
            if t[j] >= th:
                t[j] = 1.
            else:
                t[j] = 0.
        
        X1 = X.copy()
        X1[i] = t
        dict = {}
        for l in range(X1.shape[1]):
            if X1[i,l] in dict:
                dict[X1[i,l]].append(l)
            else:
                dict[X1[i,l]] = []
                dict[X1[i,l]].append(l)
        
        C = {}
        for key, value in dict.iteritems():
            X1 = X[:,value]
            Y1 = Y[value]
            C[key] = Node(X1,Y1)
        
        C1 = C[0.]
        C2 = C[1.]
            
        
   


 
        #########################################
        return C1, C2
    
    
    
    #--------------------------
    def build_tree(self, t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C1: the child node for values smaller than threshold
                t.C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t.p = Tree.most_common(t.Y)
        # if Condition 1 or 2 holds, stop recursion 
        if Tree.stop1(t.Y) == True or Tree.stop2(t.X) == True:
            t.isleaf = True
            return 


 
        # find the best attribute to split
        dt = DT()
        t.i,t.th= dt.best_attribute(t.X,t.Y)


        # recursively build subtree on each child node
        t.C1,t.C2 = DT.split(t.X,t.Y,t.i,t.th)
        dt.build_tree(t.C1)
        dt.build_tree(t.C2)

 
        #########################################
    
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        if t.isleaf == True:
            y = t.p
            
        
        else:
            if x[t.i] < t.th:
                    y = DT.inference(t.C1,x)
            else:
                y = DT.inference(t.C2,x) 





        #########################################
        return y
    
    
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        l = []
        for i in range(X.shape[1]):
            x = X[:,i]
            y = DT.inference(t,x)
            l.append(y)
        Y = np.array(l)






        #########################################
        return Y
    
    
    
    #--------------------------
    def train(self, X, Y):
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        tNode = Node(X = X, Y = Y)
        dt = DT()
        dt.build_tree(tNode)
        t = tNode
   


 
        #########################################
        return t


    #--------------------------
    @staticmethod
    def load_dataset(filename='data2.csv'):
        '''
            Load dataset 2 from the CSV file: data2.csv. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        reader = csv.reader(open("data2.csv", "rb"), delimiter=",")
        x = list(reader)
        a = np.array(x) 
        b = a.copy()
        A = a[1:,1:].T
        X = np.array(A, dtype=float)
        Y = b[1:,0].T


 
        #########################################
        return X,Y





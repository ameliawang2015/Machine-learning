# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:23:17 2018

@author: dhana
"""

# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import numpy.matlib as nm
import matplotlib.pyplot as plt
from __future__ import division
from sympy import *

def single_layer_classification_hw():

    # load data
    X, y = load_data()
    global M
    M = 4     # number of hidden units

    # perform gradient descent to fit tanh basis sum
    b,w,c,V = tanh_softmax(X.T,y,M)

    # plot resulting fit
    plot_separator(b,w,c,V,X,y)
    plt.show()


### gradient descent for single layer tanh nn ###
def tanh_softmax(X,y,M):

    y = np.reshape(y,(np.size(y),1))
    print("shape y",y.shape)

    # initializations
    N = np.shape(X)[0]
    P = np.shape(X)[1]
    print("shape P", P)
    b = np.random.randn()
   
    w = np.random.randn(M,1)
    print("shape w", w.shape)
    c = np.random.randn(M,1)
    print("shape c", c.shape)
    V = (np.random.randn(N,M))
    print("shape V", V.shape)
    l_P = np.ones((P,1))
    
    print("shape of X",X.shape)

    # stoppers
    max_its = 10000
    grad = 1
    count = 1
    #print("shape of F",F.shape)
    F_1= obj1(c,V,X)
        #x_1= np.tanh(b + np.dot(F.T,w))
    #print("F_1",F_1.shape)
    F = obj(c,V,X)
    #q=sigmoid(-y* (b + np.sum(np.dot(w.T,F),axis=1))) 
    q=sigmoid(-y* (b*l_P + np.dot(F.T,w)))
    qq=nm.repmat(q,1,M)
    yy=nm.repmat(y,1,M)
    ww=nm.repmat(w.T,N,1)
    tn=F.T
    sn= F_1.T
    
    ### main ###
    while (count <= max_its) & (np.linalg.norm(grad) > 1e-5):

       
        #s_np=1/(np.cosh(b + np.dot(F.T,w))**2)
        
        #print("s_np",s_np.shape)
        
       # print("q", q.shape)
        # calculate gradients
       
        #grad_b = -np.dot(l_P.T,q*y)
        grad_b= -np.dot(l_P.T,q*y)
        grad_w = -(np.dot(l_P.T,qq*F.T*yy)).T
        
        grad_c =- (np.dot(l_P.T,qq*sn*yy)).T * w 
        
        
        #grad_V = -np.dot(X,np.multiply(q*F_1.T,np.dot(y,w.T)))
        grad_V= np.dot(-X,(qq*sn*yy)) *ww
            
        alpha = 1e-2
        # take gradient steps
        b = b - alpha*grad_b
        w = w - alpha*grad_w
        c = c - alpha*grad_c
        V = V - alpha*grad_V

        # update stoppers
        count = count + 1

    return b, w, c ,V


### load data
def load_data():

    data = np.array(np.genfromtxt('genreg_data.csv', delimiter=','))
    A = data[:,0:-1]
    b = data[:,-1]

    # plot data
    ind = np.nonzero(b==1)[0]
    plt.plot(A[ind,0],A[ind,1],'ro')
    ind = np.nonzero(b==-1)[0]
    plt.plot(A[ind,0],A[ind,1],'bo')
    plt.hold(True)

    return A,b



def sigmoid(z):
    return 1/(1+np.exp(-z))


# plot the seprator + surface
def plot_separator(b,w,c,V,X,y):

    s = np.arange(-1,1,.01)
    s1, s2 = np.meshgrid(s,s)

    s1 = np.reshape(s1,(np.size(s1),1))
    s2 = np.reshape(s2,(np.size(s2),1))
    g = np.zeros((np.size(s1),1))

    t = np.zeros((2,1))
    for i in np.arange(0,np.size(s1)):
        t[0] = s1[i]
        t[1] = s2[i]
        F = obj(c,V,t)
        g[i] = np.tanh(b + np.dot(F.T,w))

    s1 = np.reshape(s1,(np.size(s),np.size(s)))
    s2 = np.reshape(s2,(np.size(s),np.size(s)))
    g = np.reshape(g,(np.size(s),np.size(s)))

    # plot contour in original space
    plt.contour(s1,s2,g,1,color = 'k')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.hold(True)


def obj(z,H,A):
    F = np.zeros((M,np.shape(A)[1]))
    for p in np.arange(0,np.shape(A)[1]):
        F[:,p] = np.ravel(np.tanh(z + np.dot(H.T,np.reshape(A[:,p],(np.shape(A)[0],1)))))

    return F

def obj1(z,H,A):
    F_1 = np.zeros((M,np.shape(A)[1]))
    for p in np.arange(0,np.shape(A)[1]):
        F_1[:,p] = np.ravel(1/(np.cosh(z + np.dot(H.T,np.reshape(A[:,p],(np.shape(A)[0],1)))))**2)

    return F_1

single_layer_classification_hw()
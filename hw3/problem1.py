## Problem 1 ##

## part a ##

import numpy as np
import mltools as ml
import matplotlib.pyplot as plt

iris = np.genfromtxt("data/iris.txt",delimiter=None)
X, Y = iris[:,0:2], iris[:,-1] # get first two features & target
X,Y  = ml.shuffleData(X,Y) # reorder randomly (important later)
X,_ = ml.rescale(X) # works much better on rescaled data

XA, YA = X[Y<2,:], Y[Y<2] # get class 0 vs 1
XB, YB = X[Y>0,:], Y[Y>0] # get class 1 vs 2'

X0, Y0 = X[Y==0, :], Y[Y==0] #class 0
X1, Y1 = X[Y==1, :], Y[Y==1] #class 1
X2, Y2 = X[Y==2, :], Y[Y==2] #class 2

plt.scatter(X0[:,0], X0[:,1], c='Blue')
plt.scatter(X1[:,0], X1[:,1], c="Red")
plt.close()

plt.scatter(X1[:,0], X1[:,1], c='Blue')
plt.scatter(X2[:,0], X2[:,1], c="Red")
plt.close()

## part b ##

from logisticClassify2 import *
learner = logisticClassify2(); # create "blank" learner
learner.classes = np.unique(YA) # define class labels using YA or YB
wts = np.array([.5, 1, -.25]); # TODO: fill in values
learner.theta = wts;                    # set the learner's parameters
learner.plotBoundary(XA, YA)
plt.close()
learner.plotBoundary(XB, YB)
plt.close()

## part c ##

print "Error Rate (dataset A): "
print np.mean(YA != learner.predict(XA)) ## equivalent to expected 0.0505

print "Error Rate (dataset B): "
print np.mean(YB != learner.predict(XB)) ## .5454

## part d ##
learner.classes = np.unique(YA)
ml.plotClassify2D(learner, XA, YA, axis=plt)
plt.close()

learner.classes = np.unique(YB)
ml.plotClassify2D(learner, XB, YB, axis=plt)
plt.close()
## resulting decision boundaries matches ones computated analytically
## mostly for XA because the error rate for XB really bad .54


## part e ##






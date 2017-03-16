## Problem 1 ##

import numpy as np

import matplotlib.pyplot as plt

np.random.seed(0)
iris = np.genfromtxt("data/iris.txt",delimiter=None) # load the text file

Y = iris[:,-1] # target value is the last column

X = iris[:,0:-1] # features are the other columns

## a ##
print X.shape[1] ## number of features
print X.shape[0] ## number of data points

## b ##
X1 = X[:,0] ## extract first feature
Bins = np.linspace(4,8,17) ## use explicit bin locations
plt.hist(X1, bins=Bins) ## generate plot
plt.close()

X2 = X[:,1]
Bins = np.linspace(1,5,17)
plt.hist(X2, bins=Bins)
plt.close()

X3 = X[:,2]
Bins = np.linspace(0,8,17)
plt.hist(X3, bins=Bins)
plt.close()

X4 = X[:,3]
Bins = np.linspace(0,4,17)
plt.hist(X4, bins=Bins)
plt.close()

## c ##
print np.mean(X, axis=0) ## mean of each feature
print np.std(X, axis=0) ## standard deviation of each feature

## d ##
def change_colors(a, b):
    plt.plot(X[Y == 0, a], X[Y == 0, b], 'o', color='b')
    plt.plot(X[Y == 1, a], X[Y == 1, b], 'o', color='r')
    plt.plot(X[Y == 2, a], X[Y == 2, b], 'o', color='g')


scat = plt.plot(X[:,0], X[:,1], 'b.') ## plot data points as blue dots
change_colors(0, 1)
plt.close()

plt.plot(X[:,0], X[:,2], 'b.') ## plot data points as blue dots
change_colors(0, 2)
plt.close()

plt.plot(X[:,0], X[:,3], 'b.') ## plot data points as blue dots
change_colors(0, 3)
plt.close()
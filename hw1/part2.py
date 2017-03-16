## Problem 2 ##

## a ##

import numpy as np
import matplotlib.pyplot as plt

iris = np.genfromtxt("data/iris.txt",delimiter=None) # load the data

Y = iris[:,-1]

X = iris[:,0:2] #first two features of X

# Note: indexing with ":" indicates all values (in this case, all rows);
# indexing with a value ("0", "1", "-1", etc.) extracts only that one value (here, columns);
# indexing rows/columns with a range ("1:-1") extracts any row/column in that range.

import mltools as ml

# We'll use some data manipulation routines in the provided class code
# Make sure the "mltools" directory is in a directory on your Python path, e.g.,
# export PYTHONPATH=${PYTHONPATH}:/path/to/parent/dir
# or add it to your path inside Python:

# import sys

# sys.path.append('/path/to/parent/dir/');

X,Y = ml.shuffleData(X,Y); # shuffle data randomly

# (This is a good idea in case your data are ordered in some pathological way,
# as the Iris data are)

Xtr,Xva,Ytr,Yva = ml.splitData(X,Y, 0.75); # split data into 75/25 train/validation

for K in [1, 5, 10, 50]: ## visualize classification boundary
    knn = ml.knn.knnClassify() # create the object and train it
    knn.train(Xtr, Ytr, K) # where K is an integer, e.g. 1 for nearest neighbor prediction
    YvaHat = knn.predict(Xva) # get estimates of y for each data point in Xva

    ml.plotClassify2D( knn, Xtr, Ytr, axis=plt ) # make 2D classification plot with data (Xtr,Ytr)
    plt.close()

## b ##

K=[1,2,5,10,50,100,200]
errTrain = []
errValidation = []
for i,k in enumerate(K):
    learner = ml.knn.knnClassify() ## train
    learner.train(Xtr[:,0:2], Ytr, k)
    Yhat = learner.predict(Xtr[:,0:2]) #predict
    print Yhat
    errTrain.append(learner.err(Xtr[:,0:2], Ytr)) # TODO: to count what fraction of predictions are wrong
    learner2 = ml.knn.knnClassify()  ## train
    learner2.train(Xva[:, 0:2], Yva, k)
    Yhat2 = learner2.predict(Xtr[:, 0:2])  # predict
    errValidation.append(learner2.err(Xva[:,0:2], Yva)) #TODO: repeat prediction / error evaluation for validation data
print errTrain
plt.semilogx(K, errTrain, color='red')
plt.semilogx(K, errValidation, color='green') #TODO: " " to average and plot results on semi-log scale
plt.close()
## Problem 1 ##

## part a ##

import numpy as np
import mltools as ml
import matplotlib.pyplot as plt

data = np.genfromtxt("data/curve80.txt",delimiter=None) # load the text file

##split data into 75% training data and 25% testing data
## base predictions off training and test resulting prediction using testing data to verify correctness and minimize error

X = data[:,0]
X = X[:,np.newaxis] # code expects shape (M,N) so make sure it's 2-dimensional
Y = data[:,1] # doesn't matter for Y
Xtr,Xte,Ytr,Yte = ml.splitData(X,Y,0.75) # split data set 75/25

## part b ##

## train linear regress model and measure error against training and test MSE

lr = ml.linear.linearRegress( Xtr, Ytr ) # create and train model
xs = np.linspace(0,10,200) # densely sample possible x-values
xs = xs[:,np.newaxis] # force "xs" to be an Mx1 matrix (expected by our code)
ys = lr.predict( xs ) # make predictions at xs

print "Theta: "
print lr.theta[0][0]
print
plt.scatter(x=Xtr, y=Ytr)
plt.plot(xs, ys)
plt.close()


##e1 = [[x] for x in Ytr] - Xtr.dot( lr.theta )
##trainingMSE = e1.T.dot(e1)/ np.mean(e1**2)
print "Training MSE: "
##print trainingMSE
print lr.mse(Xtr, Ytr)
print

##e2 = [[x] for x in Yte] - Xte.dot( lr.theta )
##testMSE = e2.T.dot(e2)/ np.mean(e2**2)
print "Test MSE: "
##print testMSE
print lr.mse(Xte, Yte)
print

## part c ##
## creates linear regression and measures MSE based on predictions
## adds degrees to make function more dimensional

d = [1, 3, 5, 7, 10, 18]
errTest = []
errTrain = []

for degree in d:

    ## expand and rescale features ##

    # Create polynomial features up to "degree"; don't create constant feature
    # (the linear regression learner will add the constant feature automatically)
    XtrP = ml.transforms.fpoly(Xtr, degree, bias=False)
    # Rescale the data matrix so that the features have similar ranges / variance
    XtrP,params = ml.transforms.rescale(XtrP)
    # "params" returns the transformation parameters (shift & scale)
    # Then we can train the model on the scaled feature matrix:
    lr = ml.linear.linearRegress( XtrP, Ytr ) # create and train model
    # Now, apply the same polynomial expansion & scaling transformation to Xtest:
    XteP,_ = ml.transforms.rescale( ml.transforms.fpoly(Xte,degree,False), params)

    Phi = lambda X: ml.transforms.rescale(ml.transforms.fpoly(X, degree, False), params)[0]


    ## make a set of points that are sorted by x, and are closely, linearly spaced
    xs = np.linspace(0, 10, 200)  # densely sample possible x-values
    xs = xs[:, np.newaxis]  # force "xs" to be an Mx1 matrix (expected by our code)


    Xtr3 = Phi(xs)              ##transform the data
    Ytr3 = lr.predict(Xtr3)     ##predict data (this is what we will graph because x is sorted and will display correctly)

    YhatTrain = lr.predict(XtrP)  # predict on training data
    YhatTest = lr.predict(XteP)  # predict on test data
    # etc.

    #e1 = [[x] for x in Ytr] - Xtr.dot(lr.theta)
    #trainingMSE = e1.T.dot(e1) / np.mean(e1 ** 2)
    errTest.append(lr.mse(XteP, Yte))
    errTrain.append(lr.mse(XtrP, Ytr))

    plt.plot(Xte, Yte, "r.")
    ax = plt.axis()
    plt.plot(Xtr, Ytr, "g.")
    plt.plot(xs, Ytr3)
    plt.axis(ax)
    plt.show()
    plt.close()

plt.semilogy(d, errTrain, color='blue')
plt.semilogy(d, errTest, color='green') #TODO: " " to average and plot results on semi-log scale
plt.show()
plt.close()

## Problem 2 ##
## grabs mean MSE error of linear regression based on cross validation folds

J = []
nFolds = 5
E = []

for degree in d:
    errTi = []
    errCrossTest = []
    for iFold in range(nFolds):
        XtrP = ml.transforms.fpoly(Xtr, degree, bias=False)
        XteP = ml.transforms.fpoly(Xte, degree, bias=False)
        Xti, Xvi, Yti, Yvi = ml.crossValidate(XtrP, Ytr, nFolds, iFold)
        ##XtiP = ml.transforms.fpoly(Xti, degree, bias=False)
        ##XtiP,params = ml.transforms.rescale(Xti)
        # Rescale the data matrix so that the features have similar ranges / variance
        learner = ml.linear.linearRegress(Xti,Yti)
        errTi.append(learner.mse(Xvi, Yvi))
        errCrossTest.append(learner.mse(XteP, Yte))
    J.append(np.mean(errTi))
    E.append(np.mean(errCrossTest))
plt.show()
plt.semilogy(d, J, color='blue') ## cross validation error
plt.semilogy(d, E, color='red') ## testing error
plt.close()

print "Degrees: "
print d
print

print "Cross Validation Error by Degrees: "
print J
print

print "Degree 5 Testing Error by Degrees: "
print E
## Problem 2 ##

## part a ##

import numpy as np
import mltools as ml
import matplotlib.pyplot as plt

# ##training data
# Xt = np.genfromtxt("X_train.txt",delimiter=None)[:10000] # load the text file
# Yt = np.genfromtxt("Y_train.txt",delimiter=None)[:10000] # load the text file
#
# #validation data
# Xv = np.genfromtxt("X_train.txt",delimiter=None)[10001:20000] # load the text file
# Yv = np.genfromtxt("Y_train.txt",delimiter=None)[10001:20000] # load the text file

# ## part b ##
#
# learner = ml.dtree.treeClassify(Xt,Yt, maxDepth=50)
#
# tError = learner.err(Xt, Yt)
# vError = learner.err(Xv, Yv)
#
# print x
# print "Training Error: "
# print tError
# print "Validation Error: "
# print vError
#
# ## part c ##
#
# for x in range(1,51):
#     learner = ml.dtree.treeClassify(Xt,Yt, maxDepth=x)
#
#     tError = learner.err(Xt, Yt)
#     vError = learner.err(Xv, Yv)
#
#     print x
#     print "Training Error: "
#     print tError
#     print "Validation Error: "
#     print vError
#
#
# ## part d ##
#
# for x in range(2,13):
#     learner = ml.dtree.treeClassify(Xt,Yt, minLeaf=2**x) ## part e : replace minLeaf with minParent ##
#
#     tError = learner.err(Xt, Yt)
#     vError = learner.err(Xv, Yv)
#
#     print 2**x + 50
#     print "Training Error: "
#     print tError
#     print "Validation Error: "
#     print vError

## part f ##

# learner = ml.dtree.treeClassify(Xt,Yt, maxDepth=50)
#
# learnerAuc = learner.auc(Xv, Yv)
# print learnerAuc
#
# learnerRoc = learner.roc(Xv, Yv)
# plt.plot(learnerRoc[0], learnerRoc[1])
# plt.show()

# ## part g ##
#
# ##training data
# Xt = np.genfromtxt("X_train.txt",delimiter=None) # load the text file
# Yt = np.genfromtxt("Y_train.txt",delimiter=None) # load the text file
#
# #test data
# Xte = np.genfromtxt("X_test.txt",delimiter=None) # load the text file
#
#
# learner = ml.dtree.treeClassify(Xt,Yt, minLeaf=128)
# Ypred = learner.predictSoft( Xte )
# # Now output a file with two columns, a row ID and a confidence in class 1:
# np.savetxt('Yhat_dtree.txt',
# np.vstack( (np.arange(len(Ypred)) , Ypred[:,1]) ).T,
# '%d, %.2f',header='ID,Prob1',comments='',delimiter=',');

## Problem 3 ##

## part a ##

# ##training data
# Xt = np.genfromtxt("X_train.txt",delimiter=None)[:10000] # load the text file
# Yt = np.genfromtxt("Y_train.txt",delimiter=None)[:10000] # load the text file
#
# #validation data
# Xv = np.genfromtxt("X_train.txt",delimiter=None)[10001:20000] # load the text file
# Yv = np.genfromtxt("Y_train.txt",delimiter=None)[10001:20000] # load the text file
#
# ensemble = []
# predicts = []
# trainError = []
# validError = []
# m,n = Xt.shape
#
# for x in range(1, 26):
#     ind = ml.bootstrapData(Xt, Yt, n_boot=x)
#     ensemble.append(ml.dtree.treeClassify(ind[0], ind[1], minLeaf = 4))
#     trainError.append(ensemble[x-1].err(Xt, Yt))
#     validError.append(ensemble[x-1].err(Xv, Yv))
#
# print trainError
# print validError

# for i in [1, 5, 10, 25]:
#     plt.plot([x for x in range(1, i+1)], validError)
#     plt.show()
#     plt.close()
#     plt.plot([x for x in range(1,i+1)], trainError)
#     plt.show()
#     plt.close()
#

## part b ##

##training data
Xt = np.genfromtxt("X_train.txt",delimiter=None)[:10000] # load the text file
Yt = np.genfromtxt("Y_train.txt",delimiter=None)[:10000] # load the text file

##validation data
Xv = np.genfromtxt("X_train.txt",delimiter=None)[10001:20000] # load the text file
Yv = np.genfromtxt("Y_train.txt",delimiter=None)[10001:20000] # load the text file

#test data
Xte = np.genfromtxt("X_test.txt",delimiter=None) # load the text file

ensemble = []
trainError = []
validError = []
predicts = []
aucs = []

for x in range(14, 21):
    ind = ml.bootstrapData(Xt, Yt, n_boot=x)
    ensem = ml.dtree.treeClassify(ind[0], ind[1], minLeaf = 4) ## 0.55670,
    ensemble.append(ensem)
    trainError.append(ensem.err(Xt, Yt))
    validError.append(ensem.err(Xv, Yv))
    predicts.append(ensem.predict(Xte))
    aucs.append(ensem.auc(Xv, Yv))

aucMean = 0
for x in aucs:
    aucMean = aucMean + x

print aucMean / len(aucs)

Ypred = [[x] for x in predicts[0]]
for x in Ypred:
    x = 0

for x, pred in enumerate(predicts):
    for item in pred:
        Ypred[x].append(item)

for x in Ypred:
    x = np.array(x)

newPred = []
for x in Ypred:
    newPred.append(np.mean(x))

# Now output a file with two columns, a row ID and a confidence in class 1:
np.savetxt('Yhat_dtree.txt',
np.vstack( (np.arange(len(newPred)) , newPred)).T,
'%d, %.2f',header='ID,Prob1',comments='',delimiter=',');

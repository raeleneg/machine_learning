import numpy as np
import mltools as ml
import mltools.cluster as cl
import matplotlib.pyplot as plt

## part a ##

np.random.seed(0)
iris = np.genfromtxt('data/iris.txt',delimiter=None) # load the text file
X = iris[:,0:2] ## restrict iris to 2 features, ignore class var


## part b ##

sumd = float("inf")
for i in range(5):
    Zi, Ci, SUMDi = cl.kmeans(X, 5, 'random') ## 5 clusters
    if sumd > SUMDi:
        Z = Zi
        C = Ci
        sumd = SUMDi

print "Best Score (5 Clusters): "
print sumd

ml.plotClassify2D(None, X, Z)
# plt.show()

sumd = float("inf")
for i in range(5):
    Zi, Ci, SUMDi = cl.kmeans(X, 20, 'random')  ## 20 clusters
    if sumd > SUMDi:
        Z = Zi
        C = Ci
        sumd = SUMDi

print "Best Score (20 Clusters): "
print sumd

ml.plotClassify2D(None, X, Z)
# plt.show()

## part c ##

Z, join = cl.agglomerative(X, 5, 'min') ## single linkage method (5 clusters)
ml.plotClassify2D(None, X, Z)
#plt.show()

Z, join = cl.agglomerative(X, 5, 'max') ## complete linkage method (20 clusters)
ml.plotClassify2D(None, X, Z)
#plt.show()


Z, join = cl.agglomerative(X, 20, 'min') ## single linkage method (5 clusters)
ml.plotClassify2D(None, X, Z)
#plt.show()

Z, join = cl.agglomerative(X, 20, 'max') ## complete linkage method (20 clusters)
ml.plotClassify2D(None, X, Z)
#plt.show()

## part d  (Optional) ##

ll = float("inf")
for i in range(5):
    Zi, Ti, softi, lli = cl.gmmEM(X, 5, 'k++')  ## 5 clusters
    if ll > lli:
        Z = Zi
        T = Ti
        soft = softi
        ll = lli

print "Best Score (ll): "
print ll

ml.plotClassify2D(None, X, Z)
# plt.show()


import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
import random
import scipy.linalg
import math
import mltools.transforms as trans

X = np.genfromtxt("data/faces.txt", delimiter=None) # load face dataset
plt.figure()
i = random.randint(0, 4917) # pick a data point i for display
img = np.reshape(X[i,:],(24,24)) # convert vectorized data point to 24x24 imag plt.imshow( img.T , cmap="gray") # display image patch; you may have to squint

## part a ##

mu = np.mean(X)
X0 = X - mu  # data zero mean: equiv to 576 pixels

print X0

## part b ##

U, S, V = scipy.linalg.svd(X, full_matrices=False) ## svd of data
W = U.dot(np.diag(S))

## part c ##

err = []

for k in range(1, 11):  # first k eigendirections
    Xhat0 = W[:,:k].dot(V[:k,:])
    err.append(np.mean(X0 - Xhat0)**2) ## mean squared error in SVD aprx
plt.plot([x for x in range(1, 11)], err)
plt.show()


## part d ##

for j in range(1, 4): # first 3 principal directions
    alpha = 2*np.median(np.abs(W[:,j])) # scale factor
    img1 = np.reshape(mu + alpha*V[j,:], [24, 24])
    img2 = np.reshape(mu - alpha*V[j,:], [24, 24])
    plt.imshow(img1, cmap='gray')
    plt.show()
    plt.imshow(img2, cmap='gray')
    plt.show()


## part e ##

for i in [16, 24]: # reconstruct two faces
    im = X[i,:]
    im = np.reshape(im, [24, 24])
    plt.imshow(im, cmap='gray')
    plt.show()
    for k in [5, 10, 50, 100]: # using first K principal directions
        im = mu + W[i,:k].dot(V[:k,])
        im = np.reshape(im, [24, 24])
        plt.imshow(im, cmap='gray')
        plt.show()
        plt.close()

## part f ##

idx = [x for x in range(15)] # pick data
coord, params = trans.rescale(W[:,0:2])
plt.figure() # normalize scale of "W" locations
plt.hold(True) # for pyplot
for i in idx:
    # compute where to place image (scaled W values) & size
    loc = (coord[i, 0], coord[i, 0] + 0.5, coord[i, 1], coord[i, 1] + 0.5)
    img = np.reshape(X[i, :], (24, 24))  # reshape to square
    plt.imshow(img.T, cmap="gray", extent=loc)  # draw each image
    plt.axis((-2, 2, -2, 2) ) # set axis to reasonable visual scale
plt.show()
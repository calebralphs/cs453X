import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize
import math
from sklearn.decomposition import PCA
import copy

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 1 # Number of examples on which to check the gradient

def plotSGDPath (trainX, trainY, ws1, ws2, superimpose = False):
    pca = PCA(n_components = 2)
    yfit1 = pca.fit_transform(ws1)
    
    if superimpose:
        pca = PCA(n_components = 2)
        yfit2 = pca.fit_transform(ws2)
    
    fig = plt.figure()
    ax = fig.gca(projection = '3d')

    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-8, 15, 1)
    axis2 = np.arange(-8, 15, 1)
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for x in range(len(axis1)):
        for y in range(len(axis2)):
            Zaxis[x, y] = fCE(trainX, trainY, pca.inverse_transform((Xaxis[x, y], Yaxis[x, y])))
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha = 0.5)  # Keep alpha < 1 so we can see the scatter plot too.

    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis1 = yfit1.T[0]
    Yaxis1 = yfit1.T[1]
    Zaxis1 = np.zeros(len(ws1))
    for z in range(len(ws1)):
        Zaxis1[z] = fCE(trainX, trainY, ws1[z])
    ax.scatter(Xaxis1, Yaxis1, Zaxis1, color='r')
    
    if superimpose:
        Xaxis2 = yfit2.T[0]
        Yaxis2 = yfit2.T[1]
        Zaxis2 = np.zeros(len(ws2))
        for z in range(len(ws2)):
            Zaxis2[z] = fCE(trainX, trainY, ws2[z])
        ax.scatter(Xaxis2, Yaxis2, Zaxis2, color='r')

    plt.show()

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    W1 = w[:NUM_INPUT * NUM_HIDDEN].reshape((NUM_INPUT, NUM_HIDDEN))
    b1 = w[NUM_INPUT * NUM_HIDDEN: NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN].reshape((NUM_HIDDEN))
    W2 = w[NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN:NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_OUTPUT].reshape((NUM_HIDDEN, NUM_OUTPUT))
    b2 = w[NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_OUTPUT:].reshape(NUM_OUTPUT)
    return W1, b1, W2, b2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    w = []
    W1flat = W1.flatten()
    b1flat = b1.flatten()
    W2flat = W2.flatten()
    b2flat = b2.flatten()
    w = np.concatenate((W1flat, b1flat, W2flat, b2flat))
    return w

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("mnist_{}_images.npy".format(which))
    labels = np.load("mnist_{}_labels.npy".format(which))
    return images, labels

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE (X, Y, w):
    z1, h1, z2, yhat = forwardProp(X, w)
    cost = -1 * np.sum(Y * np.log(yhat.T), axis = 1)
    return np.mean(cost)

def forwardProp(X, w):
    W1, b1, W2, b2 = unpack(w)
    z1 = X.dot(W1) + b1
    h1 = ReLU(z1)
    z2 = h1.dot(W2) + b2
    yhat = softmax(z2)
    return z1, h1, z2, yhat

def ReLU(z):
    z[z <= 0] = 0
    return z

def softmax(z):
    exp_z = np.exp(z)
    sum_exp_z = np.atleast_2d(exp_z).sum(axis = 1)
    norm_exp_z = (exp_z.T / sum_exp_z)
    return norm_exp_z

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    z1, h1, z2, yhat = forwardProp(X, w)  
    db2 = np.mean(yhat.T - Y, axis = 0)
    dW2 = np.atleast_2d(yhat.T - Y).T.dot(h1).T
    g = ((yhat.T - Y).dot(W2.T)) * ReLUprime(z1)
    db1 = np.mean(g, axis = 0)
    dW1 = X.T.dot(np.atleast_2d(g))
    dw = pack(dW1, db1, dW2, db2)
    return dw

def ReLUprime(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN. Then return the sequence of w's obtained during SGD.
def train (trainX, trainY, testX, testY, w, print_epoch = True, hyper_params = [.001, 100, 100, .1]):
    sigma = hyper_params[0]
    batch_size = hyper_params[1]
    n_epochs = hyper_params[2]
    beta = hyper_params[3]
    n_batches = math.floor(trainX.shape[0] / batch_size)
    
    ws = []
    costs = []
    for epoch in range(n_epochs):
        trainX, trainY = shuffleData(trainX, trainY)
        batchesX, batchesY = createBatches(trainX, trainY, n_batches)
        for batchX, batchY in zip(batchesX, batchesY):
            grad = gradCE(batchX, batchY, w)
            gradreg = w * beta + grad * (1 - beta)
            w -= gradreg * sigma
        ws.append(copy.deepcopy(w))
        cost = fCE(trainX, trainY, w)
        costs.append(copy.deepcopy(cost))
        if n_epochs - epoch <= 20 and print_epoch:
            print("Epoch:", epoch, "Cost:", cost)
    
    __, __, __, yhat = forwardProp(testX, w)
    accuracy = calcAccuracy(testY, yhat.T)
    loss = fCE(trainX, trainY, w)
    print("Loss:", loss)
    print("Accuracy:", accuracy)
    return ws, costs, accuracy

def shuffleData(X, y):
    shuffle = np.random.permutation(np.shape(X)[0])
    shuffled_X = X[shuffle]
    shuffled_y = y[shuffle]
    return shuffled_X, shuffled_y

def createBatches(X, y, ntilde):
    num_batches = math.floor(np.shape(X)[0] / ntilde)
    X_batches = np.split(X, num_batches)
    y_batches = np.split(y, num_batches)
    return np.array(X_batches), np.array(y_batches)

def decodeOnehot(y):
    return np.array([np.argmax(i) for i in y])

def calcAccuracy(y, yhat):
    y = decodeOnehot(y)
    yhat = decodeOnehot(yhat)
    return np.mean(y == yhat)

def findBestHyperparameters(trainX, trainY, testX, testY, w):
    sigmas = [.001, .01, .1]
    batch_sizes = [50, 100]
    betas = [.0, .1, .5]
    for s in sigmas:
        for bs in batch_sizes:
            for b in betas:
                print("Sigma:", s, "Batch Size:", bs, "Beta:", b)
                params = [s, bs, 20, b]
                __, __, __ = train(trainX, trainY, testX, testY, w, False, params)

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")
    
    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_INPUT, NUM_HIDDEN))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    w = pack(W1, b1, W2, b2)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print("Grad Check:", scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    w))
    
    #findBestHyperparameters(trainX, trainY, testX, testY, w)
    
    # Train the network and obtain the sequence of w's obtained using SGD.
    ws1, costs, accuracy = train(trainX, trainY, testX, testY, w)
    
    superimpose = False
    if superimpose:
        # Initialize weights randomly
        W1 = 2*(np.random.random(size=(NUM_INPUT, NUM_HIDDEN))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
        b1 = 0.01 * np.ones(NUM_HIDDEN)
        W2 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
        b2 = 0.01 * np.ones(NUM_OUTPUT)
        w = pack(W1, b1, W2, b2)
        ws2, costs, accuracy = train(trainX, trainY, testX, testY, w)
        # Plot the SGD trajectory
        plotSGDPath(trainX[:2500], trainY[:2500], ws1, ws2, True)
    
    plotSGDPath(trainX[:2500], trainY[:2500], ws1, None)
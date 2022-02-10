import numpy as np
import random



def calculate_gradients(X_tr, y_tr, w, b, alpha):
    n = X_tr.shape[0]
    # print(X_tr.shape)
    # print(b.shape)
    w_gradient = ((np.dot(X_tr.T,(np.dot(X_tr, w) + b - y_tr)))/n) + (alpha/n)*(np.absolute(w))
    b_gradient = ((np.dot(X_tr, w) + b - y_tr))/n 
    # print(b_gradient.shape)
    return (w_gradient, b_gradient[0])

def stochastic_gradient_descent(X_tr, y_tr, X_val, y_val, learning_rate, minibatch_size, alpha, epochs):
    w = np.random.rand(X_tr.shape[1])
    w = w.reshape(w.shape[0],1)
    b = np.random.rand(1)
    b = b.reshape(1,1)
    # learning_rate = 0.0005
    # minibatch_size = 64
    num_iterations = int((X_tr.shape[0])/minibatch_size)
    y_tr = np.reshape(y_tr, (y_tr.shape[0], 1))
    # alpha = 0.5
    for ep in range(0, epochs):
        # print("Epoch No. = {}/{}".format(ep+1, 10))
        li = np.arange(0, X_tr.shape[0])
        random.shuffle(li)
        newX_tr = X_tr[li,:]
        newY_tr = y_tr[li,:]
        # np.random.shuffle(X_tr)
        for ni in range(0, num_iterations):
            X_mb = newX_tr[ni*minibatch_size:(ni+1)*minibatch_size,:]
            y_mb = newY_tr[ni*minibatch_size:(ni+1)*minibatch_size,:]
            mini_wgradient, mini_bgradient = calculate_gradients(X_mb, y_mb, w, b, alpha)
            w = w - learning_rate*mini_wgradient
            b = b - learning_rate*mini_bgradient


        # print((np.dot( w.T, X_tr.T)).shape)
        fMSE_Train = (np.mean( np.square( (np.dot( w.T, X_tr.T)) + b.T - y_tr.T), axis = 1))/2
        fMSE_Valid = (np.mean( np.square( (np.dot( w.T, X_val.T)) + b.T - y_val.T), axis = 1))/2
        # print(fMSE_Train)
        # print(fMSE_Valid)        

    return w, b, fMSE_Valid

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    valid_size = int(0.2*X_tr.shape[0])
    X_val = X_tr[0:valid_size,:]
    y_val = yte[0:valid_size]

    # w = linear_regression(X_tr, ytr)

    # Report fMSE cost on the training and testing data (separately)
    # ...
    listAccuracy = []
    epochs = [100, 200, 300, 800]
    learningRates = [0.0005, 0.005, 0.05, 0.5]
    alphas = [0.1, 0.2, 0.3, 0.5]
    minibatch_size = [32, 64, 128, 256]
    minLoss = float('inf')
    bestW = np.random.rand(X_tr.shape[1])
    bestB = np.random.rand(1,1)
    counter = 0
    for ep in epochs:
        for lr in learningRates:
            for alpha in alphas:
                for ms in minibatch_size:
                    print("Epochs = {}, Learning Rate = {}, Regularization Strength = {}, Minibatch Size = {}".format(ep, lr, alpha, ms))


                    w, b, currValidLoss = stochastic_gradient_descent(X_tr, ytr, X_val, y_val, lr, ms, alpha, ep)
                    if currValidLoss < minLoss:
                        # print(minLoss)
                        bestW = w
                        bestB = b
                        minLoss = currValidLoss

                    # print(counter)
                    counter = counter + 1
                    print(minLoss)


    fMSE_Train = (np.mean( np.square( (np.dot( bestW.T, X_tr.T)+ bestB.T) - ytr), axis = 1))/2
    fMSE_Test = (np.mean( np.square( (np.dot( bestW.T, X_te.T)+ bestB.T) - yte), axis = 1))/2
    print(minLoss)
    # print(fMSE_Train)
    # print(fMSE_Test)



train_age_regressor()
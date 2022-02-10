# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:34:26 2022

@author: SAMEER
"""

import numpy as np
import random




def softmax(z):
    
     
    exp = np.exp(z - np.max(z))
    
    # Calculating softmax for all examples.
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])
        
    return exp

def one_hot(y, c):
    
   
    
    # A zero matrix of size (m, c)
    y_hot = np.zeros((len(y), c))
    
    # Putting 1 for column where the label is,
    # Using multidimensional indexing.
    y_hot[np.arange(len(y)), y] = 1
    
    return y_hot

def predict(X, w, b):
    
    # X --> Input.
    # w --> weights.
    # b --> bias.
    
    # Predicting
    z = X@w + b
    y_hat = softmax(z)
    
   
    return np.argmax(y_hat, axis=1)

def accuracy(y, y_hat):
    return np.sum(y==y_hat)/len(y)



def calculate_gradients(X_tr, y_tr, w, b, alpha):
    n = X_tr.shape[0]
    # print(X_tr.shape)
    # print(b.shape)
    w_gradient = ((np.dot(X_tr.T,(np.dot(X_tr, w) + b - y_tr)))/n) + (alpha/n)*(np.absolute(w))
    b_gradient = ((np.dot(X_tr, w) + b - y_tr))/n 
    # print(b_gradient.shape)
    return (w_gradient, b_gradient[0])

def stochastic_gradient_descent(X_tr, y_tr, X_val, y_val, learning_rate, minibatch_size, alpha, epochs):
   
    w=np.random.random((784,10))
    b=np.random.random(10)
   
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
            z = X_tr@w + b
            y_hat = softmax(z)
            
            # One-hot encoding y.
            y_hot = one_hot(y_tr, 10)
            ##change gradients functions for appropiate w
            mini_wgradient, mini_bgradient = calculate_gradients(X_mb, y_mb, w, b, alpha)
            w = w - learning_rate*mini_wgradient
            b = b - learning_rate*mini_bgradient


        # print((np.dot( w.T, X_tr.T)).shape)
        # fMSE_Train = (np.mean( np.square( (np.dot( w.T, X_tr.T)) + b.T - y_tr.T), axis = 1))/2
        #change cost function
        
        #  complete this cost function for validation
        fMSE_Valid = (np.mean( np.square( (np.dot( w.T, X_val.T)) + b.T - y_val.T), axis = 1))/2
        # print(fMSE_Train)
        # print(fMSE_Valid)        

    return w, b, fMSE_Valid

def train_cloth_softmax_regressor ():
    # Load data
    X_tr = (np.load("C:/Users/SAMEER/Downloads/fashion_mnist_train_images.npy") )
    ytr = np.load("C:/Users/SAMEER/Downloads/fashion_mnist_train_labels.npy")
    X_te = np.load("C:/Users/SAMEER/Downloads/fashion_mnist_test_images.npy")
    yte = np.load("C:/Users/SAMEER/Downloads/fashion_mnist_test_labels.npy")
   
    valid_size = int(0.2*X_tr.shape[0])
    X_val = X_tr[0:valid_size,:]
    y_val = ytr[0:valid_size]
  
    
    # w = linear_regression(X_tr, ytr)

    # Report fMSE cost on the training and testing data (separately)
    # ...
    m, n = X_tr.shape
    c=10
    epochs = [100, 200, 300, 800]
    learningRates = [0.0005, 0.005, 0.05, 0.5]
    alphas = [0.1, 0.2, 0.3, 0.5]
    minibatch_size = [32, 64, 128, 256]
    minLoss = float('inf')
    bestW = np.random.random((n, c))
    bestB = np.random.random(c)
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

    # performance evaluation
    ## change this according to loss function
    fMSE_Test = (np.mean( np.square( (np.dot( bestW.T, X_te.T)+ bestB.T) - yte), axis = 1))/2
    
    
    print(fMSE_Test)
    train_preds = predict(X_tr, bestW, bestB)
    print(accuracy(ytr, train_preds))
    test_preds = predict(X_te, bestW, bestB)
    print("accuracy in test")
    print(accuracy(yte, test_preds)*100)
    


train_cloth_softmax_regressor()
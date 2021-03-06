import numpy as np

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return (np.dot(A, B) - C)

def problem_1c (A, B, C):
    return (A*B - C.T)

def problem_1d (x, y):
    return np.dot(x, y)

def problem_1e (A, x):
    return np.linalg.solve(A, x)

def problem_1f (A, x):

    return np.linalg(A.T, x.T)

def problem_1g (A, i):
    arrayBool = np.zeros([1, A.shape[1]],dtype=bool)
    arrayBool[:, 0:A.shape[1]:2] = True
    return np.sum(np.add.reduce(A[i,:], where=arrayBool, axis=0))

def problem_1h (A, c, d):
    
    mask1 = ((A>=c))
    mask2 = ((A<=d))
    mask = mask1 * mask2
    nonzeroElements = np.nonzero(mask)
    meanNonZero = np.mean(nonzeroElements)
    return meanNonZero

def problem_1i (A, k):
    eigValues, eigVectors = np.linalg.eig(A)
    indices = np.argpartition(eigValues, -k)[-k:]
    sortedIndices = indices[np.argsort((eigValues[indices]))]
    reverseSortedIndices = sortedIndices[::-1]
    return eigVectors[:, reverseSortedIndices]

def problem_1j (x, k, m, s):
    mean = x + m
    covariance = s*np.identity(x.shape[0])
    distribution = np.random.multivariate_normal(mean, covariance, size=k)
    finalDistribution = distribution.T
    return finalDistribution

###CHANGE THIS this doesnt return anything:
def problem_1k (A):
    np.random.shuffle(A)
    return A

def problem_1l (x):
    return ((x - np.mean(x))/(np.std(x)))

def problem_1m (x, k):
    x = x.reshape(x.shape[0],1)
    return np.repeat(x,k, axis=1)

def problem_1n (X):
    
    B = X.reshape((X.shape[0], X.shape[1], -1))
    print(B.shape)
    C = np.repeat(B, X.shape[1], 2)
    D = np.transpose(C, [0, 2, 1])
    L2 = np.linalg.norm(C - D, axis = 0)
    return L2

def linear_regression (X_tr, y_tr):
    X_tr = X_tr.T
    y_tr = np.reshape(y_tr, (y_tr.shape[0], 1))
    w = np.linalg.solve(np.dot(X_tr, X_tr.T), np.dot(X_tr, y_tr))
    return w

    # ...

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)

    # Report fMSE cost on the training and testing data (separately)
    # ...
    fMSE_Train = (np.mean( np.square( (np.dot( w.T, X_tr.T)) - ytr), axis = 1))/2
    fMSE_Test = (np.mean( np.square( (np.dot( w.T, X_te.T)) - yte), axis = 1))/2

train_age_regressor()


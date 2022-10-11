import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from FrankeFunction import FrankeFunction
from OLS import OLS, designmatrix
import os
work_path = os.path.dirname(__file__) 
fig_path = os.path.join(work_path, "fig/") #path to directory storing figures


def k_fold(X ,y ,n_splits, reg_method, maxdegree, mindegree=1, show=True, lmb_index=0, filename=""):
    reg_name = filename.split("_")[0]

    n = len(y)
    random_index = np.linspace(0,n-1,n, dtype="int")
    np.random.shuffle(random_index)
    if ceil(n/n_splits)!=int(n/n_splits):
        raise ValueError(f"Data set can't split into {n_splits} equal length subsets")
    
    else:
        split_index = random_index.reshape(n_splits,int(n/n_splits)) 
    
    N = maxdegree - mindegree + 1 
    error = np.zeros(N)
    polydegree = np.zeros(N)

    for degree in range(mindegree, maxdegree + 1):
        y_pred = np.zeros( n_splits)
        l = int((degree+1)*(degree+2)/2)		# Number of elements in beta
        i = 0
        scores_KFold = np.zeros(n_splits)
        for test_indecies in split_index:
            train_indecies = np.delete(split_index, int(i), 0).ravel()
            X_test = X[test_indecies, :l]
            y_test = y[test_indecies]
            X_train = X[train_indecies, :l]
            y_train = y[train_indecies]
            y_pred = reg_method(X_train, X_test, y_train)[1]
            if len(y_pred.shape)>1:
                scores_KFold[i] = np.mean((y_pred[:, lmb_index] - y_test)**2)
                lambdas = reg_method(X_train, X_test, y_train)[3]
            else:
                scores_KFold[i] = np.mean((y_pred - y_test)**2)
            i+=1
        error[degree-mindegree] = np.mean(scores_KFold)
        polydegree[degree-mindegree] = degree
    plt.plot(polydegree, error, "*-")
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    if len(y_pred.shape)>1:
        plt.title(reg_name+" cross-validation "r"$\lambda $" f"= {lambdas[lmb_index]:.3g}")
    else:
        plt.title(reg_name+" cross-validation")
    if filename!="":
        plt.savefig(fig_path+filename)
    if show:
        plt.show()
    

if __name__=="__main__":
    np.random.seed(17)

    x = np.sort(np.random.uniform(0,1,20))
    y = np.sort(np.random.uniform(0,1,20))
    noise = 0.1
    f = np.ravel(FrankeFunction(x, y, noise)) 
    maxdegree = 12
    X = designmatrix(maxdegree, x, y)
    k_fold(X, f, 10, OLS, maxdegree)
    
    
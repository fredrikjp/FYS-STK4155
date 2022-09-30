import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from FrankeFunction import FrankeFunction
from OLS import designmatrix


def k_fold(x,y,n_splits):
    n = len(y)
    random_index = np.linspace(0,n-1,n, dtype="int")
    np.random.shuffle(random_index)
    if ceil(n/n_splits)!=int(n/n_splits):
        print(f"Data set can't split into {n_splits} equal length subsets")
        return 
    else:
        split_index = random_index.reshape(n_splits,int(n/n_splits)) 
    
    i = 0
    scores_KFold = np.zeros(n_splits)
    for test_indecies in split_index:
        train_indecies = np.delete(split_index, int(i), 0).ravel()
        x_test = x[test_indecies]
        y_test = y[test_indecies]
        x_train = x[train_indecies]
        y_train = y[train_indecies]
        OLSbeta = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ y_train
        y_pred = x_test @ OLSbeta

        scores_KFold[i] = np.sum((y_pred - y_test)**2)/np.size(y_pred)
        i+=1
    return np.linspace(0,n_splits-1,n_splits), scores_KFold
if __name__=="__main__":
    x = np.sort(np.random.uniform(0,1,20))
    y = np.sort(np.random.uniform(0,1,20))
    x_grid, y_grid = np.meshgrid(x,y)
    noise = 0.1
    f = np.ravel(FrankeFunction(x_grid,y_grid)) 
    f += noise*np.random.normal(0,1, len(f))    
    X = designmatrix(5,x_grid,y_grid)

    split, KFold = k_fold(X,f,10)
    plt.plot(split, KFold, "*-")
    plt.show()
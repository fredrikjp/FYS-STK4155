import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from FrankeFunction import FrankeFunction
from OLS import OLS, designmatrix
import os
work_path = os.path.dirname(__file__) 
fig_path = os.path.join(work_path, "fig/") #path to directory storing figures


def bias_var(X_train, X_test, y_train, y_test, reg_method, maxdegree, n_boostraps, lmb_index=0, filename=""):
    
    y_test = y_test.reshape(-1, 1)
    error = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)
    polydegree = np.zeros(maxdegree)
    reg_name = filename.split("_")[0]
    for degree in range(1, maxdegree + 1):
        y_pred = np.empty((y_test.shape[0], n_boostraps))
        l = int((degree+1)*(degree+2)/2)		# Number of elements in beta
        for i in range(n_boostraps):
            X_ , y_ = resample(X_train[:,:l], y_train)
            reg = reg_method(X_, X_test[:,:l], y_)
            ytilde_test = reg[1]
            if len(ytilde_test.shape)>1:
                y_pred[:,i] = ytilde_test[:,lmb_index]
                lambdas = reg[3]
            else:
                y_pred[:,i] = ytilde_test
        polydegree[degree-1] = degree
        error[degree-1] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
        bias[degree-1] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
        variance[degree-1] = np.mean( np.var(y_pred, axis=1, keepdims=True) )
    
    

    plt.plot(polydegree, error, label='Error')
    plt.plot(polydegree, bias, label=r'Bias$^2$')
    plt.plot(polydegree, variance, label='Variance')
    plt.xlabel("Polynomial degree")
    plt.legend()
    if len(ytilde_test.shape)>1:
        plt.title(reg_name+" bootstrap "+r"$\lambda $"+f"= {lambdas[lmb_index]:.3g}")
    else:
        plt.title(reg_name+" bootstrap")
    if filename!="":
        plt.savefig(fig_path+filename)
    plt.show()


if __name__ == "__main__":
    
    np.random.seed(17)

    n = 2
    n_boostraps = 100
    maxdegree = 2
    noise = 0.1

    # Make data set.
    x = np.sort(np.random.uniform(0,1,n))
    y = np.sort(np.random.uniform(0,1,n))
    X = designmatrix(maxdegree, x, y)
    f = np.ravel(FrankeFunction(x, y, noise))

    X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2)
    bias_var(X_train, X_test, y_train, y_test, OLS, maxdegree, n_boostraps)

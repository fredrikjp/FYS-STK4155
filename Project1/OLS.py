import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from FrankeFunction import FrankeFunction
from plot_3D import plot_3D

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n




def designmatrix(n, x, y):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X
          



def OLS_Frank(x, y, pol_degree, noise = 0, plot=False):
    x_grid, y_grid = np.meshgrid(x,y)
    f = np.ravel(FrankeFunction(x_grid,y_grid))
    f+= noise*np.random.normal(0,1, len(f))
    
    if pol_degree == 0:
        X = designmatrix(pol_degree+1, x_grid, y_grid)
        X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2)
        x0, y0 = X_train[:,1], X_train[:,2]
        x1, y1 = X_test[:,1], X_test[:,2]
        X_train, X_test = X_train[:,:0], X_test[:,:0]

    else:
        X = designmatrix(pol_degree, x_grid, y_grid)
        X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2)
        x0, y0 = X_train[:,1], X_train[:,2]
        x1, y1 = X_test[:,1], X_test[:,2]
    #scaler = StandardScaler()
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)

    # matrix inversion to find beta
    OLSbeta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train

    # and then make the prediction
    ytilde_train = X_train @ OLSbeta
    ytilde_test = X_test @ OLSbeta
    if plot:
        plot_3D(x0, y0, ytilde_train, f"OLS train, polynomial degree = {pol_degree}")
        plot_3D(x1, y1, ytilde_test, f"OLS test, polynomial degree = {pol_degree}")


    return X_train, y_train, ytilde_train, X_test, y_test, ytilde_test

def plot_poldeg_analysis(x, y, pol_deg_lim, noise=0, beta_bool=True, MSE_bool=True, R2_bool=True):
    n = []
    
    R2_train = []
    MSE_train = []
    
    R2_test = []
    MSE_test = []
    
    OLS = OLS_Frank(x, y, pol_deg_lim, noise)

    X_train = OLS[0]
    y_train = OLS[1]
    X_test = OLS[3]
    y_test = OLS[4]
    for i in range (0, pol_deg_lim):
        l = int((i+1)*(i+2)/2)		# Number of elements in beta
        X_tr = X_train[:,:l]
        X_te = X_test[:,:l]

        n.append(i)
        beta = np.linalg.pinv(X_tr.T @ X_tr) @ X_tr.T @ y_train

        # and then make the prediction
        ytilde_train = X_tr @ beta
        ytilde_test = X_te @ beta

        beta_index = np.linspace(0, len(beta) - 1, len(beta))

        R2_train.append(R2(y_train, ytilde_train))
        MSE_train.append(MSE(y_train, ytilde_train))
        
        R2_test.append(R2(y_test, ytilde_test))
        MSE_test.append(MSE(y_test, ytilde_test))
        
        if beta_bool:
            plt.plot(beta_index, beta, "*-")
    if beta_bool:
        plt.show()

    if MSE_bool:
        plt.plot(n, MSE_train, "*-")
        plt.plot(n, MSE_test, "*-")
        plt.xlabel("Polynomial degree")
        plt.ylabel("MSE")
        plt.legend(["MSE train", "MSE test"])
        plt.show()
    
    if R2_bool:
        plt.plot(n, R2_train, "*-")
        plt.plot(n, R2_test, "*-")
        plt.legend(["R2 train", "R2 test"])
        plt.xlabel("Polynomial degree")
        plt.ylabel("R2")
        plt.show()


if __name__=="__main__":
    # A seed just to ensure that the random numbers are the same for every run.
    # Useful for eventual debugging.
    np.random.seed(5)

    x = np.sort(np.random.uniform(0,1,100))
    y = np.sort(np.random.uniform(0,1,100))

    #OLS_Frank(x, y, 0, 0.1, plot=True)
    #plot_poldeg_analysis(x,y,5)

    #plot_poldeg_analysis(x,y,5, noise = 1)

    plot_poldeg_analysis(x, y, 20, R2_bool=False, beta_bool=False, noise=1)






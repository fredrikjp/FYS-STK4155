import numpy as np
from OLS import designmatrix, plot_poldeg_analysis
from FrankeFunction import FrankeFunction
from sklearn.model_selection import train_test_split
from plot_3D import plot_3D
import os
work_path = os.path.dirname(__file__) 
fig_path = os.path.join(work_path, "fig/") #path to directory storing figures


def Ridge(X_train, X_test, y_train, plot_index=-1, lambdas=np.logspace(-5,1,6), filename=""):

    #Get x0, y0 (train points) and x1, y1 (test points) for plotting
    x0, y0 = X_train[:,1], X_train[:,2]
    x1, y1 = X_test[:,1], X_test[:,2]
    
    n=len(lambdas)
    I = np.eye(len(X_train[0]),len(X_train[0]))
    # Decide which values of lambda to use
    ytilde_train = np.zeros((len(y_train), n))
    ytilde_test = np.zeros((len(X_test[:,0]), n))
    Ridge_beta = np.zeros((len(X_test[0]), n))
    for i in range(n):
        lmb = lambdas[i]
        Ridgebeta = np.linalg.inv(X_train.T @ X_train+lmb*I) @ X_train.T @ y_train
        Ridge_beta[:,i] = Ridgebeta
        # and then make the prediction
        ytilde_train[:,i] = X_train @ Ridgebeta
        ytilde_test[:,i] = X_test @ Ridgebeta

    
    if plot_index >= 0:
        if filename!="":
            plot_3D(x0,y0,ytilde_train[:,plot_index],f"Ridge train (lambda = {lambdas[plot_index]:.3})", fig_path+"train"+filename)
            plot_3D(x1,y1,ytilde_test[:,plot_index],f"Ridge test (lambda = {lambdas[plot_index]:.3})", fig_path+"test"+filename)
        else:    
            plot_3D(x0,y0,ytilde_train[:,plot_index],f"Ridge train (lambda = {lambdas[plot_index]:.3})")
            plot_3D(x1,y1,ytilde_test[:,plot_index],f"Ridge test (lambda = {lambdas[plot_index]:.3})")

    return ytilde_train, ytilde_test, Ridge_beta, lambdas

if __name__ == "__main__":
    #np.random.seed(1)
    n = 40
    x = np.sort(np.random.uniform(0,1,n))
    y = np.sort(np.random.uniform(0,1,n))
    f = np.ravel(FrankeFunction(x,y,1))
    
    pol_deg = 12
    X = designmatrix(pol_deg, x, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2)

    #Ridge(X_train, X_test, y_train, plot_index=0)
    plot_poldeg_analysis(X_train, X_test, y_train, y_test, pol_deg, Ridge, beta_bool=False, R2_bool=False)
    
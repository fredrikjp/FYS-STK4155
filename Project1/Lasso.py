import numpy as np
from OLS import OLS, designmatrix, plot_poldeg_analysis
from FrankeFunction import FrankeFunction
from sklearn.model_selection import train_test_split
from plot_3D import plot_3D
from sklearn import linear_model
import warnings
import os
work_path = os.path.dirname(__file__) 
fig_path = os.path.join(work_path, "fig/") #path to directory storing figures

warnings.filterwarnings('ignore')

def Lasso(X_train, X_test, y_train, plot_index=-1, lambdas=np.logspace(-5,1,6), filename=""):
    #Get x0, y0 (train points) and x1, y1 (test points) for plotting
    x0, y0 = X_train[:,1], X_train[:,2]
    x1, y1 = X_test[:,1], X_test[:,2]
    
    n = len(lambdas)

    ytilde_train = np.zeros((len(y_train), n))
    ytilde_test = np.zeros((len(X_test[:,0]), n))
    Lasso_beta = np.zeros((len(X_test[0]), n))

    for i in range(n):
        lmb = lambdas[i]
        RegLasso = linear_model.Lasso(lmb, fit_intercept=False, max_iter=10000)
        RegLasso.fit(X_train, y_train)
        # and then make the prediction
        ytilde_train[:,i] = RegLasso.predict(X_train)
        ytilde_test[:,i] = RegLasso.predict(X_test)
        Lasso_beta[:,i] = RegLasso.coef_
    
    if plot_index >= 0:
        if filename!="":
            plot_3D(x0,y0,ytilde_train[:,plot_index],f"Lasso train (lambda = {lambdas[plot_index]:.3})", fig_path+"train"+filename)
            plot_3D(x1,y1,ytilde_test[:,plot_index],f"Lasso test (lambda = {lambdas[plot_index]:.3})", fig_path+"test"+filename)
        else:    
            plot_3D(x0,y0,ytilde_train[:,plot_index],f"Lasso train (lambda = {lambdas[plot_index]:.3})")
            plot_3D(x1,y1,ytilde_test[:,plot_index],f"Lasso test (lambda = {lambdas[plot_index]:.3})")

    return ytilde_train, ytilde_test, Lasso_beta, lambdas

if __name__ == "__main__":
    #np.random.seed(3)
    n = 40
    x = np.sort(np.random.uniform(0,1,n))
    y = np.sort(np.random.uniform(0,1,n))
    f = np.ravel(FrankeFunction(x,y,1))   
    pol_deg = 20
    X = designmatrix(pol_deg, x, y)  
    X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2)
    #Lasso(X_train, X_test, y_train, plot_index=0)
    plot_poldeg_analysis(X_train, X_test, y_train, y_test, pol_deg, Lasso, beta_bool=False, R2_bool=False)
    plot_poldeg_analysis(X_train, X_test, y_train, y_test, pol_deg, OLS, beta_bool=False, R2_bool=False)
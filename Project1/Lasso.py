import numpy as np
from OLS import designmatrix
from FrankeFunction import FrankeFunction
from sklearn.model_selection import train_test_split
from plot_3D import plot_3D
from sklearn import linear_model

def Lasso_Frank(x, y, pol_degree, noise=0, plot_index=-1):
    x_grid, y_grid = np.meshgrid(x,y)

    f = np.ravel(FrankeFunction(x_grid,y_grid))
    f+= noise*np.random.normal(0,1, len(f))

    #Get x0, y0 (train points) and x1, y1 (test points) for plotting
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

    # Decide which values of lambda to use
    nlambdas = 100
    ytilde_train = np.zeros((len(y_train), nlambdas))
    ytilde_test = np.zeros((len(y_test), nlambdas))

    lambdas = np.logspace(-4, 4, nlambdas)
    for i in range(nlambdas):
        lmb = lambdas[i]
        RegLasso = linear_model.Lasso(lmb)
        RegLasso.fit(X_train,y_train)
        #RegLasso_test.fit(X_test,y_test)
    #    print(Lassobeta)
        # and then make the prediction
        ytilde_train[:,i] = RegLasso.predict(X_train)
        ytilde_test[:,i] = RegLasso.predict(X_test)
        
    
    if plot_index >= 0:
        plot_3D(x0,y0,ytilde_train[:,plot_index],f"Rigde train (lambda = {lambdas[plot_index]:.3}), polynomial degree = {pol_degree}")
        plot_3D(x1,y1,ytilde_test[:,plot_index],f"Rigde test (lambda = {lambdas[plot_index]:.3}), polynomial degree = {pol_degree}")

    return X_train, y_train, ytilde_train, X_test, y_test, ytilde_test

if __name__ == "__main__":
    n = 40
    x = np.sort(np.random.uniform(0,1,n))
    y = np.sort(np.random.uniform(0,1,n))
    Lasso_Frank(x,y,2,0.1,1)
from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload
import poly_data
import gradient_descent
import plot


def mse(y_test, y_pred):
    return np.mean((y_test - y_pred)**2)


if __name__ == '__main__':
    np.random.seed(0)
    reload(poly_data)
    reload(gradient_descent)
    reload(plot)

    eta = 0.001
    n_epochs = 100 
    n_epochs_sgd = 25
    p = poly_data.PolyData(n_data = 100)
    momentum = 0.1
    size_batch = 20
    
    gm = gradient_descent.GradientDescent(p)


#  ____                  _            _          _   _             
# |  _ \ ___  __ _ _   _| | __ _ _ __(_)______ _| |_(_) ___  _ __  
# | |_) / _ \/ _` | | | | |/ _` | '__| |_  / _` | __| |/ _ \| '_ \ 
# |  _ <  __/ (_| | |_| | | (_| | |  | |/ / (_| | |_| | (_) | | | |
# |_| \_\___|\__, |\__,_|_|\__,_|_|  |_/___\__,_|\__|_|\___/|_| |_|
#            |___/                                                 

    lamb = 0.1
    #lamb = 0

    
    gm.gd(eta, n_epochs, lamb=lamb)
    gd_plain = gm.get_thetas()

    #eta = 0.05
    gm.gd(eta, n_epochs, gamma=momentum, lamb=lamb, tuning_method="ADAM")
    # gm.gd(eta, n_epochs, 0.1) 
    gd_momentum = gm.get_thetas()

    gm.sgd(eta, n_epochs_sgd, size_batch, lamb=lamb, tuning_method="RMSprop")
    sgd = gm.get_thetas()

    gm.sgd(eta, n_epochs_sgd, size_batch, gamma=momentum, lamb=lamb, tuning_method="AdaGrad")
    sgd_momentum = gm.get_thetas()

    # Plotting
    """
    theta_dict = {'gd': gd_plain, 'gd_momentum': gd_momentum, 'sgd_theta': sgd, 'sgd_momentum': sgd_momentum}
    pl = plot.Plot(p)
    pl.plot_mse_vs_theta(theta_dict)
    """    
    etaa = np.linspace(0.1, 1, 5)
    lmb = [0, 1e-4, 1e-3, 1e-2, 1e-1]

    n = len(etaa)
    m = len(lmb)

    MSE_gd = np.zeros((n,m))
    MSE_gdm = np.zeros((n,m))
    MSE_sgd = np.zeros((n,m))
    MSE_sgdm = np.zeros((n,m))

    i = 0

    X = gm.X_data    
    y = gm.y_data

    for eta in etaa:
        j = 0
        for lamb in lmb:
            gm.gd(eta, n_epochs, lamb=lamb)
            gd_pred = X @ gm.get_theta()
            

            gm.gd(eta, n_epochs, gamma=momentum, lamb=lamb)
            gdm_pred = X @ gm.get_theta()

            gm.sgd(eta, n_epochs_sgd, size_batch, lamb=lamb)
            sgd_pred = X @ gm.get_theta()

            gm.sgd(eta, n_epochs_sgd, size_batch, gamma=momentum, lamb=lamb)
            sgdm_pred = X @ gm.get_theta()

            MSE_gd[i,j] = mse(y, gd_pred)
            MSE_gdm[i,j] = mse(y, gdm_pred)
            MSE_sgd[i,j] = mse(y, sgd_pred)
            MSE_sgdm[i,j] = mse(y, sgdm_pred)

            j += 1
        i += 1

    # Heatmaps
    i = 0
    for name, MSE in {"MSE gradient descent" : MSE_gd, "MSE gradient descent with momentum" : MSE_gdm, "MSE stochastic gradient descent" : MSE_sgd, "MSE stochastic gradient descent with momentum" : MSE_sgdm}.items():

        plt.figure(figsize=(12,8))
        plt.title(name)
        sns.heatmap(MSE, annot=True, fmt='.5g',
                 vmax = 0.1, 
                cbar_kws={'label': "MSE"}, 
                xticklabels = [f"{x_val:.5g}" for x_val in lmb],
                yticklabels=[f"{y_val:.5g}" for y_val in etaa]) 
        plt.xlabel(f"$\\lambda$")
        plt.ylabel(f"$\\eta$")
        if i == 0:
            plt.savefig("gd_MSE(eta,lmb).pdf")
        if i == 1:
            plt.savefig("gdm_MSE(eta,lmb).pdf")
        if i == 2:
            plt.savefig("sgd_MSE(eta,lmb).pdf")
        if i == 3:
            plt.savefig("sgdm_MSE(eta,lmb).pdf")
        i +=1
    plt.show()

    eta = 0.75
    for method in ["Plain", "AdaGrad", "RMSprop", "ADAM"]:
        
        gm.gd(eta, n_epochs, lamb=lamb, tuning_method=method)
        gd_plain = gm.get_thetas()

        gm.gd(eta, n_epochs, gamma=momentum, lamb=lamb, tuning_method=method)
        # gm.gd(eta, n_epochs, 0.1) 
        gd_momentum = gm.get_thetas()

        gm.sgd(eta, n_epochs_sgd, size_batch, lamb=lamb, tuning_method=method)
        sgd = gm.get_thetas()

        gm.sgd(eta, n_epochs_sgd, size_batch, gamma=momentum, lamb=lamb, tuning_method=method)
        sgd_momentum = gm.get_thetas()

        # Plotting
        theta_dict = {'gd': gd_plain, 'gd_momentum': gd_momentum, 'sgd': sgd, 'sgd_momentum': sgd_momentum}
        plt.title(method)

        pl = plot.Plot(p)
        pl.plot_iter_MSE(theta_dict)
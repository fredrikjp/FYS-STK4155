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

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression


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
    etaa = np.linspace(0.6, 1, 5)
    lmb = [0, 1e-4, 1e-3, 1e-2, 1e-1]

    n = len(etaa)
    m = len(lmb)

    MSE_gd = np.zeros((n,m))
    MSE_gdm = np.zeros((n,m))
    MSE_sgd = np.zeros((n,m))
    MSE_sgdm = np.zeros((n,m))

    i = 0

    X_train = gm.X_data    
    y_train = gm.y_data

    X = p.get_X_test()
    y = p.get_y_test()

    """
    tuning_method = ""
    
    for eta in etaa:
        j = 0
        for lamb in lmb:
            gm.gd(eta, n_epochs, lamb=lamb, tuning_method=tuning_method)
            gd_pred = X @ gm.get_theta()
            

            gm.gd(eta, n_epochs, gamma=momentum, lamb=lamb, tuning_method=tuning_method)
            gdm_pred = X @ gm.get_theta()

            gm.sgd(eta, n_epochs_sgd, size_batch, lamb=lamb, tuning_method=tuning_method)
            sgd_pred = X @ gm.get_theta()

            gm.sgd(eta, n_epochs_sgd, size_batch, gamma=momentum, lamb=lamb, tuning_method=tuning_method)
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
    """
    
    """
    # sgd MSE(eta, momentum)

    lamb = 0
    etaa = np.linspace(0.1, 1, 10)
    gamma = [0, 0.2, 0.4, 0.6, 0.8]

    n = len(etaa)
    m = len(gamma)
    MSE_sgdm = np.zeros((n,m))
    
    
    np.random.seed(1)
    for tuning_method in ["", "AdaGrad", "RMSprop", "ADAM"]:
        
        etaa = np.linspace(0.1, 1, 10)
        if tuning_method == "ADAM":
            etaa = np.logspace(-3, 0, 10)
        i = 0

        for eta in etaa:
            j = 0
            for momentum in gamma:
                
                gm.sgd(eta, n_epochs_sgd, size_batch, gamma=momentum, lamb=lamb, tuning_method=tuning_method)
                sgdm_pred = X @ gm.get_theta()

                MSE_sgdm[i,j] = mse(y, sgdm_pred)

                j += 1
            i += 1

        # Heatmap

        plt.figure(figsize=(12,8))
        plt.title(tuning_method+" MSE stochastic gradient descent")
        sns.heatmap(MSE_sgdm, annot=True, fmt='.5g',
                vmax = 0.1, 
                cbar_kws={'label': "MSE"}, 
                xticklabels = [f"{x_val:.5g}" for x_val in gamma],
                yticklabels=[f"{y_val:.3e}" for y_val in etaa]) 
        plt.xlabel(f"$\\gamma$")
        plt.ylabel(f"$\\eta$")

        plt.savefig(tuning_method+"_sgdm_MSE(eta,momentum).pdf")
    plt.show()
        
    """

    # sgd(eta,epochs)
    """
    np.random.seed(0)
    lamb = 0
    #size_batch = np.linspace(2, len(y_train)/5, 5, dtype=int)
    n_epochs_sgd = np.linspace(10, 200, 5, dtype=int)

    for tuning_method in ["", "AdaGrad", "RMSprop", "ADAM"]:

        i = 0

        momentum = 0.4

        if tuning_method == "ADAM":
            etaa = np.logspace(-3, -1, 10)
            n = len(etaa)
            momentum = 0
        else:
            etaa = np.linspace(0.1, 1, 10)
            n = len(etaa)

        MSE_sgdm = np.zeros((n,m))

        for eta in etaa:
            j = 0
            for epochs in n_epochs_sgd:
                gm.sgd(eta, epochs, size_batch, gamma=momentum, lamb=lamb, tuning_method=tuning_method)
                sgdm_pred = X @ gm.get_theta()

                MSE_sgdm[i,j] = mse(y, sgdm_pred)

                j += 1
            i += 1

        # Heatmaps

        plt.figure(figsize=(12,8))
        plt.title(tuning_method+f" MSE stochastic gradient descent $\\gamma$={momentum}")
        sns.heatmap(MSE_sgdm, annot=True, fmt='.5g',
                #vmax = 0.1, 
                cbar_kws={'label': "MSE"}, 
                xticklabels = [f"{x_val:.5g}" for x_val in n_epochs_sgd],
                yticklabels=[f"{y_val:.2e}" for y_val in etaa]) 
        plt.xlabel(f"Number of epochs")
        plt.ylabel(f"$\\eta$")
        plt.savefig(tuning_method+"_sgdm_MSE(eta,epochs).pdf")
    plt.show()
    """

    
    
    #MSE(iteration) for different minibatch sizes
    """
    np.random.seed(1)
    n_epochs = 200

    lamb = 0
    eta = 0.1
    momentum = 0.4
    for minibatch in [2, 5, 10, 80]:
        n_epochs_sgd = int(minibatch/len(y_train)*200)
        
        gm.sgd(eta=0.9, n_epochs=n_epochs_sgd, size_batch=minibatch, gamma=momentum, lamb=lamb, tuning_method="Plain")
        sgd_plain_momentum = gm.get_thetas()

        gm.sgd(eta=0.9, n_epochs=n_epochs_sgd, size_batch=minibatch, gamma=momentum, lamb=lamb, tuning_method="AdaGrad")
        sgd_AdaGrad = gm.get_thetas()

        gm.sgd(eta=0.9,n_epochs=n_epochs_sgd, size_batch=minibatch, gamma=momentum, lamb=lamb, tuning_method="RMSprop")
        sgd_RMSprop = gm.get_thetas()

        gm.sgd(eta=0.2, n_epochs=n_epochs_sgd, size_batch=minibatch, gamma=0, lamb=lamb, tuning_method="ADAM")
        sgd_ADAM = gm.get_thetas()

        # Plotting
        theta_dict = {'ADAM | $\\gamma = 0$': sgd_ADAM, 'RMSprop | $\\gamma = 0.4$': sgd_RMSprop, 'AdaGrad | $\\gamma = 0.4$': sgd_AdaGrad, 'Plain | $\\gamma = 0.4$': sgd_plain_momentum}
        
        sns.set_style("darkgrid")
        plt.figure()
        plt.title(f"Iterations with minibatch size {minibatch}")
        pl = plot.Plot(p)
        pl.plot_iter_MSE(theta_dict)
        plt.savefig("minibatch_"+str(minibatch)+"_MSE(iter).pdf")
    plt.show()

    """

    # Polynomial prediction and target plot
    #"""
    X_full = np.concatenate((X_train, X), axis=0)
    print(X_full.shape)
    target_full = np.concatenate((y_train, y))

    gm.sgd(eta=0.9, n_epochs=20, size_batch=10, gamma=0.4, lamb=0, tuning_method="AdaGrad")
    sgd_AdaGrad = gm.get_theta()
    print(sgd_AdaGrad.shape)
    y_pred = X_full @ sgd_AdaGrad

    sns.set_style("darkgrid")
    plt.title("Model using sgd with AdaGrad compared to target")
    plt.xlabel('x')
    plt.ylabel('y')
    sns.lineplot(x=X_full[:,1], y=target_full, linewidth=1, label="target polynomial")
    sns.lineplot(x=X_full[:,1], y=y_pred, linewidth=1, linestyle="--",label="model polynomial")
    plt.legend()
    plt.savefig("sgd_polynomial_fit.pdf")
    plt.show()
    #"""

    
    #MSE(iteration)
    """
    np.random.seed(1)
    n_epochs_sgd = 5
    n_epochs = 20

    lamb = 0
    eta = 0.1
    momentum = 0.4
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
        
        sns.set_style("darkgrid")
        plt.figure()
        plt.title(method)
        pl = plot.Plot(p)
        pl.plot_iter_MSE(theta_dict)
        plt.savefig(method+"MSE(iter).pdf")
    plt.show()
    """
from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload
import time

from sklearn.model_selection import  train_test_split 
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import sklearn as skl 

import jens_neural_network as neural_network
import jens_scores as scores
import jens_activation as activation
import jens_optimizer as optimizer
import analysis
from poly_data import PolyData

if __name__ == '__main__':
    reload(neural_network)
    reload(scores)
    reload(activation)
    reload(optimizer)
    reload(analysis)

np.random.seed(5)

# Load polynomial data
polydata = PolyData(1000)

X_train, Y_train = polydata.get_train()
X_test, Y_test = polydata.get_test()

noise = 0
Y_test += noise*np.random.normal(0,1,Y_test.shape)
Y_train += noise*np.random.normal(0,1,Y_train.shape)


x_train, x_test = X_train[:,1][:,np.newaxis], X_test[:,1][:,np.newaxis]
y_train, y_test = Y_train[:,np.newaxis], Y_test[:,np.newaxis]



parameters = {
    "X_train": x_train,
    "y_train": y_train,
    "X_test": x_test,
    "y_test": y_test,
    "eta": 0.001, 
    "depth": 1,
    "width": 5,
    "n_output_nodes": 1,
    "cost_score": 'mse',
    "activation_hidden": 'sigmoid',
    "activation_output": "none",
    "gamma": 0,
    "lambd": 0,
    "tuning_method": 'none',
    "n_minibatches": 10,
    "epochs": 2000
    }



"""
a = analysis.Analysis(**parameters)
a.width = np.linspace(1, 7, 7, dtype=int)
a.depth = 10*np.linspace(0, 5, 6, dtype=int)
a.plot_heatmap("cost")
"""

b = analysis.Analysis(**parameters)

eta = np.linspace(0.005,0.4,5) 
# Eta = 0.4 resulting in weights imploding
#b.eta = eta
b.eta = 0.4
lambd = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
b.lambd = lambd
b.plot_model()

b.eta = eta
b.plot_heatmap("cost", filename="_sigmoid_MSE(eta,lmb).pdf")



b.lambd = lambd[::2]
b.eta = 0.01
b.plot_score("cost")


c = analysis.Analysis(**parameters)
c.activation_hidden = "relu"
c.eta = np.logspace(-7, -1, 7)
c.lambd = np.linspace(0.001, 0.1, 7)
c.plot_heatmap("cost", filename="_relu_MSE(eta,lmb).pdf")

d = analysis.Analysis(**parameters)
d.activation_hidden = "leaky_relu"
d.eta = np.logspace(-7, -1, 7)
d.lambd = np.linspace(0.001, 0.1, 7)
d.plot_heatmap("cost", filename="_leaky_relu_MSE(eta,lmb).pdf")


"""
    np.random.seed(5)
    # Load the data
    cancer = load_breast_cancer()
    targets = cancer.target[:,np.newaxis]
    test_size = 0.2
    features = cancer.feature_names
    X_train, X_test, y_train, y_test = train_test_split(cancer.data,targets,random_state=0, test_size=test_size)

    # Scale data with mean and std
    scaler = skl.preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    parameters = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "eta": 0.001, 
        "depth": 1 ,
        "width": 10,
        "n_output_nodes": 1,
        "cost_score": 'cross_entropy',
        "activation_hidden": 'sigmoid',
        "activation_output": 'sigmoid',
        "gamma": 0.9,
        "lambd": 0,
        "tuning_method": 'none',
        "n_minibatches": 20,
        "epochs": 400
        }

    a = analysis.Analysis(**parameters)



#  ____            _         _ 
# |  _ \ __ _ _ __| |_    __| |
# | |_) / _` | '__| __|  / _` |
# |  __/ (_| | |  | |_  | (_| |
# |_|   \__,_|_|   \__|  \__,_|
    # a = analysis.Analysis(**parameters)
    # # find optimal number of n_minibatches
    # a.epochs = 400
    # a.n_minibatches = [1, 5, 10, 15, 20]
    # a.plot_score('accuracy')

    # # Find optimal values fro lmabda and eta
    # a = analysis.Analysis(**parameters) # XXX: in progress
    # a.epochs = 200  # XXX: Reduced number of ephcos to speedup
    # a.lambd = np.logspace(-5, 0, 6)
    # a.lambd[-1] = 0
    # a.eta = np.logspace(-5, -1, 5)
    # a.plot_heatmap('accuracy')


    # Use new best parameters for lambd and eta
    parameters = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "eta": 0.1,  # XXX: changed
        "depth": 1 ,
        "width": 10,
        "n_output_nodes": 1,
        "cost_score": 'cross_entropy',
        "activation_hidden": 'sigmoid',
        "activation_output": 'sigmoid',
        "gamma": 0.9,
        "lambd": 0.1,  # XXX: changed
        "tuning_method": 'none',
        "n_minibatches": 20,
        "epochs": 200 # XXX: changed
        }

    # # Example: heatmap with width and depth
    # a = analysis.Analysis(**parameters)
    # a.width = [5, 10, 15, 20]
    # a.depth = [1, 2, 3]
    # a.plot_heatmap('accuracy')


    # # Example: heatmap with width and depth
    # a = analysis.Analysis(**parameters)
    # a.lambd = 0
    # a.width = [5, 10, 15, 20]
    # a.depth = [1, 2, 3]
    # a.plot_heatmap('accuracy')


    # Use new parameters for network dept and width
    parameters = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "eta": 0.1,  
        "depth": 2 ,  # XXX: changed 
        "width": 5,  # XXX: changed
        "n_output_nodes": 1,
        "cost_score": 'cross_entropy',
        "activation_hidden": 'sigmoid',
        "activation_output": 'sigmoid',
        "gamma": 0.9,
        "lambd": 0.1, 
        "tuning_method": 'none',
        "n_minibatches": 20,
        "epochs": 200 
        }


    # a = analysis.Analysis(**parameters)
    # a.lambd = 0
    # a.eta = 0.001 # Low eta to compare convergence rate
    # a.lambd = [0, 1e-3, 1e-2, 0.1]
    # a.plot_score('accuracy')


    # Test different learning rate tuning 
    a = analysis.Analysis(**parameters)
    a.lambd = 0
    a.eta = 0.001 # Low eta to compare convergence rate
    a.tuning_method = ['none', 'adam', 'rms_prop', 'adagrad']
    a.plot_score('accuracy')

    

    # a.activation_hidden = ['sigmoid', 'relu', 'leaky_relu']
"""
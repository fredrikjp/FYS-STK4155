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

from sklearn.neural_network import MLPRegressor


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



# # sklearn activation sigmoid
"""
gamma = 0

lambd = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
etaa = np.linspace(0.1, 2, 5) 

n_minibatches = 10
batch_size = int(len(Y_train)/n_minibatches)
n_epochs = 2000

n = len(lambd)
m = len(etaa)

MSE = np.zeros((m,n))

j=0
for eta in etaa:
    i = 0
    for lmb in lambd:
        NN_sigmoid = MLPRegressor(hidden_layer_sizes=(5), activation="logistic", solver="sgd", alpha=lmb, batch_size = batch_size, learning_rate_init = eta, momentum = 0, max_iter=n_epochs ,n_iter_no_change=2000)
        NN_sigmoid.fit(x_train, y_train.ravel())
        MSE[j, i] = NN_sigmoid.loss_
        i+=1
    j+=1
plt.figure(figsize=(12,8))
plt.title("Sklearn neural network with activation function sigmoid train MSE")
sns.heatmap(MSE, annot=True, fmt='.5g',
        vmax = 0.1, 
        cbar_kws={'label': "MSE"}, 
        xticklabels = [f"{x_val:.5g}" for x_val in lambd],
        yticklabels=[f"{y_val:.5g}" for y_val in etaa]) 
plt.ylabel(f"$\\eta$")
plt.xlabel(f"$\\lambda$")
plt.savefig("train_sklearn_sigmoid_NN_MSE(eta,lmb).pdf")
plt.show()

"""


# # sklearn activation relu
"""
etaa = np.logspace(-6, 0, 7)



n = len(lambd)
m = len(etaa)

MSE = np.zeros((m,n))
j=0
for eta in etaa:
    i = 0
    for lmb in lambd:
        NN_relu = MLPRegressor(hidden_layer_sizes=(5), activation="relu", solver="sgd", alpha=lmb, batch_size = batch_size, learning_rate_init = eta, momentum = 0, max_iter=n_epochs ,n_iter_no_change=2000)
        NN_relu.fit(x_train, y_train.ravel())
        MSE[j, i] = NN_relu.loss_
        i+=1
    j+=1
plt.figure(figsize=(12,8))
plt.title("Sklearn neural network with activation function relu train MSE")
sns.heatmap(MSE, annot=True, fmt='.5g',
        vmax = 0.1, 
        cbar_kws={'label': "MSE"}, 
        xticklabels = [f"{x_val:.5g}" for x_val in lambd],
        yticklabels=[f"{y_val:.5g}" for y_val in etaa]) 
plt.ylabel(f"$\\eta$")
plt.xlabel(f"$\\lambda$")
plt.savefig("train_sklearn_relu_NN_MSE(eta,lmb).pdf")
plt.show()

"""




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





# NN sigmoid
""" 

a = analysis.Analysis(**parameters)
eta = np.linspace(0.05, 0.5, 5) 
# Eta = 0.4 resulting in weights imploding
#a.eta = eta
a.eta = 0.4
#a.plot_model()

lambd = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
a.lambd = lambd
#a.plot_model()

a.eta = eta
a.lambd = lambd
a.plot_heatmap("cost", filename="_sigmoid_MSE(eta,lmb).pdf")
"""

"""
b = analysis.Analysis(**parameters)
b.eta = 0.3875
b.width = np.linspace(0, 4, 5, dtype=int)
b.depth = 10*np.linspace(0, 4, 5, dtype=int)
b.plot_heatmap("cost")
"""

# NN relu
"""
etaa = np.logspace(-7, -1, 7)

c = analysis.Analysis(**parameters)
c.activation_hidden = "relu"
c.eta = etaa
c.lambd = lambd
c.plot_heatmap("cost", filename="_relu_MSE(eta,lmb).pdf")
"""

# NN leaky relu
"""
d = analysis.Analysis(**parameters)
d.activation_hidden = "leaky_relu"
d.eta = etaa
d.lambd = lambd
d.plot_heatmap("cost", filename="_leaky_relu_MSE(eta,lmb).pdf")
"""

# Resulting NN polynomial versus target
#"""
e = analysis.Analysis(**parameters)
e.activation_hidden = "sigmoid"
e.eta = 0.1
e.lambd = 0
e.plot_model(filename="NN_polyfit_result.pdf")
#"""
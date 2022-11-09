from Neural_Network import NeuralNetwork
from poly_data import PolyData
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

polydata = PolyData(1000)

X_train, Y_train = polydata.get_train()
X_test, Y_test = polydata.get_test()


X_train, X_test = X_train, X_test
Y_train, Y_test = np.array([Y_train]).T, Y_test

"""
# MSE testing
N = NeuralNetwork(X_train, Y_train, eta=0.0001, activation_function="sigmoid", hiddenLayerSize=5, n_hiddenLayers=1)
for i in range(10000):
    N.forward()
    print(np.mean((N.ao.ravel()-Y_train.ravel())**2))
    N.backpropagation()

y_pred = N.forward_out(X_test)

plt.scatter(X_test[:,1], y_pred)
plt.scatter(X_test[:,1], Y_test.ravel())
plt.show()
"""

"""
n = 6
eta = np.logspace(-1,-5,n)
lmbd = np.logspace(-1,-5,n)
MSE = np.zeros((n,n))
i = 0
for et in eta:
    j=0
    for lm in lmbd:
        NN = NeuralNetwork(X_train, Y_train, eta=et, lmbd=lm)
        NN.train()
        Y_pred = NN.forward_out(X_train)
        mse = np.mean((Y_pred-Y_train)**2)
        MSE[i,j] = mse
        j+=1
    i+=1
ax = sns.heatmap(MSE, annot=True, vmax=1, fmt=".3g", xticklabels=[f"{lm:.1e}" for lm in lmbd], yticklabels=[f"{et:.1e}" for et in eta], cbar_kws={"label": "MSE"})
ax.set(xlabel=r"$\lambda$", ylabel=r"$\eta$", title=r"MSE($\eta$, $\lambda$)")
plt.tight_layout()
plt.show()
"""
np.random.seed(0)


NN = NeuralNetwork(X_train, Y_train, activation_function="sigmoid")

NN.train()
y_pred = NN.forward_out(X_test)

plt.scatter(X_test[:,1], Y_test.ravel(), label = "Target")
plt.scatter(X_test[:,1], y_pred, label = "Model")

plt.legend()
plt.title("Sigmoid")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

NN = NeuralNetwork(X_train, Y_train, activation_function="relu") 

NN.train()
y_pred = NN.forward_out(X_test)

plt.scatter(X_test[:,1], Y_test.ravel(), label = "Target")
plt.scatter(X_test[:,1], y_pred, label = "Model")
plt.legend()
plt.title("ReLU")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

NN = NeuralNetwork(X_train, Y_train, activation_function="leaky relu")

NN.train()
y_pred = NN.forward_out(X_test)

plt.scatter(X_test[:,1], Y_test.ravel(), label = "Target")
plt.scatter(X_test[:,1], y_pred, label = "Model")
plt.legend()
plt.title("Leaky ReLU")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

import numpy as np
from Neural_Network import NeuralNetwork
from poly_data import PolyData
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier

np.random.seed(1)

data = load_breast_cancer()
X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, train_size=0.8)
Y_train = np.array([Y_train]).T

np.random.seed(0)


#print(np.sum(N.binary_predict(N.forward_out(X_test)) == np.array([Y_test]).T)/len(Y_test))
epochs = 1000
N1 = NeuralNetwork(X_train, Y_train, X_test=X_test, Y_test=Y_test, activation_function="sigmoid", output_function="sigmoid", cost_function="BCE", eta=0.001, n_hiddenLayers=3, hiddenLayerSize=10, epochs=epochs, normalization=True, accuracy=True, batch_size=50, gamma=0.8, lmbd=0)
N1.train()


np.random.seed(0)

N2 = NeuralNetwork(X_train, Y_train, X_test=X_test, Y_test=Y_test, activation_function="relu", output_function="sigmoid", cost_function="BCE", eta=0.001, n_hiddenLayers=3, hiddenLayerSize=10, epochs=epochs, normalization=True, accuracy=True, batch_size=50, gamma=0.8)
N2.train()

np.random.seed(0)

N3 = NeuralNetwork(X_train, Y_train, X_test=X_test, Y_test=Y_test, activation_function="leaky relu", output_function="sigmoid", cost_function="BCE", eta=0.001, n_hiddenLayers=3, hiddenLayerSize=10, epochs=epochs, normalization=True, accuracy=True, batch_size=50, gamma=0.8)
N3.train()


x = np.linspace(0,epochs,epochs)


plt.figure()
plt.plot(x, N1.acc_train, label="sigmoid train")
plt.plot(x, N1.acc_test, label="sigmoid test")
plt.legend()

plt.figure()
plt.title("Train accuracy")
plt.plot(x, N1.acc_train, label="sigmoid")
plt.plot(x, N2.acc_train, label="relu")
plt.plot(x, N3.acc_train, label="leaky relu")
plt.legend()

plt.figure()
plt.title("Test accuracy")
plt.plot(x, N1.acc_test, label="sigmoid")
plt.plot(x, N2.acc_test, label="relu")
plt.plot(x, N3.acc_test, label="leaky relu")
plt.legend()
plt.show()


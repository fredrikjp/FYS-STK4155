from Neural_Network import NeuralNetwork
from poly_data import PolyData
import numpy as np
import matplotlib.pyplot as plt

polydata = PolyData(100)

X_train, y_train = polydata.get_train()
X_test, y_test = polydata.get_test()

x_train, x_test = np.array([X_train[:,1]]).T, X_test[:,1]

print(x_train.shape)
#X_train, X_test = X_train[:,1], X_test[:,1]
NN = NeuralNetwork(x_train, y_train)
print(NN.forward(x_train))
NN.backpropagation()
print(NN.forward(x_train))
"""
y_pred = NN.predict(x_test)
print(y_pred)
print(y_pred.shape)
print(y_test.shape)

plt.scatter(x_test, y_pred)
plt.show()
plt.scatter(x_test, y_test)
plt.show()
"""

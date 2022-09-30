from ctypes import OleDLL
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from OLS import OLS_Frank

np.random.seed(17)

n = 20
n_boostraps = 100
maxdegree = 12
noise = 0.2

# Make data set.
x = np.sort(np.random.uniform(0,1,n))
y = np.sort(np.random.uniform(0,1,n))
OLS = OLS_Frank(x,y,maxdegree,noise)
X_test = OLS[3]
X_train = OLS[0]
y_train = OLS[1]
y_test = OLS[4].reshape(-1, 1)





error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)

for degree in range(maxdegree):
    y_pred = np.empty((y_test.shape[0], n_boostraps))
    l = int((degree+1)*(degree+2)/2)		# Number of elements in beta

    for i in range(n_boostraps):
        #OLS = OLS_Frank(x,y,degree,noise)
        #OLSbeta = OLS[4]
        X_ , y_ = resample(X_train[:,:l], y_train)
        OLSbeta = np.linalg.pinv(X_.T @ X_) @ X_.T @ y_
        ytilde_test = X_test[:,:l] @ OLSbeta
        y_pred[:,i] = ytilde_test
    polydegree[degree] = degree
    error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )
    #print('Polynomial degree:', degree)
    #print('Error:', error[degree-1])
    #print('Bias^2:', bias[degree-1])
    #print('Var:', variance[degree-1])
    #print('{} >= {} + {} = {}'.format(error[degree-1], bias[degree-1], variance[degree-1], bias[degree-1]+variance[degree-1]))

plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
plt.show()
import numpy as np

class NeuralNetwork:
    def __init__(self, X_data, Y_data, 
                    activation_function = "sigmoid",
                    n_categories = 1,
                    n_hiddenLayers = 0, 
                    hiddenLayerSize = 20, 
                    eta = 0.0025, 
                    lmbd = 0,
                    batch_size = 10, 
                    epochs = 10,
                    iterations = 1000):
        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.X_data = X_data
        self.Y_data = Y_data
        self.f = activation_function
        self.n_categories = n_categories
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hiddenLayers = n_hiddenLayers
        self.hiddenLayerSize = hiddenLayerSize
        self.batch_size = batch_size
        self.input_size = len(X_data)
        self.output_size = len(Y_data)
        self.epochs = epochs
        self.iterations = iterations
        self.eta = eta
        self.lmbd = lmbd

        if self.n_hiddenLayers > 0:
            # input weights 
            self.Wi = np.random.randn(self.n_features, self.hiddenLayerSize)
            # rest of hidden layer weights and biases
            self.Wh = np.random.randn(self.hiddenLayerSize, self.hiddenLayerSize*(self.n_hiddenLayers-1))
            self.bh = np.zeros((self.n_hiddenLayers, self.hiddenLayerSize)) + 0.01
            # output weights and biases
            self.Wo = np.random.randn(self.hiddenLayerSize, self.n_categories)
            self.bo = np.zeros(self.n_categories) + 0.01
        else:
            # output weights and biases
            self.Wo = np.random.randn(self.n_features, self.n_categories)
            self.bo = np.zeros(self.n_categories) + 0.01
            
    def __f(self, x):
        if self.f == "sigmoid":
            return 1/(1+np.exp(-x))

        if self.f == "relu":
            return np.maximum(0, x)

        if self.f == "leaky relu":
            return np.where(x > 0, x, 0.01*x)
    
    def __df(self, x):
        if self.f == "sigmoid":
            return x * (1 - x)

        if self.f == "relu":
            return np.where(x > 0, 1, 0)

        if self.f == "leaky relu":
            return np.where(x > 0, 1, 0.01)


    def forward(self):
        k = self.hiddenLayerSize
        l = self.X_data.shape[0]
        o = self.Y_data.shape[0]

        if self.n_hiddenLayers == 0:
            self.zo = np.matmul(self.X_data, self.Wo) + self.bo
            self.ao = self.zo  
            return 

        # hidden layer a and z
        self.ah = np.zeros((self.n_hiddenLayers * l, k))
        self.zh = np.zeros((self.n_hiddenLayers * l, k))
        # output a and z
        self.ao = np.zeros((self.n_categories, o))
        self.zo = np.zeros((self.n_categories, o))
        
        self.zh[0:l] = np.matmul(self.X_data, self.Wi) + self.bh[0]
        self.ah[0:l] = self.__f(self.zh[0:l])
        for i in range(1, self.n_hiddenLayers):
            self.zh[l*i:l*(i+1)] = np.matmul(self.ah[l*(i-1):l*i], self.Wh[:,k*(i-1):k*i]) + self.bh[i]
            self.ah[l*i:l*(i+1)] = self.__f(self.zh[l*i:l*(i+1)])

        self.zo = np.matmul(self.ah[(self.n_hiddenLayers-1)*l:], self.Wo) + self.bo
        self.ao = self.zo    
    
    def forward_out(self, X):
        k = self.hiddenLayerSize

        if self.n_hiddenLayers == 0:
            zo = np.matmul(X, self.Wo) + self.bo
            ao = zo  
            return ao

        zh = np.matmul(X, self.Wi) + self.bh[0]
        ah = self.__f(zh)
        for i in range(1, self.n_hiddenLayers):
            zh = np.matmul(ah, self.Wh[:,k*(i-1):k*i]) + self.bh[i]
            ah = self.__f(zh)

        zo = np.matmul(ah, self.Wo) + self.bo
        ao = zo    
        return ao    
    
    def backpropagation(self):


        l = self.X_data.shape[0]
        k = self.hiddenLayerSize

        if self.n_hiddenLayers == 0:
            delta = self.ao - self.Y_data
            hidden_weights_gradient = np.matmul(self.X_data.T, delta)
            if self.lmbd > 0.0:
                hidden_weights_gradient += self.lmbd * self.Wo
            hidden_bias_gradient = np.sum(delta, axis=0)
            self.Wo -= self.eta * hidden_weights_gradient
            self.bo -= self.eta * hidden_bias_gradient
            return 
        
        df = self.__df(self.ah)
        delta = (self.ao - self.Y_data) 
        self.output_weights_gradient = np.matmul(self.ah[(self.n_hiddenLayers-1)*l:].T, delta)
        self.output_bias_gradient = np.sum(delta, axis=0)
        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.Wo

        self.Wo -= self.eta * self.output_weights_gradient
        self.bo -= self.eta * self.output_bias_gradient
        if self.n_hiddenLayers > 1:
            delta = np.matmul(delta, self.Wo.T) * df[(self.n_hiddenLayers-1)*l:]
            hidden_weights_gradient = np.matmul(self.ah[(self.n_hiddenLayers-2)*l:(self.n_hiddenLayers-1)*l].T, delta) 
            hidden_bias_gradient = np.sum(delta, axis=0)
            if self.lmbd > 0.0:
                hidden_weights_gradient += self.lmbd * self.Wh[:,k*(self.n_hiddenLayers-2):]

            self.Wh[:,k*(self.n_hiddenLayers-2):] -= self.eta * hidden_weights_gradient
            self.bh[-1] -= self.eta * hidden_bias_gradient

        for i in range(self.n_hiddenLayers - 2, 0, -1):
            delta = np.matmul(delta, self.Wh[:,k*i:k*(i+1)].T) * df[i*l:(i+1)*l]
            hidden_weights_gradient = np.matmul(self.ah[(i-1)*l:i*l].T, delta) 
            if self.lmbd > 0.0:
                hidden_weights_gradient += self.lmbd * self.Wh[:,k*(i-1):k*i]
            hidden_bias_gradient = np.sum(delta, axis=0)
            self.Wh[:,k*(i-1):k*i] -= self.eta * hidden_weights_gradient
            self.bh[i] -= self.eta * hidden_bias_gradient
        if self.n_hiddenLayers > 1:
            delta = np.matmul(delta, self.Wh[:,:k].T) * df[:l]
        else:
            delta = np.matmul(delta, self.Wo.T) * df[:l]
        hidden_weights_gradient = np.matmul(self.X_data.T, delta)
        if self.lmbd > 0.0:
                hidden_weights_gradient += self.lmbd * self.Wi
        hidden_bias_gradient = np.sum(delta, axis=0)
        self.Wi -= self.eta * hidden_weights_gradient
        self.bh[0] -= self.eta * hidden_bias_gradient
        
        

    def train(self):
        data_indices = np.arange(self.input_size)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints without replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]
                self.forward()
                self.backpropagation()

if __name__ == "__main__":
    from poly_data import PolyData
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    polydata = PolyData(1000)

    X_train, Y_train = polydata.get_train()
    X_test, Y_test = polydata.get_test()

    
    X_train, X_test = X_train, X_test
    Y_train, Y_test = np.array([Y_train]).T, Y_test
    
    # MSE testing
    """
    N = NeuralNetwork(X_train, Y_train, eta=0.0000025)
    for i in range(100):
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
   
    NN = NeuralNetwork(X_train, Y_train, lmbd=0.0, activation_function="sigmoid")

    NN.train()
    y_pred = NN.forward_out(X_test)

    plt.scatter(X_test[:,1], Y_test.ravel(), label = "Target")
    plt.scatter(X_test[:,1], y_pred, label = "Model")

    plt.legend()
    plt.title("Sigmoid")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    NN = NeuralNetwork(X_train, Y_train, lmbd=0.0, activation_function="relu")

    NN.train()
    y_pred = NN.forward_out(X_test)

    plt.scatter(X_test[:,1], Y_test.ravel(), label = "Target")
    plt.scatter(X_test[:,1], y_pred, label = "Model")
    plt.legend()
    plt.title("ReLU")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
 
    NN = NeuralNetwork(X_train, Y_train, lmbd=0.0, activation_function="leaky relu")

    NN.train()
    y_pred = NN.forward_out(X_test)
    
    plt.scatter(X_test[:,1], Y_test.ravel(), label = "Target")
    plt.scatter(X_test[:,1], y_pred, label = "Model")
    plt.legend()
    plt.title("Leaky ReLU")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


import numpy as np

class NeuralNetwork:
    def __init__(self, X_data, Y_data, 
                    activation_function = "sigmoid",
                    output_function = None,
                    n_categories = 1,
                    n_hiddenLayers = 1, 
                    hiddenLayerSize = 20, 
                    eta = 0.0025, 
                    lmbd = 0,
                    batch_size = 10, 
                    epochs = 10,
                    iterations = 100,
                    cost_function = "SE"):
        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.X_data = X_data
        self.Y_data = Y_data
        self.f = activation_function
        self.g = output_function
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
        self.C = cost_function

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
        if self.f == None:
            return x

        if self.f == "sigmoid":
            return 1/(1+np.exp(-x))

        if self.f == "relu":
            return np.maximum(0, x)

        if self.f == "leaky relu":
            return np.where(x > 0, x, 0.01*x)
    
    def __df(self, x):
        if self.f == None:
            return np.ones_like(x)

        if self.f == "sigmoid":
            return x * (1 - x)

        if self.f == "relu":
            return np.where(x > 0, 1, 0)

        if self.f == "leaky relu":
            return np.where(x > 0, 1, 0.01)

    def __g(self, x):
        if self.g == None:
            return x
        if self.g == "sigmoid":
            return 1/(1+np.exp(-x))
    
    def __dg(self, x):
        if self.g == None:
            return 1
        if self.g == "sigmoid":
            return x * (1 - x)
    
    # Cost function derivative with respect to model output
    def __dC(self, x):
        if self.C == "SE":
            return (x-self.Y_data)
        if self.C == "BCE":
            return -self.Y_data/x + (1-self.Y_data)/(1-x)
            

    
    def forward(self):
        k = self.hiddenLayerSize
        l = self.X_data.shape[0]
        o = self.Y_data.shape[0]

        if self.n_hiddenLayers == 0:
            self.zo = np.matmul(self.X_data, self.Wo) + self.bo
            self.ao = self.__g(self.zo)  
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
        self.ao = self.__g(self.zo)  
    
    def forward_out(self, X):
        k = self.hiddenLayerSize

        if self.n_hiddenLayers == 0:
            zo = np.matmul(X, self.Wo) + self.bo
            ao = self.__g(zo)  
            return ao

        zh = np.matmul(X, self.Wi) + self.bh[0]
        ah = self.__f(zh)
        for i in range(1, self.n_hiddenLayers):
            zh = np.matmul(ah, self.Wh[:,k*(i-1):k*i]) + self.bh[i]
            ah = self.__f(zh)

        zo = np.matmul(ah, self.Wo) + self.bo
        ao = self.__g(zo)   
        return ao 

    def binary_predict(self, x):
        return np.where(x < 0.5, 0, 1)   
    
    def backpropagation(self):
        l = self.X_data.shape[0]
        k = self.hiddenLayerSize

        dg = self.__dg(self.ao)

        if self.n_hiddenLayers == 0:
            delta = self.__dC(self.ao) * dg
            hidden_weights_gradient = np.matmul(self.X_data.T, delta)
            if self.lmbd > 0.0:
                hidden_weights_gradient += self.lmbd * self.Wo
            hidden_bias_gradient = np.sum(delta, axis=0)
            self.Wo -= self.eta * hidden_weights_gradient
            self.bo -= self.eta * hidden_bias_gradient
            return 
        
        df = self.__df(self.ah)
        delta = self.__dC(self.ao) * dg 
        #print(np.mean((self.ao-self.Y_data)**2))
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
                    data_indices, size=self.batch_size, replace=True
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
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer
    from sklearn.neural_network import MLPClassifier

    data = load_breast_cancer()
    print(dir(data))
    
    X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, train_size=0.8)
    Y_train = np.array([Y_train]).T

    np.random.seed(0)
    N = NeuralNetwork(X_train, Y_train, activation_function="sigmoid", output_function="sigmoid", cost_function="BCE", eta=0.000025, n_hiddenLayers=5, hiddenLayerSize=100, iterations=10)
    accuracy = []
    n = 100
    """
    for i in range(n):
        N.train()
        print(np.sum(N.binary_predict(N.forward_out(X_train)) == Y_train)/len(Y_train))
    
    """
    for i in range(n):
        N.forward()
        accuracy.append(np.sum(N.binary_predict(N.ao) == Y_train)/len(Y_train))
        N.backpropagation()

    x = np.linspace(0,n,n)
    plt.plot(x, accuracy)
    plt.show()


    N = NeuralNetwork(X_train, Y_train, activation_function="sigmoid", output_function="sigmoid", cost_function="BCE", eta=0.000025, n_hiddenLayers=5, hiddenLayerSize=100, iterations=10)
    accuracy = []
    sklearn_accuracy = []
    dnn = MLPClassifier(hidden_layer_sizes=100, activation="logistic", learning_rate_init=0.000025, max_iter=1, alpha=0)
    for i in range(n):
        dnn.fit(X_train, Y_train.reshape(-1,1))
        
        sklearn_accuracy.append(np.sum(dnn.predict(X_test)==Y_test)/len(Y_test))
        N.forward()
        accuracy.append(np.sum(N.binary_predict(N.forward_out(X_test)) == np.array([Y_test]).T)/len(Y_test))
        N.backpropagation()

    x = np.linspace(0,n,n)
    plt.plot(x, accuracy)
    plt.plot(x, sklearn_accuracy)
    plt.show()

import numpy as np

class NeuralNetwork:
    def __init__(self, X_data, Y_data, 
                    n_hiddenLayers = 10, 
                    hiddenLayerSize = 20, 
                    eta = 0.1, 
                    batch_size = 10, 
                    epochs = 10,
                    iterations = 100):
        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.X_data = X_data
        self.Y_data = Y_data
        self.n_categories = 1
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

        # input weights 
        self.Wi = np.random.randn(self.n_features, self.hiddenLayerSize)
        # rest of hidden layer weights and biases
        self.Wh = np.random.randn(self.hiddenLayerSize, self.hiddenLayerSize*(self.n_hiddenLayers-1))
        self.bh = np.zeros((self.n_hiddenLayers, self.hiddenLayerSize)) + 0.01
        # output weights and biases
        self.Wo = np.random.randn(self.hiddenLayerSize, self.n_categories)
        self.bo = np.zeros(self.n_categories) + 0.01
        
    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def forward(self):
        k = self.hiddenLayerSize
        l = self.X_data.shape[0]
        o = self.Y_data.shape[0]

        # hidden layer a and z
        self.ah = np.zeros((self.n_hiddenLayers * l, k))
        self.zh = np.zeros((self.n_hiddenLayers * l, k))
        # output a and z
        self.ao = np.zeros((1, o))
        self.zo = np.zeros((1, o))
        
        self.zh[0:l] = np.matmul(self.X_data, self.Wi) + self.bh[0]
        self.ah[0:l] = self.__sigmoid(self.zh[0:l])
        for i in range(1, self.n_hiddenLayers):
            self.zh[l*i:l*(i+1)] = np.matmul(self.ah[l*(i-1):l*i], self.Wh[:,k*(i-1):k*i]) + self.bh[i]
            self.ah[l*i:l*(i+1)] = self.__sigmoid(self.zh[l*i:l*(i+1)])

        self.zo = np.matmul(self.ah[(self.n_hiddenLayers-1)*l:], self.Wo) + self.bo
        self.ao = self.zo    
    
    def forward_out(self, X):
        k = self.hiddenLayerSize

        zh = np.matmul(X, self.Wi) + self.bh[0]
        ah = self.__sigmoid(zh)
        for i in range(1, self.n_hiddenLayers):
            zh = np.matmul(ah, self.Wh[:,k*(i-1):k*i]) + self.bh[i]
            ah = self.__sigmoid(zh)

        zo = np.matmul(ah, self.Wo) + self.bo
        ao = zo    
        return ao    
    
    def backpropagation(self):

        d = self.X_data.shape[0]

        d_sigmoid = self.ah * (1 - self.ah)
        l = self.X_data.shape[0]
        delta_o = (self.ao - self.Y_data) 
        self.output_weights_gradient = np.matmul(self.ah[(self.n_hiddenLayers-1)*l:].T, delta_o)
        self.output_bias_gradient = np.sum(delta_o, axis=0)
        self.Wo -= self.eta/d * self.output_weights_gradient
        self.bo -= self.eta/d * self.output_bias_gradient

        k = self.hiddenLayerSize
        delta = np.matmul(delta_o, self.Wo.T) * d_sigmoid[(self.n_hiddenLayers-1)*l:]
        hidden_weights_gradient = np.matmul(self.ah[(self.n_hiddenLayers-2)*l:(self.n_hiddenLayers-1)*l].T, delta) 
        hidden_bias_gradient = np.sum(delta, axis=0)
        self.Wh[:,k*(self.n_hiddenLayers-2):] -= self.eta/d * hidden_weights_gradient
        self.bh[-1] -= self.eta/d * hidden_bias_gradient

        for i in range(self.n_hiddenLayers - 2, 0, -1):
            delta = np.matmul(delta, self.Wh[:,k*i:k*(i+1)].T) * d_sigmoid[i*l:(i+1)*l]
            hidden_weights_gradient = np.matmul(self.ah[(i-1)*l:i*l].T, delta) 
            hidden_bias_gradient = np.sum(delta, axis=0)
            self.Wh[:,k*(i-1):k*i] -= self.eta/d * hidden_weights_gradient
            self.bh[i] -= self.eta/d * hidden_bias_gradient
        delta_i = np.matmul(delta, self.Wh[:,:k].T) * d_sigmoid[:l]
        hidden_weights_gradient = np.matmul(self.X_data.T, delta_i)
        hidden_bias_gradient = np.sum(delta_i, axis=0)
        self.Wi -= self.eta/d * hidden_weights_gradient
        self.bh[0] -= self.eta/d * hidden_bias_gradient
        
        

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
    
    polydata = PolyData(1000)

    X_train, Y_train = polydata.get_train()
    X_test, Y_test = polydata.get_test()

    #X_train, X_test = np.array([X_train[:,1]]).T, np.array([X_test[:,1]]).T
    #Y_train, Y_test = np.array([Y_train]).T, np.array([Y_test]).T
    
    X_train, X_test = X_train, X_test
    Y_train, Y_test = np.array([Y_train]).T, Y_test
    

    NN = NeuralNetwork(X_train, Y_train)
    NN.train()
    y_pred = NN.forward_out(X_test)

    plt.scatter(X_test[:,1], y_pred)
    plt.scatter(X_test[:,1], Y_test.ravel())
    plt.show()

    NN = NeuralNetwork(X_train, Y_train, )
    NN.train()
    y_pred = NN.forward_out(X_test)

    plt.scatter(X_test[:,1], y_pred)
    plt.scatter(X_test[:,1], Y_test.ravel())
    plt.show()

    """
    print(NN.forward(X_train))
    NN.backpropagation()
    print(NN.forward(X_train))
    """
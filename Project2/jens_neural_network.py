import abc
from dataclasses import dataclass, field
import sys
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import poly_data
import numpy as np
import pprint
import time
import copy
import autograd.numpy as jnp  # Thinly-wrapped numpy
#import jax.numpy as jnp 
from autograd import grad
from importlib import reload
import logging
import jens_activation as activation
import jens_optimizer as optimizer
import jens_scores as scores



logging.basicConfig(format='%(message)s', filename='./flow.log', level=logging.DEBUG)
logging.getLogger().disabled = True

@dataclass
class Layer(abc.ABC):

    @abc.abstractmethod
    def init_bias_and_weigths(): 
        " Initialize bias and weigts in layer "

    @abc.abstractmethod
    def get_output() -> np.ndarray:
        " Returns output from layer "

    @abc.abstractmethod
    def feed_forward(stdout: bool = False) -> None:
        " Calculates ouput from layer "

    @abc.abstractmethod
    def update_weights(initial_optimizer_ob: optimizer.Optimizer) -> None: 
        """ Calculates derivate of cost function with respect to weigts and bias for current layer """

@dataclass
class InputLayer(Layer):
    n_nodes: int # n_nodes = n_features
    X: np.ndarray = field(init=False, repr=False)  # Desing matrix with size (n_data, n_features)

    W = None
    b = None

    prev_Layer = None
    next_Layer = object()

    def init_bias_and_weigths(self):
        pass

    def set_optimizer(self, op: optimizer.Optimizer): 
        return

    def get_output(self): 
        return self.X

    def feed_forward(self, X: np.ndarray):
        logging.info('================== InputLayer.feed_forward ===============')
        self.X = X

        logging.info(f'  X: ({self.X.shape})')

    def update_weights(self, eta, stdout=False) -> None:
        logging.info('=============== InputLayer.update_weights ===============')

    
@dataclass
class HiddenLayer(Layer):
    n_nodes: int 

    W: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)

    prev_Layer: Layer = field(init=False, repr=False, default=lambda: object())
    next_Layer: Layer = field(init=False, repr=False, default=lambda: object())

    update: bool = None # True when layers is ready for update, else False

    a_l: np.ndarray = field(init=False, repr=False, default=None) # output from Activation function. Size (n nodes in layer l-) x (n_data)

    # Imparatn quantities for back back propagation
    derivative_activation: np.ndarray = field(init=False, repr=False) # Derivative of activation function with respect to z^l
    delta: np.ndarray = field(init=False, repr=False) # Derivate of Cost with respect to bias

    op: optimizer.Optimizer = field(init=False, default=None)

    def init_bias_and_weigths(self):
        # TODO: check if prev layer is Initialized
        self.b = np.zeros(self.n_nodes) + 0.01
        self.W = np.random.randn(self.prev_Layer.n_nodes, self.n_nodes) * 2

    def get_output(self):
        if np.size(self.a_l) == 1: 
            raise ValueError('Call method feed_forward to initlize variable')

        return self.a_l

    def feed_forward(self, activation_function: str) -> None:
        logging.info('=============== HiddenLayer.feed_forward ===============')
        output_prev_layer = self.prev_Layer.get_output()
        #breakpoint()
        z_l = output_prev_layer @ self.W + self.b # Feed forward

        # Choose activation func
        sigma = activation.Activation(z_l, activation_function)
        self.a_l, self.derivative_activation = sigma.get_values()

        logging.info(f'z_l: ({z_l.shape}) = a_(l-1): ({output_prev_layer.shape}) @ W: ({self.W.shape}) + b: ({self.b.shape})')
        logging.info(f'W = {self.W}')
        logging.info(f'b = {self.b}')

    def update_weights(self, initial_optimizer_ob: optimizer.Optimizer) -> None:
        logging.info('=============== HiddenLayer.update_weights ===============')
        if self.update != True:
            raise ValueError('Run method feed_forward, before atmepting to update weigts')
        pass

        if self.op == None: 
            assert(isinstance(initial_optimizer_ob, optimizer.Optimizer))
            self.op = copy.deepcopy(initial_optimizer_ob)

        delta = self.next_Layer.delta @ np.transpose(self.next_Layer.W) * self.derivative_activation
        gradient_weights = np.transpose(self.prev_Layer.get_output()) @ delta
        gradient_bias = np.sum(delta, axis = 0)
        # Use optimizer class to find new change
        W_change, b_change = self.op.update_change(gradient_weights, gradient_bias)

        W_new = self.W - W_change
        b_new = self.b - b_change

        self.W = W_new
        self.b = b_new
        self.delta = delta

        self.update = False
        self.prev_Layer.update = True # Previous layer (l-1) is ready for update 

        logging.info(f'delta: ({delta.shape}) = next_Layer.delta: ({self.next_Layer.delta.shape}) @ transpose(next_Layer.W): ({np.transpose(self.next_Layer.W).shape}) * derivative_activation ({self.derivative_activation.shape})')
        logging.info(f'W_new: ({W_new.shape}) = W_old: ({self.W.shape}) - eta * gradient_weights: ({gradient_weights.shape})')
        logging.info(f'b_new: ({b_new.shape}) = b_old: ({self.b.shape}) - eta * gradient_weights: ({gradient_bias.shape})')


@dataclass
class OutputLayer(Layer):
    n_nodes: int # n_nodes = n_targets
    score: str

    # t: np.ndarray = field(init=False, repr=False) # Targets

    W: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)

    prev_Layer: Layer = field(init=False, repr=False, default=lambda: object())
    next_Layer = None

    update: bool = None # True when layers is ready for update, else False

    a_l: np.ndarray = field(repr=False, default=None) # output from Activation function. Size (n nodes in layer l-) x (n_data)
                           # Initialize varablie with method: feed_forward
    # def __post_init__(self):
    op: optimizer.Optimizer = field(init=False, default=None)

    def init_bias_and_weigths(self): 
        self.b = np.zeros(self.n_nodes) + 0.01
        self.W = np.random.randn(self.prev_Layer.n_nodes, self.n_nodes)

    # def set_optimizer(self, op: optimizer.Optimizer): 
    #     if self.op != None: 
    #         raise AttributeError('Optimizer object is already initialized')

    #     self.op = op 

    def get_output(self):
        if np.size(self.a_l) == 1: 
            raise ValueError('Call method feed_forward to initlize variable')

        return self.a_l

    def feed_forward(self, activation_output: str, ignore: bool = False) -> None:
        logging.info('=============== OutputLayer.feed_forward ===============')
        if ignore == False and self.update == True:  
            raise ValueError('Network is ready for update. Run method update_weights before next iteration')

        output_prev_layer = self.prev_Layer.get_output()
        z_l = output_prev_layer @ self.W + self.b # Feed forward
        
        # Choose activation function
        ac = activation.Activation(z_l, activation_output)
        self.a_l, self.derivative_activation = ac.get_values()

        logging.info(f'z_l: ({z_l.shape}) = a_(l-1): ({output_prev_layer.shape}) @ W: ({self.W.shape}) + b: ({self.b.shape})')
        logging.info(f'W = {self.W}')
        logging.info(f'b = {self.b}')

        if ignore == True: 
            return

        self.update = True

    def update_weights(self, initial_optimizer_ob: optimizer.Optimizer, targets: np.ndarray) -> None:

        logging.info('=============== OutputLayer.update_weights ===============')
        # TODO: need access to update sheme
        if self.update != True:
            raise ValueError('Run method feed_forward, before atmepting to update weigts')

        if self.op == None: 
            assert(isinstance(initial_optimizer_ob, optimizer.Optimizer))
            self.op = copy.deepcopy(initial_optimizer_ob)

        sc = scores.Scores(self.a_l, targets, self.score)  # Errors are handled in Scores class
        grad_cost = sc.get_derivative()
        # grad_cost = (self.a_l-targets)/(self.a_l*(1-self.a_l)) # FIXME: testing


        delta = self.derivative_activation * grad_cost 
        gradient_weights = np.transpose(self.a_l) @ delta
        gradient_bias = np.sum(delta, axis = 0)
        # Calcaulate change in optimizer object

        # TODO: maybe just calc W_new and b_new in optimizer
        W_change, b_change = self.op.update_change(gradient_weights, gradient_bias) # add weigts


        # FIXME: if no optimizer, maybe? 
        # W_change = eta * gradient_weights
        # b_change = eta * gradient_bias

        W_new = self.W - W_change
        b_new = self.b - b_change

        self.W = W_new
        self.b = b_new
        self.delta = delta

        self.update = False
        self.prev_Layer.update = True # Previous layer is ready for update

        logging.info(f'delta: ({delta.shape}) = derivative_activation: ({self.derivative_activation.shape}) * grad_cost: ({grad_cost.shape})')
        logging.info(f'W_new: ({W_new.shape}) = W_old: ({self.W.shape}) - eta * gradient_weights: ({gradient_weights.shape})')
        logging.info(f'b_new: ({b_new.shape}) = b_old: ({self.b.shape}) - eta * gradient_weights: ({gradient_bias.shape})')


@dataclass
class NeuralNetwork:
    """ Class with the basic arcithecture of the Network """
    X_data: np.ndarray = field(repr=False)
    y_data: np.ndarray = field(repr=False)

    n_hidden_layers: int
    n_nodes_per_hidden_layer: int
    n_output_nodes: int

    cost_score: str

    activation_hidden: str  # TODO
    activation_output: str  # TODO

    n_features: int = field(init=False)

    Layers: Layer = field(default_factory=lambda: [])
    op: optimizer.Optimizer = field(init=False, default=None)

    def __post_init__(self):
        X_data, y_data = self.X_data, self.y_data

        if len(X_data.shape) == 1:
            X_data = X_data[:,np.newaxis]

        if len(y_data.shape) == 1:
            y_data = y_data[:,np.newaxis]

        self.X_data, self.y_data = X_data, y_data
        self.n_features = self.X_data.shape[1]
        self.__init_layers()


    def __init_layers(self): 
        # Init input layer

        # Get input data
        n_features = self.X_data.shape[1]
        self.Layers.append(InputLayer(n_features)) 

        # Init hidden layers
        for i in range(self.n_hidden_layers):
            self.Layers.append(HiddenLayer( self.n_nodes_per_hidden_layer))

        # Init ouput layer
        self.Layers.append(OutputLayer(self.n_output_nodes, self.cost_score)) # XXX: remove cost score

        # Connect layers
        for i, Layer in enumerate(self.Layers):
            if Layer.next_Layer != None:
                Layer.next_Layer = self.Layers[i+1]

            if Layer.prev_Layer != None:
                Layer.prev_Layer = self.Layers[i-1]

        # Initialize bias and weigts
        for i, layer in enumerate(self.Layers): 
            layer.init_bias_and_weigths()


    def get_layers(self, l: int = None):
        if l == None: 
            return self.Layers
        return self.Layers[l]

    def get_targets(self): 
        return self.y_data

    def set_optimizer(self, op: optimizer.Optimizer) -> None:
        self.op = op

    def get_optimizer(self) -> optimizer.Optimizer: 
        return self.op

    def get_X_data(self): 
        return self.X_data



@dataclass
class TrainNetwork: 
    """ Class used to train the network
    parameters:
        nn: NeuralNetwork object - Contains the arcithecture of the network
        op: Optimizer object - Contains shemes (gradient methods, etc.) for updateing weigts and biases 
    """
    nn: NeuralNetwork
    op: optimizer.Optimizer


    n_minibatches: int = None
    size_minibatch: int = None 

    X_full: np.ndarray = field(init=False, repr=False) # X full trianing data
    t_full: np.ndarray = field(init=False, repr=False) # Targets

    # data
    scores_minibatch: list = field(init=False, repr=False, default_factory=lambda: []) # Calculated when weights are updated in output layer
    test_scores: np.ndarray = field(init=False, repr=False)
    train_scores: np.ndarray =  field(init=False, repr=False)

    def __post_init__(self):
        logging.info('=============== INIT NN ===============')
        self.t_full = self.nn.get_targets()
        assert(self.nn.op == None)
        self.nn.set_optimizer(self.op)
        assert(self.nn.op != None)

        if self.n_minibatches != None and self.size_minibatch != None:
            raise ValueError('Set n_minibatches or minibatch_size, not both.')

        # TODO: 
        X = self.nn.get_X_data()
        assert(len(X.shape) != 1)
        assert(np.shape(X)[0] > np.shape(X)[1])

        # Set n_minibatches or size_minibatch
        if self.n_minibatches != None: 
            self.size_minibatch = int( len(X)/self.n_minibatches ) 

        elif self.size_minibatch != None: 
            self.n_minibatches = int( len(X)/self.size_minibatch )

        else:
            raise ValueError('Requires n_minibatches or size_minibatch')


        self.X_full = X







    def train(self, epochs: int, save_scores: bool = False, X_test: np.ndarray = None, y_test: np.ndarray = None): 
        """
        Parameters:
            save_scores: if True, save predicted scores for full dataset in each epoch.
                         This significantly slows down the training.
        """

        if save_scores == True: 
            if isinstance(X_test, type(None)) or isinstance(y_test, type(None)):
                raise ValueError('Test-data is reuqired in order to save scores')

            if self.nn.cost_score == 'cross_entropy':
                self.train_accuracies = np.zeros(epochs)
                self.test_accuracies = np.zeros(epochs)

            self.test_scores = np.zeros(epochs)
            self.train_scores = np.zeros(epochs)


        targets = self.nn.get_targets()  
        X = self.X_full
        n_minibatches = self.n_minibatches
        size_minibatch = self.size_minibatch
        # data_indices = np.arange(len(targets))

        tic = time.perf_counter()
        for epoch in range(epochs):
            for i in range(n_minibatches):

                # chosen_datapoints = np.random.choice(
                #         data_indices, size=n_minibatches, replace=False)
                # # minibatch training data
                # minibatch_X = X[chosen_datapoints]
                # minibatch_targets = targets[chosen_datapoints]
            
                k = np.random.randint(n_minibatches) # Pick random minibatch
                # k = i # XXX: remove
                slice_0 = k*size_minibatch
                slice_1 = (k+1)*size_minibatch 
                # XXX: Each batch is predifiend
                # XXX: Same minibatch may be selected twice
                minibatch_X = X[slice_0:slice_1]
                minibatch_targets = targets[slice_0:slice_1]

                self.__feed_forward(minibatch_X)
                self.__back_propagation(minibatch_targets)


            # Save scores
            if save_scores == True: 
                train_output = self.get_output(X)
                test_output = self.get_output(X_test)

                sc = scores.Scores(train_output, targets, self.nn.cost_score)
                self.train_scores[epoch] = sc.get_score()

                sc = scores.Scores(test_output, y_test, self.nn.cost_score)
                self.test_scores[epoch] = sc.get_score()

                self.train_scores[epoch] = self.get_score(X, targets)
                self.test_scores[epoch] = self.get_score(X_test, y_test)

            if save_scores == True and self.nn.cost_score == 'cross_entropy':
                # If cross_entropy use accuracy as classification score
                train_pred = np.where(train_output > 0.5, 1, 0)
                test_pred = np.where(test_output > 0.5, 1, 0)

                self.train_accuracies[epoch] = np.sum(train_pred == targets)/len(targets)
                self.test_accuracies[epoch] = np.sum(test_pred == y_test)/len(y_test)

            # if epoch != 0 and epochs % epoch == 0:
            if epoch % (epochs//10) == 0:
                print(f'{self.scores_minibatch[-1]}          Training: {epoch/epochs * 100}% ')


            if (time.perf_counter() - tic) > 0.03: 
                tic = time.perf_counter()
                print(self.scores_minibatch[-1])




    def __feed_forward(self, X, ignore=False):
        """
        Parameters:
            ignore: if True -> ignores update safety. 
                    Set to True if __feed_forward is called before or mid training. 
        """

        layers = self.nn.get_layers()

        # Feed forward input layer
        layers[0].feed_forward(X)

        # Fedd forward hidden layers
        l = 0  #  If no hidden layers
        if len(layers) != 2:  
            for l in range(1, len(layers)-1): 
                layers[l].feed_forward(self.nn.activation_hidden)

        # Fedd forward Ouput layer
        assert(isinstance(layers[l+1], OutputLayer))
        layers[l+1].feed_forward(self.nn.activation_output, ignore)

        # Retrun ouput from activation funciton in ouput layer
        output = layers[l+1].get_output()

        return output

    def __back_propagation(self, targets: np.ndarray):
        layers = self.nn.get_layers()
        op = self.op

        # Update weigts output layer
        assert(isinstance(layers[-1], OutputLayer))
        layers[-1].update_weights(op, targets)

        # Calculate score for minibatch
        output_minibatch = layers[-1].get_output()
        sc = scores.Scores( a_L=output_minibatch, t=targets, score_name=self.nn.cost_score) 
        score = sc.get_score()
        self.scores_minibatch.append(score)
        # print(score)

        for l in range(len(layers)-2, 0, -1): 
            layers[l].update_weights(op)

    def get_all_test_scores(self) -> np.ndarray: 
        try:
            return self.test_scores
        except AttributeError as e: 
            raise AttributeError('Run method train with parameter save_scores=True first.') from e

    def get_all_train_scores(self) -> np.ndarray: 
        try:
            return self.train_scores
        except AttributeError as e: 
            raise AttributeError('Run method train with parameter save_scores=True first.') from e

    def get_all_test_accuracyies(self) -> np.ndarray: 
        try:
            return self.test_accuracies
        except AttributeError as e: 
            raise AttributeError('Run method train with parameter save_scores=True first.' \
                                 'Variable is only created if nn.cost_score="cross_entropy"') from e

    def get_all_train_accuracyies(self): 
        pass
        try:
            return self.train_accuracies
        except AttributeError as e: 
            raise AttributeError('Run method train with parameter save_scores=True first.' \
                                 'Variable is only created if nn.cost_score="cross_entropy"') from e


    def get_output(self, X) -> np.ndarray: # XXX: will change layers, a_l value
        return self.__feed_forward(X, ignore=True)

    def get_score(self, X, t) -> float: # XXX: will change alyers, a_l value 
        sc = scores.Scores(self.get_output(X), t, self.nn.cost_score)
        return sc.get_score()

    def get_accuracy(self, X, t) -> float: 
        output = self.get_output(X)
        assert(t.shape[1] == 1)
        pred = np.where(output > 0.5, 1, 0)
        acc = np.sum(pred == t)/len(t)
        return acc
from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import warnings
import seaborn as sns
from importlib import reload

@dataclass
class Optimizer: 
    eta: float
    # gradient_method: str
    gamma: float = None
    # minibatch_size: int = None
    lambd: float = 0 
    beta: float = 0.9 # beta for rms_prop, typiclly 0.9
    beta1: float = 0.9 # beta first Momentum, typiclly 0.9
    beta2: float = 0.999 # beta secomd momentum, typiccly 0.99
    tuning_method: str = 'none'
    tuning_method_options: tuple = field(init=False, default = ('none', 'rms_prop', 'adam'))


    # gradient_method_options: list = field(init=False, default_factory=lambda: ['gd', 'sgd'])


    # Parameters used for update
    W_change: np.ndarray = field(init=False, default=None) # Previous change in weights
    b_change: np.ndarray = field(init=False, default=None) # Previous change in bias

    # rms prop Parameters
    W_first_momentum: np.ndarray = field(init=False, default=None) # Previous change in weights
    b_first_momentum: np.ndarray = field(init=False, default=None) # Previous change in bias
    W_second_momentum: np.ndarray = field(init=False, default=None) # Previous change in weights
    b_second_momentum: np.ndarray = field(init=False, default=None) # Previous change in bias

    # Adagrad Parameters
    G_iter: np.ndarray = field(init=False, default=None) #


    epsilon: float = field(init=False, default=None) # Previous change in bias


    number_of_updates: int = field(init=False, default=0)

    def __post_init__(self): 

        if self.gamma != None:
            if not 0 <= self.gamma <= 1:
                raise ValueError('Allowed range for gamma: [0, 1]')
            self.W_change = 0.0
            self.b_change = 0.0

        assert(self.tuning_method in self.tuning_method_options)

            # Inital values
        if self.tuning_method == 'none': 
            pass

        elif self.tuning_method == 'rms_prop': 
            self.__test_beta_rms()
            self.epsilon = 1e-8
            self.W_second_momentum = 0.0
            self.b_second_momentum = 0.0

        elif self.tuning_method == 'adam':
            self.__test_beta_adam()
            self.epsilon = 1e-8
            self.W_first_momentum = 0.0
            self.b_first_momentum = 0.0
            self.W_second_momentum = 0.0
            self.b_second_momentum = 0.0


    def __test_beta_rms(self):
        if self.beta == None: 
            raise ValueError(f'Tuning method {self.tuning_method} requires a value for beta, not None')

        elif not 0 <= self.beta < 1: 
            raise ValueError('Allowed range for beta: (0, 1]')

    def __test_beta_adam(self): 
        if self.beta1 == None or self.beta2 == None: 
            raise ValueError(f'Tuning method {self.tuning_method} requires a value for beta.' \
                    'Got beta1 = {self.beta1} and beta2 = {sefl.beta2}')

        elif not 0 <= self.beta1 < 1: 
            raise ValueError('Allowed range for beta1: [0, 1)')

        elif not 0 <= self.beta2 < 1: 
            raise ValueError('Allowed range for beta2: [0, 1)')

        elif self.beta1 > self.beta2: 
            warnings.warn("Warning: beta2 should be larger than beta1")
            input("Press Enter to continue...")


    def update_change(self, gradient_weights, gradient_bias):
        self.number_of_updates += 1

        if self.tuning_method == 'none':
            # Gradient descent
            W_change = self.eta * gradient_weights
            b_change = self.eta * gradient_bias

        elif self.tuning_method == 'rms_prop':
            self.W_second_momentum = self.beta * self.W_second_momentum + (1-self.beta) * gradient_weights**2 
            self.b_second_momentum = self.beta * self.b_second_momentum + (1-self.beta) * gradient_bias**2
            W_change = self.eta * gradient_weights / np.sqrt(self.W_second_momentum + self.epsilon)
            b_change = self.eta * gradient_bias / np.sqrt(self.b_second_momentum + self.epsilon)

        elif self.tuning_method == 'adam': 
            self.W_first_momentum = self.beta1 * self.W_first_momentum + (1-self.beta1) * gradient_weights
            self.W_second_momentum = self.beta2 * self.W_second_momentum + (1-self.beta2) * gradient_weights**2
            self.b_first_momentum = self.beta1 * self.b_first_momentum + (1-self.beta1) * gradient_bias
            self.b_second_momentum = self.beta2 * self.b_second_momentum + (1-self.beta2) * gradient_bias**2

            # Bias correct first momentum 
            W_first_momentum = self.W_first_momentum/(1-self.beta1**self.number_of_updates) # XXX: correction is not saved
            b_first_momentum = self.b_first_momentum/(1-self.beta1**self.number_of_updates)

            # Bias corrected second momentum
            W_second_momentum = self.W_second_momentum/(1-self.beta2**self.number_of_updates) # XXX: correction is not saved
            b_second_momentum = self.b_second_momentum/(1-self.beta2**self.number_of_updates)

            W_change = self.eta * W_first_momentum/(np.sqrt(W_second_momentum) + self.epsilon)
            b_change = self.eta * b_first_momentum/(np.sqrt(b_second_momentum) + self.epsilon)



        if self.gamma != None: 
            # Momentum
            W_change += self.gamma*self.W_change
            b_change += self.gamma*self.b_change

        if self.lambd > 0:
            # L2 regularization
            W_change += self.lambd * gradient_weights # FIXME: Add to bias ? 
            b_change += self.lambd * gradient_bias

        self.W_change, self.b_change = W_change, b_change

        return W_change, b_change

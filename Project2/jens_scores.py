from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload
import autograd.numpy as jnp  # Thinly-wrapped numpy
from autograd import grad
#import jax.numpy as jnp 
#from jax import grad

# def MSE(y, t):
#     """ Squared error cost function""" # XXX: removed 1/2
#     # assert(len(y.shape) == 1)
#     return jnp.sum((t - y)**2)

@dataclass
class Scores:
    """
    """
    a_L: np.ndarray = field(repr=False) # Ouput from output layer
    t: np.ndarray = field(repr=False) # Targets
    score_name: str
    options: list = field(init=False, default_factory=lambda: ['mse', 'cross_entropy'])

    def __post_init__(self):
        assert(self.score_name in self.options)
        assert(self.a_L.shape == self.t.shape)
        assert(self.a_L.shape[1] == 1)

    def __sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def MSE(self, y, t):
        """ Squared error cost function""" # XXX: removed or include 1/n
        return 1/2 * np.sum((t - y)**2) / len(t)

    def __cross_entropy(self, y, t): # XXX: removed or include 1/n
        # Prob belonging to class 0, defined as sigmoid
        # N = len(t)
        # Ln = -t * np.log(y) - (1 - t) * np.log(1 - y)
        # return np.sum(Ln)

        diff = 1-y
        diff[diff == 0] = 1e-12
        Ln = t * np.log(y) + (1 - t) * np.log(diff)

        Ln = np.nan_to_num(Ln)

        if np.isnan(np.min(Ln)) == True: 
            breakpoint()  

        return -1*np.sum(Ln)/len(t)

    def __derivative_cross_entropy(self, y, t):
        # return -t/y + (1 - t) * 1/(1 - y)
        
        numerator = (y*(1-y))
        numerator[numerator == 0] = 1e-12
        derivative = (y-t)/numerator

        # ans = np.where(abs(derivative)<1e-12)
        # derivative = np.nan_to_num(derivative, nan=1e-12)

        if np.isnan(np.min(derivative)) == True: 
            breakpoint()  

        # if len(ans[0]) > 0: 
        #     breakpoint() 
        return derivative / len(t)


    def get_score(self): 
        if self.score_name == 'mse': 
            return float(self.MSE(self.a_L, self.t))
        elif self.score_name == 'cross_entropy': 
            return self.__cross_entropy(self.a_L, self.t)
        else:
            raise ValueError('Allowed opitons for score_name is: {self.options}')

    def get_derivative(self): 
        if self.score_name == 'mse': 
            return (self.a_L - self.t)/len(self.t)
        elif self.score_name == 'cross_entropy': 
            return self.__derivative_cross_entropy(self.a_L, self.t)
        else:
            raise ValueError('Allowed opitons for score_name is: {self.options}')
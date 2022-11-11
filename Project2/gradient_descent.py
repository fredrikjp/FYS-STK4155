from dataclasses import dataclass, field
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import poly_data
import autograd.numpy as np  # Thinly-wrapped numpy
# import jax.numpy as jnp 
# from jax import grad
# import jax.numpy as np 
from autograd import grad 
import pprint
import time


def log_time(func):
    def wrapper(*args, **kwargs):
        tic = time.perf_counter()
        result = func(*args, **kwargs)
        toc = time.perf_counter()
        print(f'Function {func.__name__!r} executed in {(toc-tic):.4f}s')
        return result

    return wrapper


@dataclass()
class GradientDescent: 
    data_object: object  

    dataset:str = 'train'

    X_data: np.ndarray = field(init=False, repr=False) # X data with resepct to dataset variable
    y_data: np.ndarray = field(init=False, repr=False) # y --||--

    thetas: np.ndarray = field(init=False, repr=False) # Array with thetas for all iterations
    thetas_init: np.ndarray = field(init=False) # Intial condition for thata

    def __post_init__(self):
        get_func_name = f'get_{self.dataset}'
        get_data = getattr(self.data_object, get_func_name)
        self.X_data, self.y_data= get_data()
        self.set_initial_conditions()
    
    def costOLS(self, X, y, theta):
        assert(len(y) != 1)
        y_pred = X @ theta
        return np.sum((y[:,np.newaxis] - y_pred)**2) #/len(y) # XXX

    def costRidge(self, X, y, theta, lamb): 
        assert(len(y) != 1)
        y_pred = X @ theta
        return np.sum((y[:,np.newaxis] - y_pred)**2) + lamb * np.sum(y_pred**2) #/len(y) # XXX


    def grad_costOLS(self): 
        return 

    def get_hessian(self): 
        pass

    def set_initial_conditions(self):
        n_coeff = len(self.data_object.coeff) # Number of polynomail coefficents inlcuding 0
        self.thetas_init = np.random.randn(n_coeff, 1) # Inital guess for theta's 
        print(self.thetas_init)


    def delta_gd(self, X, y, theta, eta, momentum = None): 
        grad_func = grad(self.costOLS, 2)
        gradients = grad_func(X, y, theta)

        if momentum == None:
            return -eta * gradients


    def update_adagrad(self, eta, gradients, i, delta=1e-8):
        # If new epoch
        if i == 1:
            n = gradients.shape[0]
            self.Giter = np.zeros((n,n))
        # Accumulated gradients outer product
        self.Giter += gradients @ gradients.T
        
        # Adding delta to avoid the possibility of dividing by zero
        Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(self.Giter)))]
        update = np.multiply(Ginverse, gradients)
        return update

    def update_RMSprop(self, eta, gradients, i, rho = 0.99, delta=1e-8):
        # If new epoch
        if i == 1:
            n = gradients.shape[0]
            self.Giter = np.zeros((n,n))
        # Previous value for the outer product of gradients
        Previous = self.Giter
	    # Accumulated gradient
        self.Giter += gradients @ gradients.T
	    # Scaling with rho the new and the previous results
        Gnew = (rho*Previous+(1-rho)*self.Giter)
	    # Taking the diagonal only and inverting
        Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(Gnew)))]
	    # Hadamard product
        update = np.multiply(Ginverse,gradients)
        return update

    def update_ADAM(self, eta, gradients, i, beta1 = 0.9, beta2 = 0.99, delta=1e-8):
        
        if i == 1:
            self.theta_first_momentum = 0
            self.theta_second_momentum = 0

        # First and second momentum
        self.theta_first_momentum = beta1 * self.theta_first_momentum + (1-beta1) * gradients
        self.theta_second_momentum = beta2 * self.theta_second_momentum + (1-beta2) * gradients**2

        # Bias correction
        theta_first_momentum = self.theta_first_momentum/(1-beta1**i)
        theta_second_momentum = self.theta_second_momentum/(1-beta2**i)
        update = eta * theta_first_momentum/(np.sqrt(theta_second_momentum) + delta)
        return update

    def gd(self, eta: float, n_epochs: int = 100, gamma: float = None, lamb: float = 0,  tuning_method = None):
        """ 
        eta = learning rate
        gamma = momentum 
        lamb: l2 regularization parameter if 0 -> mse
        """

        if gamma != None: 
            if not 0 <= gamma <= 1:
                raise ValueError('Allowed range for gamma: [0, 1]')
            change = 0.0


        # Initial values 
        n_coeff = len(self.data_object.coeff) # Number of polynomail coefficents inlcuding 0
        theta_new = self.thetas_init

        thetas = np.zeros((n_epochs+1, n_coeff))
        thetas[0,:] = theta_new.T

        for i in range(1, n_epochs+1): 
            # Change to while wiht tolearnace
            theta_old = theta_new

            # Use OLS or L2 regularization as cost funciton
            if lamb==0: 
                grad_func = grad(self.costOLS, 2)
                gradients = grad_func(self.X_data, self.y_data, theta_new)
            else:
                grad_func = grad(self.costRidge, 2)
                gradients = grad_func(self.X_data, self.y_data, theta_new, lamb)

            if tuning_method == "AdaGrad":
                grad_step = self.update_adagrad(eta, gradients, i)
            
            elif tuning_method == "RMSprop":
                grad_step = self.update_RMSprop(eta, gradients, i)
            
            elif tuning_method == "ADAM":
                grad_step = self.update_ADAM(eta, gradients, i)
           
            else:
                grad_step = eta*gradients
            

            if gamma!=None: 
                # Use gradient descent with momentum
                new_change = grad_step + gamma*change
            else:
                # Use gradient descent to update
                new_change = grad_step


            theta_new = theta_old - new_change
            thetas[i,:] = theta_new.T

            change = new_change

        self.thetas = thetas


    def sgd(self, eta: float, n_epochs: int, size_batch: int, gamma: float = None, lamb: float = 0, tuning_method = None):
        """
        lamb: l2 regularization parameter if 0 -> mse
        """

        if gamma != None:
            if not 0 <= gamma <= 1:
                raise ValueError('Allowed range for gamma: [0, 1]')
            change = 0.0


        # Initial values 
        n_coeff = len(self.data_object.coeff) # Number of polynomail coefficents inlcuding 0
        theta_new = self.thetas_init # Intial guess for thetas


        n_data = len(self.y_data)
        n_minibatches = int(n_data/size_batch)

        thetas = np.zeros((n_epochs*n_minibatches+1, n_coeff))
        thetas[0,:] = theta_new.T
        j = 1


        for epoch in range(1, n_epochs+1): 
            for i in range(1, n_minibatches+1): 
                """
                k = np.random.randint(n_minibatches) # Pick random minibatch
                k = i # XXX: remove
                slice_0 = k*size_batch
                slice_1 = (k+1)*size_batch 
                # XXX: Each batch is predifiend
                # XXX: Same minibatch may be selected twice
                minibatch_X = self.X_data[slice_0:slice_1]
                minibatch_y = self.y_data[slice_0:slice_1]
                """
                ##########
                # Random minibatch sample
                chosen_datapoints = np.random.choice(
                    n_data, size=size_batch, replace=False
                )
                minibatch_X = self.X_data[chosen_datapoints]
                minibatch_y = self.y_data[chosen_datapoints]
                ###########


                # TODO: Change to while wiht tolearnace
                theta_old = theta_new

                # Claculate gradient

                # Use OLS or L2 regularization as cost funciton
                if lamb == 0: 
                    grad_func = grad(self.costOLS, 2)
                    gradients = grad_func(minibatch_X, minibatch_y, theta_new)
                else:
                    grad_func = grad(self.costRidge, 2)
                    gradients = grad_func(minibatch_X, minibatch_y, theta_new, lamb)

                if tuning_method == "AdaGrad":
                    grad_step = self.update_adagrad(eta, gradients, i=i)
                
                elif tuning_method == "RMSprop":
                    grad_step = self.update_RMSprop(eta, gradients, i=i)
            
                elif tuning_method == "ADAM":
                    grad_step = self.update_ADAM(eta, gradients, i=i)
           
                else:
                    grad_step = eta*gradients
                
                if gamma != None:
                    # sgd with momentum
                    new_change = grad_step + gamma*change
                else: 
                    # plain sgd 
                    new_change = grad_step

                theta_new = theta_old - new_change
                thetas[j,:] = theta_new.T

                change = new_change
                j+=1

        self.thetas = thetas

        
    def get_theta(self): 
        return self.thetas[-1, :]

    def get_thetas(self):
        return self.thetas














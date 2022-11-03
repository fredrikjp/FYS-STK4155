from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt



@dataclass(frozen=True)
class PolyData: 
    n_data: int

    x_range: tuple = (-1,1)
    coeff: tuple = (1, 0, 1)

    test_size: float = 0.2

    seed = 1
    set_seed = True

    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)

    X_train: np.ndarray = field(init=False, repr=False)
    X_test: np.ndarray = field(init=False, repr=False)
    y_train: np.ndarray = field(init=False, repr=False)
    y_test: np.ndarray = field(init=False, repr=False)

    def __post_init__(self): 
        self.__create_datapoints()
        self.__create_design_matrix()
        self.__create_polynomial()
        self.__split_data()

    def get_X_trian(self): 
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

    def get_train(self):
        return self.X_train, self.y_train

    def get_test(self):
        return self.X_test, self.y_test

    def __create_datapoints(self): 
        x = np.linspace(self.x_range[0], self.x_range[1], self.n_data)
        object.__setattr__(self, 'x', x)

    def __create_polynomial(self): 
        y = self.X @ self.coeff
        object.__setattr__(self, 'y', y)
    
    def __create_design_matrix(self): 
        X = np.zeros((self.n_data, len(self.coeff)))  # Desing matrix
        for i in range(len(self.coeff)): 
            X[:,i] = self.x**i

        object.__setattr__(self, 'X', X)

    def __split_data(self): 
        if self.set_seed:
            X_train, X_test, y_train, y_test = \
                    train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.seed)
        else:
            X_train, X_test, y_train, y_test = \
                    train_test_split(self.X, self.y, test_size=self.test_size)

        object.__setattr__(self, 'X_train', X_train)
        object.__setattr__(self, 'X_test', X_test)
        object.__setattr__(self, 'y_train', y_train)
        object.__setattr__(self, 'y_test', y_test)




    def plot(self):
        plt.figure()
        sns.set_style("darkgrid")
        sns.lineplot(x=self.x, y=self.y)
        plt.axis('equal')
        plt.show()
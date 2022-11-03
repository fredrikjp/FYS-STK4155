import sys
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload

def mse(y_test, y_pred):
    return np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))

@dataclass
class Plot:
    data_object: object


    def plot_mse_vs_theta(self, thetas_dict: dict, dataset: str ='test'): 
        get_func_name = f'get_{dataset}'
        get_data = getattr(self.data_object, get_func_name)
        X, y = get_data()
        y = y.reshape(-1, 1)


        n_features = list(thetas_dict.values())[0].shape[1]
        
        for i in range(n_features): 
            plt.figure()

            for method, thetas in thetas_dict.items():
                mse_scores = np.zeros(len(thetas))

                for j in range(np.shape(thetas)[0]):
                    theta = thetas[j,:].reshape(-1, 1)
                    mse_scores[j] = mse(y, X@theta)

                sns.set_style("darkgrid")
                plt.xlabel(f'$\\theta_{i}$')
                plt.ylabel('MSE')
                sns.scatterplot(x=thetas[:,i], y=mse_scores, label=method)
        plt.show()


    def plot_iter(self): 
        """ Plot prediction for iteration n
        """
        pass
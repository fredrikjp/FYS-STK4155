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
            fig, ax = plt.subplots()
            # plt.figure()

            for method, thetas in thetas_dict.items():
                mse_scores = np.zeros(len(thetas))

                for j in range(np.shape(thetas)[0]):
                    theta = thetas[j,:].reshape(-1, 1)
                    mse_scores[j] = mse(y, X@theta)

                sns.set_style("darkgrid")
                plt.xlabel(f'$\\theta_{i}$')
                plt.ylabel('MSE')
                sns.lineplot(x=thetas[:,i], y=mse_scores, marker='o', linewidth=0, label=method)

            self.toggle_legend(ax)

        plt.show()

    def plot_iter_MSE(self, thetas_dict: dict, dataset: str ='test'):
        
        get_func_name = f'get_{dataset}'
        get_data = getattr(self.data_object, get_func_name)
        X, y = get_data()
        y = y.reshape(-1, 1)

        for method, thetas in thetas_dict.items():
            n = len(thetas)
            mse_scores = np.zeros(n)
            iter = np.linspace(0,n,n)

            for j in range(np.shape(thetas)[0]):
                theta = thetas[j,:].reshape(-1, 1)
                mse_scores[j] = mse(y, X@theta)

            sns.set_style("darkgrid")
            plt.xlabel('Iteration')
            plt.ylabel('ln(MSE)')
            sns.lineplot(x=iter, y=np.log(mse_scores), linewidth=1, label=method)

        plt.legend()
        


    def plot_iter(self): 
        """ Plot prediction for iteration n
        """
        pass


    def toggle_legend(self, ax):
        leg = ax.legend(fancybox=True, shadow=True)

        lined = {}  # Will map legend lines to original lines.
        for legline, origline in zip(leg.get_lines(), ax.get_lines()):
            legline.set_picker(True)  # Enable picking on the legend line.
            lined[legline] = origline


        def on_pick(event):
            # On the pick event, find the original line corresponding to the legend
            # proxy line, and toggle its visibility.
            legline = event.artist
            origline = lined[legline]
            visible = not origline.get_visible()
            origline.set_visible(visible)
            # Change the alpha on the line in the legend so we can see what lines
            # have been toggled.
            legline.set_alpha(1.0 if visible else 0.2)
            ax.get_figure().canvas.draw()

        ax.get_figure().canvas.mpl_connect('pick_event', on_pick)

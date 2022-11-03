from dataclasses import dataclass, field
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from importlib import reload
import poly_data
import gradient_descent
import plot




if __name__ == '__main__':
    reload(poly_data)
    reload(gradient_descent)
    reload(plot)

    p = poly_data.PolyData(n_data = 100)
    
    gd = gradient_descent.GradientDescent(p)

    gd.plane_gd(0.001)
    theta_gd = gd.get_thetas()
    gd.gd_momentum(0.001, 0.1)
    theta_gd_momentum = gd.get_thetas()

    gd.sgd( 0.001, 20, 50)
    theta_sgd = gd.get_thetas()

    gd.sgd_momentum(0.001, 0.1, 20, 50)
    theta_sgd_momentum = gd.get_thetas()

    # Plotting
    theta_dict = {'theta_gd': theta_gd, 'theta_gd_momentum': theta_gd_momentum, 'sgd_theta': theta_sgd, 'theta_sgd_momentum': theta_sgd_momentum}
    pl = plot.Plot(p)
    pl.plot_mse_vs_theta(theta_dict)

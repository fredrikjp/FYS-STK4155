from random import random, seed
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def plot_3D(x, y, z, title="", filename=""):
    if len(x.shape)>1:
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title(title)
        if filename!="":
            plt.savefig(filename)
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x, y, z, ".")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.title(title)
        if filename!="":
            plt.savefig(filename)
        plt.show()
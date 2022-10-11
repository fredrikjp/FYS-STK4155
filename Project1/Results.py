import numpy as np
from sklearn.model_selection import train_test_split
from FrankeFunction import FrankeFunction
from plot_3D import plot_3D
from OLS import R2, designmatrix, plot_poldeg_analysis, OLS
from Bias_Var import bias_var
from k_fold import k_fold
from Ridge import Ridge
from Lasso import Lasso

from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
work_path = os.path.dirname(__file__) 
fig_path = os.path.join(work_path, "fig/") #path to directory storing figures


np.random.seed(1)

x = np.sort(np.random.uniform(0,1,100))
y = np.sort(np.random.uniform(0,1,100))

f = np.ravel(FrankeFunction(x,y,1))
pol_deg = 5
X = designmatrix(pol_deg,x,y)
X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2)

plot_poldeg_analysis(X_train, X_test, y_train, y_test, pol_deg, OLS, filename="OLS")

pol_deg_lim = 20
X = designmatrix(pol_deg_lim, x, y)
X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2)
plot_poldeg_analysis(X_train, X_test, y_train, y_test, pol_deg_lim, OLS, beta_bool=False, R2_bool=False, filename="OLS_20")


for i in range(3, 6):
    x = np.sort(np.random.uniform(0,1,int(10*i)))
    y = np.sort(np.random.uniform(0,1,int(10*i)))
    f = np.ravel(FrankeFunction(x,y,0.2))
    pol_deg_lim = 15
    X = designmatrix(pol_deg_lim, x, y)
    X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2)
    bias_var(X_train, X_test, y_train, y_test, OLS, pol_deg_lim, 50, filename=f"OLS_{(10*i)**2}points_Bootstrap")
    k_fold(X, f, 10, OLS, pol_deg_lim, filename=f"OLS_{(10*i)**2}points_Kfold")


x = np.sort(np.random.uniform(0,1,100))
y = np.sort(np.random.uniform(0,1,100))
f = np.ravel(FrankeFunction(x,y,1))

pol_deg_lim = 20
X = designmatrix(pol_deg_lim, x, y)
X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2)
plot_poldeg_analysis(X_train, X_test, y_train, y_test, pol_deg_lim, Ridge, beta_bool=False, R2_bool=False, filename="Ridge_20")
plot_poldeg_analysis(X_train, X_test, y_train, y_test, pol_deg_lim, Lasso, beta_bool=False, R2_bool=False, filename="Lasso_20")




x = np.sort(np.random.uniform(0,1,20))
y = np.sort(np.random.uniform(0,1,20))
f = np.ravel(FrankeFunction(x,y,0.2))
pol_deg_lim = 15
X = designmatrix(pol_deg_lim, x, y)
X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2)


for i in range(3):
    plot_poldeg_analysis(X_train, X_test, y_train, y_test, 5, Ridge, MSE_bool=False, R2_bool=False, lmb_index=int(2*i), filename=f"Ridge_lmb{2*i}")
    bias_var(X_train, X_test, y_train, y_test, Ridge, pol_deg_lim, 50, int(2*i), filename=f"Ridge_Bootstrap_lmb_i{2*i}")
    k_fold(X, f, 10, Ridge, pol_deg_lim, int(2*i), filename=f"Ridge_Kfold_lmb_i{2*i}")
    plot_poldeg_analysis(X_train, X_test, y_train, y_test, 5, Lasso, MSE_bool=False, R2_bool=False, lmb_index=int(2*i), filename=f"Lasso_lmb{2*i}")
    bias_var(X_train, X_test, y_train, y_test, Lasso, pol_deg_lim, 50, int(2*i), filename=f"Lasso_Bootstrap_lmb_i{2*i}")
    k_fold(X, f, 10, Lasso, pol_deg_lim, int(2*i), filename=f"Lasso_Kfold_lmb_i{2*i}")


# Load the terrain
terrain1 = imread("SRTM_data_Norway_1.tif")

# Show the terrain
plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain1, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar()
plt.show()


terrain = terrain1[210:240, 1280:1310]
plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar()
plt.savefig(fig_path+"terrain.pdf")
plt.show()



x_lim = terrain.shape[1]
y_lim = terrain.shape[0]

x = np.linspace(0, 1, x_lim)
y = np.linspace(0, 1, y_lim)
terrain = terrain.ravel()

np.random.seed(2)

pol_deg = 5
X = designmatrix(pol_deg,x,y)
X_train, X_test, y_train, y_test = train_test_split(X, terrain, test_size=0.2)


plot_poldeg_analysis(X_train, X_test, y_train, y_test, pol_deg, Ridge, beta_bool=False, R2_bool=False, filename="Ridge_terrain")
plot_poldeg_analysis(X_train, X_test, y_train, y_test, pol_deg, Lasso, beta_bool=False, R2_bool=False, filename="Lasso_Terrain")


k_fold(X, terrain, 10, OLS, pol_deg, mindegree=2, show=False)
k_fold(X, terrain, 10, Ridge, pol_deg, mindegree=2, lmb_index=0, show=False)
k_fold(X, terrain, 10, Lasso, pol_deg, mindegree=2, lmb_index=0, show=False)
plt.title("Cross-valdation MSE terrain data")
plt.legend(["OLS", "Ridge "+r"$\lambda=1.0^{-5}$", "Lasso "+r"$\lambda=1.0^{-5}$"])
plt.savefig(fig_path+"Kfold_terrain.pdf")
plt.show()
#X_test is irrelevant here, since we will only use the training prediction
OLS_pred = OLS(X, X_test, terrain)[0] 
OLS_pred = OLS_pred.reshape(y_lim, x_lim)

plt.figure()
plt.title("Terrain over Norway 1 OLS prediction")
plt.imshow(OLS_pred, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar()
plt.savefig(fig_path+"OLS_terrain.pdf")
plt.show()

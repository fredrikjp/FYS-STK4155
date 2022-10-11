
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from FrankeFunction import FrankeFunction
from plot_3D import plot_3D
import seaborn as sns
import os
work_path = os.path.dirname(__file__) 
fig_path = os.path.join(work_path, "fig/") #path to directory storing figures

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2, axis=0) / np.sum((y_data - np.mean(y_data, axis=0)) ** 2, axis=0)
def MSE(y_data,y_model):
    n = np.size(y_data)
    return np.sum((y_data-y_model)**2, axis=0)/n


def designmatrix(n, x, y):

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)
    else:       #If datapoints have not already been meshed
        x, y = np.meshgrid(x,y)
        x, y = np.ravel(x), np.ravel(y)
    N = len(x)
    l = int((n+1)*(n+2)/2)		# Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X
          
def OLS(X_train, X_test, y_train, plot=False): 

    x0, y0 = X_train[:,1], X_train[:,2]
    x1, y1 = X_test[:,1], X_test[:,2]

    # matrix inversion to find beta
    OLSbeta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train

    # and then make the prediction
    ytilde_train = X_train @ OLSbeta
    ytilde_test = X_test @ OLSbeta
    if plot:
        plot_3D(x0, y0, ytilde_train, f"OLS train")
        plot_3D(x1, y1, ytilde_test, f"OLS test")

    return  ytilde_train, ytilde_test, OLSbeta



def plot_poldeg_analysis(X_train, X_test, y_train, y_test, pol_deg_lim, reg_method, beta_bool=True, MSE_bool=True, R2_bool=True, lmb_index=0, filename=""):
    pol_deg = []
    
    R2_train = []
    MSE_train = []
    
    R2_test = []
    MSE_test = []

    reg_name = filename.split("_")[0]
    for i in range (1, pol_deg_lim + 1):
        l = int((i+1)*(i+2)/2)		# Number of elements in beta
        X_tr = X_train[:,:l]
        X_te = X_test[:,:l]

        pol_deg.append(i)
        reg = reg_method(X_tr, X_te, y_train.ravel())
        ytilde_train, ytilde_test, beta = reg[0], reg[1], reg[2]
        multi_dim = len(ytilde_test.shape) > 1
        if multi_dim:
            y_train = y_train.reshape(-1,1)
            y_test = y_test.reshape(-1,1)
            lambdas = reg[3]
        beta_index = np.linspace(0, len(beta) - 1, len(beta))

        R2_train.append(R2(y_train, ytilde_train))
        MSE_train.append(MSE(y_train, ytilde_train))
        
        R2_test.append(R2(y_test, ytilde_test))
        MSE_test.append(MSE(y_test, ytilde_test))
        if beta_bool:
            if multi_dim:
                plt.plot(beta_index, beta[:,lmb_index], "*-")
            else:
                plt.plot(beta_index, beta, "*--")
    if beta_bool:
        plt.ylabel(r"$\beta$ value")
        plt.xlabel(r"$\beta$ number")
        plt.legend([f"P{deg}" for deg in pol_deg])
        if multi_dim:
            plt.title(reg_name+r" $\lambda$="+f"{lambdas[lmb_index]:.3g}")
        else:
            plt.title(reg_name)
        if filename!="":
            plt.savefig(fig_path+"beta"+filename)
        plt.show()

    if MSE_bool:
        if multi_dim:
            ax = sns.heatmap(MSE_train, annot=True, fmt=".5g", xticklabels=[f"{lmb:.1e}" for lmb in lambdas], yticklabels=[str(deg) for deg in pol_deg], cbar_kws={"label": "MSE"})
            ax.set(xlabel=r"$\lambda$", ylabel="Polynomial degree", title=reg_name+" MSE train")
            plt.tight_layout()
            if filename!="":
                plt.savefig(fig_path+"MSE_train"+filename)
            plt.show()
            ax = sns.heatmap(MSE_test, annot=True, fmt=".5g", xticklabels=[f"{lmb:.1e}" for lmb in lambdas], yticklabels=[str(deg) for deg in pol_deg], cbar_kws={"label": "MSE"})
            ax.set(xlabel=r"$\lambda$", ylabel="Polynomial degree", title=reg_name+" MSE test")
            plt.tight_layout()
            if filename!="":
                plt.savefig(fig_path+"MSE_test"+filename)
            plt.show()
        else:  
            plt.plot(pol_deg, MSE_train, "*-")
            plt.plot(pol_deg, MSE_test, "*-")
            plt.xlabel("Polynomial degree")
            plt.ylabel("MSE")
            plt.legend(["MSE train", "MSE test"])
            plt.title(reg_name)
            if filename!="":
                plt.savefig(fig_path+"MSE"+filename)
            plt.show()
    
    if R2_bool:
        if multi_dim:
            plt.plot(pol_deg, np.array(R2_train)[:, lmb_index], "*-")
            plt.plot(pol_deg, np.array(R2_test)[:, lmb_index], "*-")
            plt.legend(["R2 train", "R2 test"])
            plt.xlabel("Polynomial degree")
            plt.ylabel("R2")
            plt.title(reg_name+r" $\lambda$="+f"{lambdas[lmb_index]:.3g}")
            if filename!="":
                plt.savefig(fig_path+"R2"+filename)
            plt.show()
            
        else:
            plt.plot(pol_deg, R2_train, "*-")
            plt.plot(pol_deg, R2_test, "*-")
            plt.legend(["R2 train", "R2 test"])
            plt.xlabel("Polynomial degree")
            plt.ylabel("R2")
            plt.title(reg_name)
            if filename!="":
                plt.savefig(fig_path+"R2"+filename)
            plt.show()


if __name__=="__main__":
    # A seed just to ensure that the random numbers are the same for every run.
    # Useful for eventual debugging.
    np.random.seed(1)

    x = np.sort(np.random.uniform(0,1,100))
    y = np.sort(np.random.uniform(0,1,100))

    f = np.ravel(FrankeFunction(x,y,1))
    #OLS(x, y, 0, 0.1, plot=True)
    #plot_poldeg_analysis(x,y,5)

    pol_deg_lim = 20
    X = designmatrix(pol_deg_lim, x, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, f, test_size=0.2)
    #OLS(X_train, X_test, y_train, plot=True)

    #plot_poldeg_analysis(X_train, X_test, y_train, y_test, 5, OLS)
    plot_poldeg_analysis(X_train, X_test, y_train, y_test, pol_deg_lim, OLS, beta_bool=False, R2_bool=False)






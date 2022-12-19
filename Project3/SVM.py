import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import read_csv
from importlib import reload
from cvxopt import matrix
from cvxopt import solvers
import sklearn as skl 

class SVM:

    def __init__(self, X, y, gamma = 0.1, C=1,test_size = 0.25, seed = None, kernel = "GRBF"):
  
        self.gamma = gamma  
        self.C = C

        self.seed = seed
        # Scale data with mean and std
        scaler = skl.preprocessing.StandardScaler()
        scaler.fit(X)
        self.X = scaler.transform(X)
        self.y = y

        # Test train split
        self.X_train, self.X_test, self.y_train, self.y_test = \
                    train_test_split(self.X, self.y, test_size=test_size, random_state=self.seed)

        self.n_features = X.shape[1]
        
        self.b = 0

    def K(self, xi, xj):
        # GRBF kernel function
        return np.exp(-self.gamma * np.linalg.norm(xi - xj)**2)    
    
    def solve(self):
        # Solution for minimizing 
        # (1/2) * lambda.T @ P @ lambda + q.T @ lambda 
        # subject to constraints

        y = self.y_train
        X = self.X_train
        n = self.X_train.shape[0]

        q = -1 * np.ones(n).T
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                P[i,j] = y[i] * y[j] * self.K(X[i,:], X[j,:])
        
        # G and h sets constraints on col vec lambda: G@lambda <= h
        G = np.zeros((2*n, n))
        G [:n, :] = np.identity(n) * (- 1)
        G[n:,:] = np.identity(n)
        h = np.zeros(2*n)
        h[n:] = np.ones(n) * self.C
        # A and b sets  constraints on labda: A@lambda = b
        A = np.array(self.y_train, dtype=float).T
        b = np.zeros(1) 
    
        # Solve 
        P, q, G, h, A, b = matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b)
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        self.lmb = np.array(sol["x"])
        self.lmb_non_zero_indecies = np.where(self.lmb > 1e-5)[0]
        # Calculate bias
        self.calc_b()

    def calc_b(self):
        # Calculates the bias term neede for the predict method
        b = 0
        for j in self.lmb_non_zero_indecies:
            for i in range(self.X_train.shape[0]):
                b -= self.lmb[i] * self.y_train[i] * self.K(self.X_train[j], self.X_train[i])
            b += self.y_train[j]
        self.b = b / len(self.lmb_non_zero_indecies)
        
    def predict(self):
        # Uses model to predict the clasification of test data
        n_predictions = self.X_test.shape[0]
        predictions = np.zeros(n_predictions)
        for j in range(n_predictions):
            for i in self.lmb_non_zero_indecies:
                predictions[j] += self.y_train[i] * self.lmb[i] * self.K(self.X_test[j], self.X_train[i])
                
        predictions += self.b
        predictions = np.where(predictions >= 0, 1, -1)
        accuracy = np.sum(np.where(predictions == self.y_test.T, 1, 0)) / len(predictions)
        return predictions, accuracy

    def k_fold(self, n_kfold = 1):
        np.random.seed(self.seed)

        n = self.X.shape[0]
        random_index = np.arange(n) 
        n_splits = (n) / self.X_test.shape[0] 
        
        if np.ceil(n/n_splits)!=int(n/n_splits):
            raise ValueError(f"n splits in k-fold needs to be an integer but is {n_splits:.2f} with current test size") 
        
        n_splits = int(n_splits)

        accuracies = np.zeros((n_kfold, n_splits))

        for j in range(n_kfold):
            random_index = random_index.ravel()
            np.random.shuffle(random_index)
            test_split_sets= random_index.reshape(n_splits,int(n/n_splits)) 
            i = 0
            for test_indecies in test_split_sets:
                train_indecies = np.delete(test_split_sets, int(i), 0).ravel()
                self.X_test = self.X[test_indecies]
                self.y_test = self.y[test_indecies]
                self.X_train = self.X[train_indecies]
                self.y_train = self.y[train_indecies]
                self.solve()
                accuracies[j, i] = self.predict()[1]
                i += 1
        mean_accuracy = np.mean(accuracies)
        return accuracies, mean_accuracy
 
    



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

#"""
for z in [0,1,4]:
    input_path = 'big_test.csv'
    r = read_csv.ReadCSV(input_path)
    r._remove_series([z, 2, 3, 5])
    X_df, y_df = r.get_df()

    # Shaping data to be compatible with the SVM class
    X = X_df.values
    y = y_df.values[:,0]
    y = np.where(y == 1, 1, -1)
    y = y.reshape(-1,1)

    
    n = 5
    c = [0.1, 0.5, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
    m = len(c)
    gamma = np.logspace(-3, 1, n)
    accuracy = np.zeros((n,m))
    
    for i in range(n):
        for j in range(m):
            obj = SVM(X, y, gamma=gamma[i], C=c[j], test_size = 0.25, seed=0)
            accuracy[i,j] = obj.k_fold(n_kfold=30)[1]

    # Heatmap of accuracies for different C and gamma

    plt.figure(figsize=(12,8))
    plt.title("SVM(All features)")
    sns.heatmap(accuracy, annot=True, fmt='.5g', 
            cbar_kws={'label': "Accuracy"}, 
            xticklabels = [f"{x_val:.5g}" for x_val in c],
            yticklabels=[f"{y_val:.3e}" for y_val in gamma]) 
    plt.xlabel(f"$C$")
    plt.ylabel(f"$\\gamma$")

    plt.savefig(f"accuracy(C,gamma){z}.pdf")
    #plt.show()
    

    # Feature selection

    # Using the optimal parameters
    gamma = gamma[np.where(accuracy == accuracy.max())[0][0]]
    c = c[np.where(accuracy == accuracy.max())[1][0]]
    
    n = X.shape[1]
    accuracy = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            X_two_features = np.c_[X[:,i], X[:,j]]
            obj = SVM(X_two_features, y, gamma=gamma, C=c, seed=0)
            accuracy[i,j] = obj.k_fold(n_kfold=20)[1]

    # Heatmap of accuracies obtained with different feature pairs
    plt.figure(figsize=(12,8), tight_layout = True)
    plt.title("SVM(Feature 1, Feature 2)")
    sns.heatmap(accuracy, annot=True, fmt='.3f', 
            cbar_kws={'label': "Accuracy"}, 
            xticklabels = [f"{x_val}" for x_val in X_df.columns],
            yticklabels=[f"{y_val}" for y_val in X_df.columns]) 
    plt.xlabel(f"Feature 1")
    plt.ylabel(f"Feature 2")

    plt.savefig(f"feature_pairs{z}.pdf")
    #plt.show()
    

    a = np.where(accuracy == accuracy.max())[0][-1]
    b = np.where(accuracy == accuracy.max())[1][-1]
    
    accuracy = np.zeros(n)
    
    for i in range(n):

        X_three_features = np.c_[X[:,a], X[:,b], X[:,i]]
        obj = SVM(X_three_features, y, gamma=gamma, C=c, seed=0)
        accuracy[i] = obj.k_fold(n_kfold=50)[1]
    
    
    x = np.arange(n)

    # Bar plot of accuracies obtained adding a third feature to the optimal feature pair
    plt.figure(figsize=(12,8), tight_layout = True)
    plt.ylabel("Accuracy")
    plt.xlabel("Third feature")
    plt.title("SVM("+X_df.columns[a]+", "+X_df.columns[b]+", Third feature)")
    plt.xticks(x, X_df.columns, rotation=20)
    plt.bar(x, accuracy)
    plt.savefig(f"third_feature{z}.pdf")
    plt.show()
    
#"""


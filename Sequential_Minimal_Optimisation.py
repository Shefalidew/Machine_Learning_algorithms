import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random as rnd

#reading the data from .csv file
df = pd.read_csv('C:/Users/shefa/OneDrive/Documents/Masters/Machine learning for Autonomous Robots/download/[004]ex04_material/ex04_material/smo_dataset.csv',sep = ';')
y = df['y']
x = df.drop('y', axis=1)

y = np.asarray(y.values)
x = np.asarray(x.values)

#dividing data into training and test sets
X_train, X_test, Y_train, Y_test = tts(x, y, test_size = 0.4, random_state = 5)

class SVM():
    def __init__(self, max_iter=100, kernel_type='linear', C=1.0, epsilon=0.001):
        self.kernels = {'linear' : self.kernel_linear} #a dictionary in case we want different types of kernels to be used
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon
        
        
    def fit(self, X, y):
        n, d = X.shape[0], X.shape[1]
        self.alpha = np.zeros((n))
        self.kernel = self.kernels[self.kernel_type]
        count = 0
        while True:
            count += 1
            alpha_prev = np.copy(self.alpha)
            for j in range(0, n):
                i = self.get_rnd_int(0, n-1, j) # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                #The second derivative of the objective function along the diagonal line
                k_ij = self.kernel(x_i, x_i) + self.kernel(x_j, x_j) - 2 * self.kernel(x_i, x_j)
                if k_ij == 0:
                    continue
                self.alpha_prime_j, self.alpha_prime_i = self.alpha[j], self.alpha[i]
                (L, H) = self.compute_L_H(self.C, self.alpha_prime_j, self.alpha_prime_i, y_j, y_i)
               
                # Compute model parameters
                self.w = self.calc_w(self.alpha, y, X)
                self.b = self.calc_b(X, y, self.w)

                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)
                
                # Set new alpha values
                self.alpha[j] = self.alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
                self.alpha[j] = max(self.alpha[j], L)
                self.alpha[j] = min(self.alpha[j], H)
                #alpha_1_new is computed from the new,clipped,alpha_2
                self.alpha[i] = self.alpha_prime_i + y_i*y_j * (self.alpha_prime_j - self.alpha[j])

            # Check convergence
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.epsilon:
                break
            if count >= self.max_iter:
                print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                return
        self.b = self.calc_b(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(self.alpha, y, X)
            
        # Get support vectors
        alpha_idx = np.where(self.alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors, count
    
    
    def predict(self, X):
        val =self.u(X, self.w, self.b)
        return val
    
    #calculating threshold
    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T) 
        return np.mean(b_tmp)
    
    #calculating the normal vector to the hyperplane
    def calc_w(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha,y)) 
    
    #Output of a linear SVM
    def u(self, X, w, b):
        result = np.sign(np.dot(w.T, X.T) + b).astype(int)
        return result
    
    #calculating error on the value of the training sample
    def E(self, x_k, y_k, w, b):
        return self.u(x_k, w, b) - y_k
    
    
    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))
        
    def get_rnd_int(self, a,b,z):
        i = z
        cnt=0
        while i == z and cnt<1000:
            i = rnd.randint(a,b)
            cnt=cnt+1
        return i
    
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)
    
    def decision_function(self,y, X):
        """Applies the SVM decision function to the input feature vectors in `x_test`."""

        result = np.dot((self.alpha * y),self.kernel_linear(X,X_train)) - self.b
        return result

def plot_decision_boundary(model,X,y, ax, resolution=100, colors=('b', 'k', 'r'), levels=(-1, 0, 1)):
    """Plots the model's decision boundary on the input axes object.
    Range of decision boundary grid is determined by the training data.
    Returns decision boundary grid and axes object (`grid`, `ax`)."""
        
    # Generate coordinate grid of shape [resolution x resolution]
    # and evaluate the model over the entire space
    xrange = np.linspace(X[:,0].min(), X[:,0].max(), resolution)
    yrange = np.linspace(X[:,1].min(), X[:,1].max(), resolution)
    grid = [[model.decision_function( y,np.array([xr, yr])) for xr in xrange] for yr in yrange]
    grid = np.array(grid).reshape(len(xrange), len(yrange))
        
    # Plot decision contours using grid and
    # make a scatter plot of training data
    ax.contour(xrange, yrange,grid, levels=levels, linewidths=(1, 1, 1),linestyles=('--', '-', '--'), colors=colors),ax.scatter(X[:,0], X[:,1],c=y, cmap=plt.cm.viridis, lw=0, alpha=0.25)
        
    # Plot support vectors (non-zero alphas)
    # as circled points (linewidth > 0)
    mask = np.round(model.alpha, decimals=2) != 0.0
    ax.scatter(X[mask,0], X[mask,1],c=Y_train[mask], cmap=plt.cm.viridis, lw=1, edgecolors='k')
        
    return grid, ax
    

if __name__ == "__main__":
    model = SVM(max_iter=10, kernel_type='linear', C=1000.0, epsilon=0.001)
    model.fit(X_train, Y_train)

    Y_predicted = [model.predict(x) for x in X_test]

    cm = confusion_matrix(Y_test, Y_predicted)
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    print(accuracy)

    fig, ax = plt.subplots()
    grid, ax = plot_decision_boundary(model,X_train,Y_train, ax)
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.title ('smo_dataset with separating hyperplane')

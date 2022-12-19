# DO NOT USE ANY OTHER IMPORTS!
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris


def apply_k_fold_cv(X, y, train_and_eval=None, n_folds: int = 5, **kwargs):
    """K fold cross validation.

    Parameters
    ----------
    X : array-like, shape (n_samples, feature_dim)
        The data for the cross validation

    y : array-like, shape (n-samples, label_dim)
        The labels of the data used in the cross validation

    train_and_eval : function
        The function that is used for classification of the training data

    n_folds : int, optional (default: 5)
        The number of folds for the cross validation

    kwargs :
        Further parameters that get used e.g. by the classifier

    Returns
    -------
    accuracies : array, shape (n_splits,)
        Vector of classification accuracies for the n_splits folds.
    """
    assert X.shape[0] == y.shape[0]

    if len(X.shape) < 2:
        X = np.atleast_2d(X).T
    if len(y.shape) < 2:
        y = np.atleast_2d(y).T

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in cv.split(X):
        train_data = X[train_index, :]
        train_label = y[train_index, :]
        test_data = X[test_index, :]
        test_label = y[test_index, :]

        score = train_and_eval(train_data, test_data, train_label, test_label, **kwargs)

        scores.append(score)

    return np.array(scores)


class NaiveBayes:
    """Naive Bayes Classifier"""

    def __init__(self):
        self.X = np.array
        self.y = np.array
        self.n_samples = int
        self.n_features = int        
        self.n_bins=5

    #splitting continuous features into 5 equidistant bins
    def split_into_bins(self,feature_arr):
        self.n_bins
        binwidth = int(len(feature_arr)/self.n_bins)
        bin = []
        x = 0
        #split the data-points in each feature into 5 equidistant bins
        for i in range(self.n_bins):
            bin.append(feature_arr[x:x+binwidth])
            x+=binwidth
        updated_bin = []
        #dividing the data-points in each bin into different categories
        for j in range(len(bin)):
            updated_bin.extend([j for x in range(len(bin[j])) ])
        return updated_bin

    #this function gives a list of list containing all the elements belonging to each feature
    def _accessing_features(self, X:ArrayLike):
        features = []
        #iterating over columns
        for i in range(len(X[0])):
            feature_val = []
            #iterating over rows
            for j in range(len(X)):
                feature_val.append(X[j][i])
            features.append(feature_val)
        return np.array(features)

    def train(self, X: ArrayLike, y: ArrayLike, discretize: bool = False):
        """Trains the classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, feature_dim)
            The data for the training of the classifier

        y: array-like, shape (n-samples, label_dim)
            The labels for the training of the classifier

        discretize: bool
            Flag to indicate if continuous features will be discretized into 5 bins or not.
        """

        # TODO: implement me!
        self.X = X
        self.y = y
        self.n_samples, self.n_features = self.X.shape
        self._classes = np.unique(self.y)
        self.n_classes = len(self._classes)
        self._mean = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self._var = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self._priors = np.zeros(self.n_classes, dtype=np.float64)
        self.discretize = discretize

        # calculate mean, var, and prior for each class
        for idx, c in enumerate(self._classes):
            result = np.where(self.y==c)
            X_c = self.X[result[0]]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(self.n_samples) #this gives me p_y

        if self.discretize is True:
            features = self._accessing_features(X)
            
            #Every feature is divided into categorical equidistant bins
            self.updated_feature_1 = self.split_into_bins(features[0])
            self.updated_feature_2 = self.split_into_bins(features[1])
            self.updated_feature_3 = self.split_into_bins(features[2])
            self.updated_feature_4 = self.split_into_bins(features[3])
            list_updated_features = np.array([updated_feature_1,updated_feature_2,updated_feature_3,updated_feature_4])
            self.count_matrix=[]
            for i in range(len(list_updated_features)):
                count_feature=[]
                for c in self._classes:
                    idx_label = np.where(y==c)
                    r = list_updated_features[i][idx_label[0]]
                    val = []
                    for j in range(self.n_bins):
                        val.append(np.count_nonzero([x==j for x in r]))
                    count_feature.append(val)
                self.count_matrix.append(count_feature)
            self.likelihood = self._likelihood_categorical(self.count_matrix,self.n_features)

    def _likelihood_categorical(self,count_matrix,n_features):
        #this gives me p_x_given_y
        final_likelihood = []
        for i in range(n_features):
            likelihood=[]
            for j in range(len(self.count_matrix[i])):
                likelihood.append(self.count_matrix[i][j]/np.sum(self.count_matrix[i][j]))
            final_likelihood.append(likelihood)
        return final_likelihood 
        

    def predict(self, X: ArrayLike) -> ArrayLike[int]:
        """Trains the classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, feature_dim)
            The data points to be classified.

        Returns
        ----------
        An ArrayLike of the shape (n_samples, 1) containing the predicted class label for each data point.
        """

        # TODO: implement me!
        if self.discretize is False:
            y_pred = [self._predict(x) for x in X]
        else:
            y_pred = [self._predict_categorical(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, feat_val):
        #considering gaussian distribution
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((feat_val - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_categorical(self,x):
        for idx, c in enumerate(self._classes):
            # Intializing an empty array
            posterior = np.zeros((1,self.n_classes))
            # For each feature
            for i in range(self.n_features):
                feat_val = x[i]
                # Fetch the corresponding log_probability table and add continue to add them for all the features
                probs+=np.log(self.likelihood)[i][:,feat_val]
            # Finally add posterior probability
            posterior+=np.log(self._priors[idx])
            # Finding the maximum of the probabilities and fetching the corresponding class
        return self._classes[np.argmax(posterior)]

    def evaluate(self, X: ArrayLike, y: ArrayLike) -> float:
        """Trains the classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, feature_dim)
            The test data points to be classified.

        y: array-like, shape (n-samples, label_dim)
            The labels of the test data.

        Returns
        ----------
        The accuracy of the classifier on the given data points.

        """

        # TODO: implement me!
        y_predict = np.array(self.predict(X))
        y_test = y
        correct = 0
        for i in range(len(y_test)):
            if y_test[i] == y_predict[i]:
                correct += 1
        accuracy = correct / float(len(y_test)) * 100.0
        return accuracy

    def calculate_score(self,X_train:ArrayLike,y_train:ArrayLike,X_test:ArrayLike,y_test:ArrayLike):
        self.train(X_train,y_train)
        score = self.evaluate(X_test,y_test)
        return score
    


if __name__ == '__main__':

    # TODO: implement
    X_data,y_data = load_iris(return_X_y=True)
    nb = NaiveBayes()
    nb.train(X_data, y_data,discretize =True)
    scores = apply_k_fold_cv(X_data,y_data,train_and_eval = nb.calculate_score,n_folds=5)
    plt.hist(scores)

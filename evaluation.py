#!/usr/bin/env python3

# do not use any other imports!
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import KNeighborsClassifier as BlackBoxClassifier
from sklearn.datasets import load_iris

class Evaluation:
    """This class provides functions for evaluating classifiers """
    
    def train_test_creation(self,fold, n_folds):
        ''' This function creates a tuple of train and test dataset 
        The input arguments are the complete data 
        and based on the number of folds required 
        it divides it into test and train dataset'''
        train_dataset = []
        test_dataset= []

        for k in range(n_folds):
            '''For every value of k in k-fold cv,
            each fold act as test_dataset once 
             '''       
            test_dataset = fold[k]
            train_dataset =fold[:k] + fold[k+1:]
        return (train_dataset,test_dataset)

    def generate_cv_pairs(self, n_samples, n_folds=5, n_rep=1, rand=False,y=None):
        """ Train and test pairs according to k-fold cross validation

        Parameters
        ----------

        n_samples : int
            The number of samples in the dataset

        n_folds : int, optional (default: 5)
            The number of folds for the cross validation

        n_rep : int, optional (default: 1)
            The number of repetitions for the cross validation

        rand : boolean, optional (default: False)
            If True the data is randomly assigned to the folds. The order of the
            data is maintained otherwise. Note, *n_rep* > 1 has no effect if
            *random* is False.

        y : array-like, shape (n_samples), optional (default: None)
            If not None, cross validation is performed with stratification and
            y provides the labels of the data.

        Returns
        -------

        cv_splits : list of tuples, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split. The list has the length of
            *n_folds* x *n_rep*.

        """
        ### YOUR IMPLEMENTATION GOES HERE ###
        
        cv_splits =[]
        #for dividing the dataset into equal sized groups of size = group_size
        group_size = int(n_samples/n_folds)
        data= np.array([x for x in range(0,n_samples)])
        
        # raising errors with cause for proper generation of cv_pairs    
        if n_rep == 0:
            print('Value of repetition parameter cannot be less than 1')
            raise ValueError
        
        if n_folds<2:
            print('Minimum 2 groups are required to be split into train and test set')
            raise ValueError
        
        if n_samples<n_folds:
            print('Number of samples cannot be lesser than the number of folds')
            raise ValueError
        
        if y is None:
            #for cross-validation with repetition
            for j in range(n_rep):
                fold = []
                x = 0
            
                #for cross-validation with randomization
                if rand==True:
                    data = random.choices(data,k=n_samples) #randomly shuffling the complete dataset                
                
                #creating different groups/folds
                for i in range(n_folds):
                    fold.append(data[x:x+group_size])
                    x = x+group_size

                # to create train-test pairs
                train_data,test_data = self.train_test_creation(fold,n_folds)
                
        # for cross-validation with stratification
        # if y is given as a list of class labels
        else:
            #to check names and counts of different labels
            uniques, counts = np.unique(y, return_counts=True)
            packet = dict(zip(uniques, counts ))
            #this gives a list of all the labels present in the iris_dataset
            list_labels=list(packet.keys())
            
            #this will consists of indices of objects belonging to same class as an element of a list
            list_of_indices = []
            for m in list_labels:
                indices = []
                for n in range(len(y)):
                    if y[n]== m:
                        indices.append(n)
                list_of_indices.append(indices)
                # it consists of list of lists having elements consisting the indices of each label
                
            # again for repetition
            for j in range(n_rep):
                big_fold=[]
                #if randomization is required
                if rand == True:
                    for label in list_labels:
                        #it shuffles the indices of each label
                        list_of_indices[label] = random.choices(list_of_indices[label],k=packet[label])
                            
                for i in range(n_folds):
                    x=0
                    fold=[]
                    for l in list_labels:
                        label_size = int((len(list_of_indices[l]))/int(n_folds))
                        if label_size % 1 != 0:
                            print(f'The samples of label {l} cannot be divided into {n_folds} groups equally')
                        fold.append(list_of_indices[l][x:x+label_size])
                        x +=label_size
                    big_fold.extend(fold)
            # to create train-test pairs
            train_data, test_data = self.train_test_creation(big_fold,n_folds)
            
            
        cv_splits.append((train_data,test_data))
        return cv_splits
            

    def apply_cv(self, X, y, train_test_pairs, classifier):
        """ Use cross validation to evaluate predictions and return performance

        Apply the metric calculation to all test pairs

        Parameters
        ----------

        X : array-like, shape (n_samples, feature_dim)
            All data used within the cross validation

        y : array-like, shape (n-samples)
            The actual labels for the samples

        train_test_pairs : list of tuples, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split

        classifier : function
            Function that trains and tests a classifier and returns a
            performance measure. Arguments of the functions are the training
            data, the testing data, the correct labels for the training data,
            and the correct labels for the testing data.

        Returns
        -------

        performance : float
            The average metric value across train-test-pairs
        """
        ### YOUR IMPLEMENTATION GOES HERE ###
        
        list_of_accuracy = []
        for pair in train_test_pairs:
            #taken one tuple containing train and test data
            index_train_values= pair[0]
            index_test_values =pair[1]
            
            #as index_train_values is a list of list 
            #and scikit-learn expects 2d num arrays for the training dataset for a fit function.
            train_values=[]
            for i in range(len(index_train_values)):
                train_values.extend(index_train_values[i])
                

            X_train = X[train_values]
            X_test = X[index_test_values]
            y_train = y[train_values]
            y_test = y[index_test_values]
            accuracy = classifier(X_train,X_test,y_train,y_test)
            list_of_accuracy.append(accuracy)

        performance = float(sum(list_of_accuracy)/len(train_test_pairs))
        return performance


    def black_box_classifier(self, X_train, X_test, y_train, y_test):
        """ Learn a model on the training data and apply it on the testing data

        Parameters
        ----------

        X_train : array-like, shape (n_samples, feature_dim)
            The data used for training

        X_test : array-like, shape (n_samples, feature_dim)
            The data used for testing

        y_train : array-like, shape (n-samples)
            The actual labels for the training data

        y_test : array-like, shape (n-samples)
            The actual labels for the testing data

        Returns
        -------

        accuracy : float
            Accuracy of the model on the testing data
        """
        bbc = BlackBoxClassifier(n_neighbors=10)
        bbc.fit(X_train, y_train)
        acc = bbc.score(X_test, y_test)
        return acc

if __name__ == '__main__':
    # Instance of the Evaluation class
    e = Evaluation()

    ### YOUR IMPLEMENTATION FOR PROBLEM 1.1 GOES HERE ###
    X_data,y_data = load_iris(return_X_y=True)
    train_test_pairs = e.generate_cv_pairs(len(y_data),n_folds=10,n_rep=5,rand=False,y=y_data)
    print(train_test_pairs)
    performance = e.apply_cv(X=X_data,y=y_data,train_test_pairs=train_test_pairs,classifier=e.black_box_classifier)
    print(f'Accuracy of the train-test pair:{performance}')

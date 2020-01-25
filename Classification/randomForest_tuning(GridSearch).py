# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:38:23 2019

@author: sabab05
"""

# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.grid_search import GridSearchCV

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import KFold
# Load pandasfile:///D:/CLoud/Academic/Research/___/Analysis (Photoshop)/4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)/_Working/randomForest(%25split).py
import pandas as pd

from sklearn.preprocessing import StandardScaler 
# Load numpy
import numpy as np

from sklearn.model_selection import cross_val_score  


def read_csv(csv_file, nrows=None):
    df = pd.read_csv(csv_file, nrows=nrows)
    print("File = {}".format(csv_file))
    print("Shape = {:,} rows, {:,} columns".format(df.shape[0], df.shape[1]))
    print("Memory usage = {:.2f}GB".format(df.memory_usage().sum() / 1024**3))
    return df

def classifierRandomForest(start, end, iteration,seed):
    accuracyList = []
    stdList = []
    n_est = []
    for i in range(start, end+1, iteration):
        clf = RandomForestClassifier(
        n_estimators=i,
        criterion='gini',
        max_depth=100,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=15,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=seed,
        verbose=0,
        warm_start=False,
        class_weight='balanced'
        )
        all_accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)
#        print(all_accuracies) 
#        print(all_accuracies.mean())
#        print(all_accuracies.std())
        stdList.append(all_accuracies.std())
        accuracyList.append(all_accuracies.mean())
        n_est.append(i)

    return n_est, accuracyList, stdList 
    

data_dir = r'''D:\CLoud\Academic\Research\___\Analysis (Fusion 360)\1. Feature (Word)\Topic Model Output\topic_distribution_mallet_20_V1.csv'''
##data_dir = r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\_Working\topic_distribution_mallet_30_V1_train.csv'''

df = read_csv(data_dir)


# Set random seed
np.random.seed(0)

#print('The shape of our features is:', df.shape)
#print(df.head(5))
#print(df.describe())

labelIndex =  df.columns.get_loc("Label")
onlyTopic = labelIndex-5
X = df.iloc[:, 1:labelIndex].values  
y = df.iloc[:, labelIndex].values  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=0) # 70% training and 30% test
#

#n_est, accuracy, stdDev = classifierRandomForest(490,490,10,0)

#m = max(accuracy)
#index = accuracy.index(max(accuracy))
#print("Accuracy: "+repr(m))
#print("Std Deviation: "+ repr(stdDev[index]))
#print("n_estimator: "+repr(n_est[index]))

param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 400, 500]
}
# Create a based model
rf = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True, random_state=0) 
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 10)
grid_search.fit(X, y)
print(grid_search.best_params_)



# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import RandomForestRegressor

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import KFold
# Load pandas
import pandas as pd

# Load numpy
import numpy as np

from sklearn.model_selection import cross_val_score  

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import StratifiedKFold

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    #train_sizes = [i for i in range(100,675,25)]
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.errorbar(train_sizes, train_scores_mean,yerr= train_scores_std,fmt='o',color="r", capsize=3)
    plt.errorbar(train_sizes, test_scores_mean,yerr= test_scores_std,fmt='o',color="g", capsize=3)
#    plt.errorbar(train_sizes, train_scores_mean,yerr= train_scores_std,fmt='o',color="r", ecolor='lightgray', elinewidth=3, capsize=3)
#    plt.errorbar(train_sizes, test_scores_mean,yerr= test_scores_std,fmt='o',color="g", ecolor='lightgray', elinewidth=3, capsize=3)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
  
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    print(train_sizes)
    print(train_scores_mean)
    print(test_scores_mean)
    plt.legend(loc="best")
    return plt


def read_csv(csv_file, nrows=None):
    df = pd.read_csv(csv_file, nrows=nrows)
    print("File = {}".format(csv_file))
    print("Shape = {:,} rows, {:,} columns".format(df.shape[0], df.shape[1]))
    print("Memory usage = {:.2f}GB".format(df.memory_usage().sum() / 1024**3))
    return df

#data_dir = r'''D:\CLoud\Academic\Research\___\Analysis (Fusion 360)\1. Feature (Word)\Topic Model Output\topic_distribution_mallet_20_V1.csv'''
data_dir = r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\_Working\topic_distribution_mallet_30_V1_train_.csv'''

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

 





#X_train = X
#X_test = X_t
#y_train = y
#y_test = y_t


clf = RandomForestClassifier(
        n_estimators=360,
        criterion='gini',
        max_depth=100,
        min_samples_split=13,
        min_samples_leaf=2,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=0,
        verbose=0,
        warm_start=False,
        class_weight='balanced'
        )


#Train the model using the training sets y_pred=clf.predict(X_test)

title = "Learning Curves (Random Forest)"

#cv_ = ShuffleSplit(n_splits = 10, test_size = 0.30, random_state = 0)
cv_ = StratifiedKFold(n_splits=10 ,random_state= 1, shuffle = True)

estimator = clf
#estimator = GaussianNB()
#estimator = SVC(gamma=0.001)


plot_learning_curve(estimator, 
                    'Learning Curves',
                    X, y, 
                    cv = cv_,
                    n_jobs = 1,
                    ylim=(0,1.1))



 



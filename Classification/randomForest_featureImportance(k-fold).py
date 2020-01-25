# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

import matplotlib.pyplot as plt

from statistics import mean, stdev

import seaborn as sns

from sklearn.model_selection import StratifiedKFold
# Load pandas
import pandas as pd

# Load numpy
import numpy as np

from numpy import array
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score  


def average(nums, default=float('nan')):
    return sum(nums) / float(len(nums)) if nums else default

def read_csv(csv_file, nrows=None):
    df = pd.read_csv(csv_file, nrows=nrows)
    print("File = {}".format(csv_file))
    print("Shape = {:,} rows, {:,} columns".format(df.shape[0], df.shape[1]))
    print("Memory usage = {:.2f}GB".format(df.memory_usage().sum() / 1024**3))
    return df

data_dir = r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\_Working\topic_distribution_mallet_30_V1_train.csv'''


df = read_csv(data_dir)



# Set random seed
np.random.seed(0)


labelIndex =  df.columns.get_loc("Label")


onlyTopic = labelIndex-5
X = df.iloc[:, 1:labelIndex].values  
y = df.iloc[:, labelIndex].values  





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=0) # 70% training and 30% test

#X_train = X
#X_test = X_t
#y_train = y
#y_test = y_t


#clf = RandomForestClassifier(
#    n_estimators=490,
#    criterion='gini',
#    max_depth=100,
#    min_samples_split=2,
#    min_samples_leaf=1,
#    min_weight_fraction_leaf=0.0,
#    max_features=15,
#    max_leaf_nodes=None,
#    min_impurity_decrease=0.0,
#    min_impurity_split=None,
#    bootstrap=True,
#    oob_score=False,
#    n_jobs=-1,
#    random_state=1,
#    verbose=0,
#    warm_start=False,
#    class_weight='balanced'
#)


clf = RandomForestClassifier(
        n_estimators=360,
        criterion='gini',
        max_depth=100,
        min_samples_split=13,
        min_samples_leaf=2,
        min_weight_fraction_leaf=0.0,
        max_features=15,
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


featureImpValues = []

kf = StratifiedKFold(n_splits=10, random_state= 0, shuffle = True)
count = 1
# test data is not needed for fitting
accuracyList = []
stdList = []
for train, testInd in kf.split(X_train, y_train):
    
    xxtr = X_train[train, :]
    yytr = y_train[train]
    
    xxts = X_train[testInd, :]
    yyts = y_train[testInd]
    clf.fit(X_train[train, :],y_train[train])
    y_pred=clf.predict(X_train[testInd, :]) 
    confMat = confusion_matrix(y_train[testInd], y_pred)
    modelAccuracy = metrics.accuracy_score(y_train[testInd], y_pred)
    
    
    accuracyList.append(modelAccuracy)
    print("Accuracy:",modelAccuracy)
    
    
    
    
    # sort the feature index by importance score in descending order
#    feature_imp = pd.Series(clf.feature_importances_,df.columns.values[1:labelIndex ]).sort_values(ascending=False)
    feature_imp = (pd.Series(clf.feature_importances_,df.columns.values[1:labelIndex ]).tolist())    
    #feature_labels = feature_imp.index
    featureImpValues.append(feature_imp)
    
#    plt.figure()
#    plt.bar(feature_labels, clf.feature_importances_[label])
#    plt.xticks(feature_labels, rotation='vertical')
#    plt.ylabel('Importance')
#    plt.xlabel('Features')
#    plt.title('Fold {}'.format(count))
#    count = count + 1
#plt.show()
#

feature_imp = []
feature_imp = [average(feature) for feature in zip(*featureImpValues)]
#commnet out the following if you want to normalize x axis within the range [0-1]
#feature_imp = [average(feature)/max(feature_imp)*100 for feature in zip(*featureImpValues)]
feature_sum = sum(feature_imp)
feature_labels=df.columns.values[1:labelIndex ]

features= pd.Series(feature_imp,feature_labels).sort_values(ascending=False)


print("Mean Accuracy:",mean(accuracyList))
print("Standard Deviation", stdev(accuracyList))

print(features)
print(feature_sum)

# Creating a bar plot
sns.barplot(x=features, y=features.index)
# Add labels to your graph
plt.xlabel('Relative Feature Importance Score', fontsize=18)
#plt.ylabel('Features',fontsize=18)
plt.title("Visualizing Important Features",fontsize=28)
plt.legend()
plt.show()



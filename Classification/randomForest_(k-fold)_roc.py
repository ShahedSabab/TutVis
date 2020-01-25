# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from scipy import interp
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, roc_auc_score,roc_curve, auc

from sklearn.ensemble import RandomForestRegressor

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

from statistics import mean, stdev

import seaborn as sns

from sklearn.model_selection import StratifiedKFold
# Load pandas
import pandas as pd

# Load numpy
import numpy as np

from sklearn import preprocessing

from numpy import array
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score,cross_val_predict


def average(nums, default=float('nan')):
    return sum(nums) / float(len(nums)) if nums else default

def read_csv(csv_file, nrows=None):
    df = pd.read_csv(csv_file, nrows=nrows)
    print("File = {}".format(csv_file))
    print("Shape = {:,} rows, {:,} columns".format(df.shape[0], df.shape[1]))
    print("Memory usage = {:.2f}GB".format(df.memory_usage().sum() / 1024**3))
    return df


    

data_dir = r'''D:\CLoud\Academic\Research\___\Analysis (Fusion 360)\1. Feature (Word)\Topic Model Output\topic_distribution_mallet_20_V1.csv'''
#data_dir = r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\_Working\topic_distribution_mallet_30_V1_train.csv'''


df = read_csv(data_dir)



# Set random seed
np.random.seed(0)


labelIndex =  df.columns.get_loc("Label")


onlyTopic = labelIndex-5
X = df.iloc[:, 1:labelIndex].values  
y = df.iloc[:, labelIndex].values  

# converts the nominal labels to binary which is necessary for calculating ROC 
y = pd.get_dummies(y).values[:,0]



clf = RandomForestClassifier(
    n_estimators=390,
    criterion='gini',
    max_depth=100,
    min_samples_split=2,
    min_samples_leaf=1,
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



cv = StratifiedKFold(n_splits=10, random_state= 0, shuffle = True)
classifier = clf
accuracyList = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()




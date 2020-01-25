# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 02:03:18 2018

@author: sabab05
"""
from __future__ import print_function
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_ind_from_stats
from scipy.special import stdtr

# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 

path = r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\1. Feature (Words)\Topic Model\Trial COdes\Model\topic_distribution_mallet_30_V1.csv'''
adv_topic_avg = []
beg_topic_avg = []
labels = ['Advanced','Beginner']


def find_avg(column_list,df):
    topic_avg = []
    for column in column_list:
        topic_avg.append(np.average(df[column]))
    return topic_avg        

def calculate_ind_t_test(adv_rows, beg_rows, column_list):
    t = []
    p = []
    data = []
    sig = {}
    t,p= ttest_ind(adv_rows[column_list], beg_rows[column_list], equal_var=False)
    
    for row in p:
        if row < 0.05:
            data.append(row)
        else:
            data.append(0)
            
    sig = dict(zip(column_list, data))
   
    return sig,t,p
    
    

number_of_adv=375 
number_of_beg=375    
df = pd.read_csv(path) 


adv_rows = df.head(number_of_adv)
beg_rows = df.tail(number_of_beg)



column_list = list(df.columns.values[1:-1])
adv_topic_avg = find_avg(column_list, adv_rows)
beg_topic_avg = find_avg(column_list, beg_rows)
sig, t, p = calculate_ind_t_test(adv_rows, beg_rows, column_list)

for key, value in sig.items(): 
    print(key+" = "+str(value))





fig, ax = plt.subplots()

n_groups = len(column_list)

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.6
error_config = {'ecolor': '0.2'}

rects1 = ax.bar(index, adv_topic_avg, bar_width,
                alpha=opacity, color='b',
                error_kw=error_config,
                label='Advanced')

rects2 = ax.bar(index + bar_width, beg_topic_avg, bar_width,
                alpha=opacity, color='r',
                 error_kw=error_config,
                label='Beginner')

xtickLabel = []
for i in range (1,len(column_list)+1):    
    xtickLabel.append("T"+str (i))



ax.set_xlabel('Topics',fontsize=16)
ax.set_ylabel('Average Distribution',fontsize=16)
ax.set_title('Advanced vs Beginner Document Topic Distribution',fontsize=16)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(xtickLabel)
ax.legend(fontsize=16)

fig.tight_layout()
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 17:00:08 2018

@author: sabab05
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:43:03 2018

@author: sabab05
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 02:03:18 2018

@author: sabab05
"""

import pandas as pd 
from collections import Counter
import plotly.plotly as py
import plotly.tools as tls
import matplotlib.pyplot as plt
import numpy as np
import random
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 

path = r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\1. Feature (Words)\Topic Model\Trial COdes\Model\dominant_topic_mallet_30_V1.csv'''
adv_topic_avg = []
beg_topic_avg = []
labels = ['Advance','Beginner']


def freq_dist(list_item):
    item_freq = Counter(list_item) 
    item_freq = sorted(item_freq.items(), key=lambda s: s[0])
    return item_freq       

def check_items(dict_item):
    global number_of_topics
    
    for x in range(1,number_of_topics+1):
        if float(x) not in dict_item.keys():
            dict_item[float(x)]=0
    dict_item = sorted(dict_item.items(), key=lambda s: s[0])
    return dict_item
    
        
number_of_adv=375 
number_of_beg=375    
number_of_topics = 30
opacity = 0.6
df = pd.read_csv(path) 


adv_rows = df.head(number_of_adv)
beg_rows = df.tail(number_of_beg)

adv_topic_list = adv_rows[(df.columns.values[1])]
beg_topic_list = beg_rows[(df.columns.values[1])]




adv_fd = dict(freq_dist(adv_topic_list))
beg_fd = dict(freq_dist(beg_topic_list))

#cehck items assign 0 to the missing topic value 
adv_fd = dict(check_items(adv_fd))
beg_fd = dict(check_items(beg_fd))

adv_sample = [adv_val for adv_val in adv_fd.values()]
beg_sample = [beg_val for beg_val in beg_fd.values()]


topic_wise_total_documents = [sum(x) for x in zip(adv_sample, beg_sample)]
topic_wise_document_percentage = [round(x*100/sum(topic_wise_total_documents),2) for x in topic_wise_total_documents]



N = number_of_topics
xtickLabel = []
for i in range (1,number_of_topics+1):    
    xtickLabel.append("T"+str (i))

concat_func = lambda x,y: x + ":" + str(y)
xtickLabel = list(map(concat_func,xtickLabel,topic_wise_total_documents))    
concat_func = lambda x,y: x + " (" + str(y)+"%)"
xtickLabel = list(map(concat_func,xtickLabel,topic_wise_document_percentage))

number_of_colors = number_of_topics

color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]

wid = [0.05]
explode = wid*number_of_topics
# =============================================================================
# pie chart 

fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

ticklabel = xtickLabel

data = topic_wise_total_documents

wedges, texts = ax.pie(data, wedgeprops=dict(width=0.4), startangle=-10, colors=color,explode = explode)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(ticklabel[i], xy=(x, y), xytext=(1.4*np.sign(x), 1.4*y),
                 horizontalalignment=horizontalalignment, **kw)
    

plt.show()
# =============================================================================



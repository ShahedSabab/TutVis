# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:41:14 2019

@author: sabab05
"""
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import random

def findMaxTopics(arr,numberOfTopics,topN):
    global topicIndex
    global topicContributionValue
    for j in range(0,topN):
        maximum = 0
        topic = ""
        for i in range (0,numberOfTopics):
        
            if arr[i]>maximum:
                maximum = arr[i]
                index = i;
                topic = "T"+str(i+1)
        print(topic+":"+repr(maximum))
        topicIndex.append(index+1)
        topicContributionValue.append(arr[index])
        arr[index]=0


def combinedTopics(arr,numberOfTopics):
    global topicType
    global topicTypeNum
    for i in range(0,numberOfTopics):
        if topicType[i] == 'S':
            length = len(topicTypeNum[i].split("+")) 
            topicCombinedNum = topicTypeNum[i].split("+") 
            print(topicCombinedNum) 
            #print(length)
            sumVal = 0
            initialIndex = topicCombinedNum[0]
            for j in range(0, length+1):
                if j == length:
                    arr[int(initialIndex)-1]=sumVal                    
                else:
                    sumVal += arr[int(topicCombinedNum[j])-1]
                    arr[int(topicCombinedNum[j])-1]=0
    
    return arr


fileNumber =488
totalTopics = 30
totalDocuments=750
topN  = 18
topicIndex = []
topicContributionValue= []


path = r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\1. Feature (Words)\Topic Model\Trial COdes\Model\topic_distribution_mallet_30_V1.csv'''
pathTopicFile = r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\1. Feature (Words)\Topic Model\Trial COdes\Model\topic_details_V1.csv'''
df = pd.read_csv(path) 
df1 = pd.read_csv(pathTopicFile)

#topic_number = [df['topic #']]
#topic_items = [x for x in df['topic_item']]

topics = {}

documents = []
topicList = []

document_number = df['0'] 
topicIndexName = df1['Name']
topicRename = df1['Rename']
topicType = df1['Comment']
topicTypeNum = df1['Combined']
documentList = [] 



for i in range(0, totalDocuments):
    for j in range (1, totalTopics+1):
        topicName = str('Topic '+repr(j))
        val = float(df.at[i,topicName])
        documents.insert(j-1,val)
    #documents.sort(reverse=True)    
    documentList.append(list(documents))
    documents.clear()
    

for i in range(fileNumber-1, fileNumber):
    print("Document "+str(i+1))
    print("--------------------")
    print(documentList[i])
    arr = combinedTopics(documentList[i],totalTopics)
    findMaxTopics(arr,totalTopics,topN)
    print("Topic "+repr(topicIndex)+":"+repr(topicContributionValue))

    

topic_wise_percentage = [round((x*100),2) for x in topicContributionValue]
#others = round(100.0-sum(topic_wise_percentage))
others = round(100.0 - np.sum(np.array(topic_wise_percentage), axis=0))
topic_wise_percentage.append(others)
topicContributionValue.append(1-np.sum(np.array(topicContributionValue), axis=0))
topicIndex.append(31)



N = topN+1
xtickLabel = []
for i in topicIndex:    
    ###for printing topic number only
    #xtickLabel.append("T "+str(i))
    ###for printing topic name
    #xtickLabel.append("T "+str(i)+":"+str(topicIndexName[i-1]))
    xtickLabel.append("T "+str(i)+":"+str(topicRename[i-1]))
    
concat_func = lambda x,y: x + "=" + str(round(y,2))
xtickLabel = list(map(concat_func,xtickLabel,topicContributionValue))    
concat_func = lambda x,y: x + " (" + str(y)+"%)"
xtickLabel = list(map(concat_func,xtickLabel,topic_wise_percentage))

number_of_colors = N

#color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])         
             for i in range(number_of_colors)]

wid = [0.05]
explode = wid*N
# =============================================================================
# pie chart 

fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect="equal"))

ticklabel = xtickLabel

data = topicContributionValue

wedges, texts = ax.pie(data, wedgeprops=dict(width=0.4), startangle=25, colors=color,explode = explode)

bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=0.72)
kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(ticklabel[i], xy=(x, y), xytext=(1.2*np.sign(x), 1.2*y),
                 horizontalalignment=horizontalalignment, **kw)
    
plt.title("Document #"+repr(fileNumber),fontsize=20)        
plt.show()
# =============================================================================

    
    
    
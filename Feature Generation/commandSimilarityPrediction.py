# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 03:52:32 2018

@author: sabab
"""

import nltk
import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sortedcontainers import SortedDict
from collections import OrderedDict
from collections import defaultdict




def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']

def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']

def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

def penn_to_wn(tag):
    if is_noun(tag):
        return wn.NOUN
    elif is_adjective(tag):
        return wn.ADJ
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    else:
        return None

def remove_stopwords(texts):
    stopWords = []
    returnString = ""
    with open(r'''D:\CLoud\Academic\Research\___\Backup\stopwords_en.txt''') as f1:    
        for line in f1:
            stopWords.append(line.strip())
            
    texts = re.sub('[\.\,\;]', '', texts)
    document = texts
    
    for words in stopWords:
        pattern = r'\b'+words+r'\b'
        index = [m.start() for m in re.finditer(pattern, document)]
        if len(index) > 0:
            document = re.sub(pattern, "",document)

    return document

def commandFrequency(finalstr,cmds,file):
    count =0
    document = finalstr
    commands = cmds
    data = dict()
    for item in commands:
        item = item.strip('\n')
        val=document.count(item)
        data[item] = val
        
    for key,value in data.items():
        #print(key + " => " + repr(value))
        if value > 0:
            key=key.replace(" ","_")
            #print(key + " => " + repr(value))
            count = count +1
            #file.writelines(key+"\n")
    #print(count)
    
def sort2NGram(commands):
 
    newCommandList = list()
    maxN = 0
    for item in commands:
        item = item.strip('\n')
        N=(len(item.split()))
        if(N>maxN):
            maxN=N
    for i in range(maxN,0,-1):    
        for item in commands:
            item = item.strip('\n')
            if len(item.split())==i:
                newCommandList.append(item)
    return newCommandList
    
def commandList(finalStr, cmds, file, fileTemp):
    document = finalStr
    commands = cmds
    commandIndexDict = dict()
    fileTemp.writelines(document)
    count = 0;
    for item in commands:
        item = item.strip('\n')
        pattern = r'\b'+item+r'\b'
        index = [m.start() for m in re.finditer(pattern, document)]
       # print(item +" = "+ repr(index))
        if len(index) > 0:
            for i in range (0,len(index)):
                commandIndexDict[index[i]] = item
            #the following line prints commands and found index in a file
            #file.writelines(item+"  "+repr(index)+"\n")
            
            # ---------------------- The following line print all the commands with index and frequency
            #print(item+"  "+repr(index)+" ("+repr(len(index))+")")
            count += len(index)
            replaceString=""
            for i in range (0,len(item)):
                replaceString = replaceString+"*"
            document = document.replace(item, replaceString)
    #the following line prints the total number of commands found in this tutorial
    #print(count)
    #the following line also sort a dict based on the key but in ascending
    #commandIndexDict = SortedDict(commandIndexDict)
    #the following line sort a dict but gives flexibity to sort it in ascending or descending order
    commandIndexDict = OrderedDict(sorted(commandIndexDict.items(), key=lambda v: v))
    commandWordCount = 0
    for key,value in commandIndexDict.items():
        commandWordCount = commandWordCount+len(value.split())
        file.writelines(value+"\n")

    #document = document.split()
    #print(document[0])
    return commandWordCount
            
def calculateCommandRatio(finalStr, commandCount):
    
    document = finalStr
    documentWordCount = len(document.split())
    commandWordCount = commandCount
    #print(documentWordCount)
    #print(commandWordCount)
    commandRatio = (commandWordCount*100)/documentWordCount
    commandRatio = round(commandRatio,2)
    
    return commandRatio, documentWordCount
   
def fileWrite(fileNumber):

    
    input_file =   r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\5. Feature(Command Ratio)\Data\Input\S('''+repr(fileNumber)+''').txt'''
    
    #input_file =   r'''D:\CLoud\Academic\Research\___\Photoshop COmmands\Tutplus Dataset (A-189, B-189)\B'''+repr(fileNumber)+'''.txt'''
    out_path = r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\5. Feature(Command Ratio)\Data\Output\S'''+repr(fileNumber)+'''_.txt'''
    out_temp = r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\5. Feature(Command Ratio)\Data\check\S'''+repr(fileNumber)+'''_.txt'''
    command_path = r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\5. Feature(Command Ratio)\Command List\commandList_processed_V1_.txt'''
    
    with open(command_path,"r") as f:    
        commands = f.readlines()
    
    fileOut =open(out_path,"w",encoding="utf-8")
    fileTemp =open(out_temp,"w",encoding="utf-8")
    f = open (input_file, 'r',encoding="utf-8",errors='ignore') 


        
    lines = ''.join(f.readlines())

    lines = lines.strip('\n')
    #lines= remove_stopwords(lines)
    #print(temp)
    lines = sent_tokenize(lines)

    #line = '\n'.join(sent_tokenize_list)
    
    finalString = ""
    for line in lines:
         #file1.write('>>>' +line+"\n")
         tags = nltk.pos_tag(word_tokenize(line))
         result = ""
         for tag in tags:
             wn_tag = penn_to_wn(tag[1])
             if wn_tag is not None:
                 result = result +" "+WordNetLemmatizer().lemmatize(tag[0],wn_tag)
                 result= re.sub('[^A-Za-z\.\,]+', ' ', result)
                 result = result.strip().lower()
         #file1.write(result+" ")
         finalString = finalString+result+" "
    
    commands = sort2NGram(commands)
    commandFrequency(finalString, commands, fileOut)
    commandCount = commandList(finalString, commands, fileOut, fileTemp)
    CR, L = calculateCommandRatio(finalString, commandCount)
    
    return CR, L

commandRatio = list()
documentLength = list()
documentNumber = list()

for x in range(1,751):
    print(x)
    CR, L = fileWrite(x)
    documentNumber.append("S"+repr(x))
    commandRatio.append(CR)
    documentLength.append(L)
    
print(documentNumber)
print(commandRatio)
print(documentLength)

path = r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\5. Feature(Command Ratio)\Result\out_V2.csv'''

df = pd.DataFrame()
df = pd.DataFrame({'Serial':documentNumber, 'Word Count (W/O SW)':documentLength, 'Command Ratio':commandRatio})
df.to_csv(path, index=False)

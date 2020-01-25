# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 03:05:09 2018

@author: sabab
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 03:52:32 2018

@author: sabab
"""

import nltk
import re
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
import numpy as np

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
    #elif is_adverb(tag):
        #return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    else:
        return None

def similarity(a,b):    
    matching_string = a
    checking_string = b
    
    documents=(matching_string,checking_string)
    #print(distance.levenshtein(matching_string, checking_string))
    #print(fuzz.ratio(matching_string, checking_string))
    
    tfidf_vectorizer=TfidfVectorizer(analyzer="char")
    tfidf_matrix=tfidf_vectorizer.fit_transform(documents)
    #print tfidf_matrix.shape
    
    cs=cosine_similarity(tfidf_matrix[0:1],tfidf_matrix)
    
    if cs[0,1] > 0.9:
        print(a)
    

def commandPredictor(finalStr,commands):

    sentence = finalStr
    for command in commands:
        nGram = len(command.split())
        sixgrams = ngrams(sentence.split(), nGram)
        for grams in sixgrams:
            check=' '.join(grams)
            similarity(command,check)

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
            print(key + " => " + repr(value))
            count = count +1
            file.writelines(key+"\n")
    print(count)
            


def fileWrite(fileNumber):

    porter_stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    
    stopWords = []

    with open(r'''D:\CLoud\Academic\Research\___\Backup\stopwords_en.txt''') as f1:    
         for line in f1:
             stopWords.append(line.strip())

    
    input_file =   r'''D:\CLoud\Academic\Research\___\Photoshop COmmands\4.1 Analysis (using 750 symmetrical data)\1. Feature (Words)\Tf IDF\Input\S('''+repr(fileNumber)+''').txt'''
    
    #input_file =   r'''D:\CLoud\Academic\Research\___\Photoshop COmmands\Tutplus Dataset (A-189, B-189)\B'''+repr(fileNumber)+'''.txt'''
    #out_path = r'''D:\CLoud\Academic\Research\___\Photoshop COmmands\3. Analysis\1. Feature (Words)\Tf IDF\Output\A'''+repr(fileNumber)+'''_.txt'''


    
    #fileOut =open(out_path,"w",encoding="utf-8")
    f = open (input_file, 'r',encoding="utf-8",errors='ignore') 

# =============================================================================
# #   
# =============================================================================
    #stopWords = list(get_stop_words('en'))     
        
    lines = ''.join(f.readlines())
    lines = lines.strip('\n')
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
       
        # fileOut.write(result+" ")
         finalString = finalString+result+" "
    
    finalString = finalString.split(' ')
    finalString = [word for word in finalString if word not in stopWords]
    finalString = " ".join(finalString)    
    #fileOut.write(finalString)
    #print(finalString)
    return finalString

out_path = r'''D:\CLoud\Academic\Research\___\Photoshop COmmands\4.1 Analysis (using 750 symmetrical data)\1. Feature (Words)\Tf IDF\Output\Output.txt'''    
fileOut =open(out_path,"w",encoding="utf-8")

corpus=[]

for x in range(1,751):
    print("File number: "+repr(x))
    corpus.append(fileWrite(x))

# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(corpus)
# summarize


vocab = vectorizer.vocabulary_.keys()

vocabList=[]
for x in vocab: 
    vocabList.append(x)

vocabOut=str(vocabList).strip("[]") 
vocabOut=vocabOut.replace('\'', '')

#print(vocabOut)

fileOut.write(vocabOut)
fileOut.write("\n")

#print(len(vectorizer.vocabulary_))

lst = []
trace=""
for doc in corpus:    
    vector = vectorizer.transform([doc])
   #print(vectorizer.get_feature_names())
    # summarize encoded vector
    #print(vector.shape)
    #print(vector.toarray())
    lst = vector.toarray().tolist()
    #trace = ','.join(str(x) for x in vector.toarray()) # '0,3,5'
    trace = str(lst)
    trace = trace.strip(']')
    trace = trace.strip('[')
    #print(trace)
    fileOut.write(trace)
    fileOut.write("\n")
    

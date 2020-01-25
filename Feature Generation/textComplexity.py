"""
Created on Sat Jan  5 01:21:04 2019

@author: sabab
"""

import requests
import re
import math

def writeSource(fileNum,texts):
    out_path = r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\3. Feature (Readability Index)\Source\S(''' + repr(fileNum) + ''').txt'''
    fileOut = open(out_path, "w", encoding="utf-8")
    fileOut.write(texts)

def readFile(fileNum):
    input_file = r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\1. Feature (Words)\Topic Model\Input\S(''' + repr(fileNum) + ''').txt'''

    with open(input_file, 'r', encoding="utf-8", errors='ignore') as f:
        lines = f.read().replace('\n', ' ')
    r = requests.post("http://www.readabilityformulas.com/freetests/six-readability-formulas.php", data={'text': lines, 'securitycheck': 2})

    #print(r.status_code, r.reason)
    htmlSource=r.text
    writeSource(fileNum,htmlSource)

    startTag1 = '<font style="color:red;">'
    startTag2 = '<font style="color:red">'
    startTag = 'Grade Level: '
    endTag='</font>'
    endTag2='</font> == $0'
    pattern = r'(\d+(?:\.\d+)?)'
    numberOfWordsTag = 'Total # of words:<b> '
    numberOfUniqueWordsTag = 'Total # of unique words:<b> '


   
    try:
        m = re.findall(startTag1+pattern+endTag,htmlSource)
        n = re.findall(startTag2+pattern+endTag,htmlSource)
        o = re.findall(startTag + pattern, htmlSource)
        flesch = m[0]
        gunning = m[1]
        kincaid = m[2]
        colemon = n[0]
        smog = n[1]
        ari = n[2]
        linsear = n[3]
        consensus=o[0]
        numberOfWords = re.findall(numberOfWordsTag+pattern, htmlSource)
        numberOfUniqueWords = re.findall(numberOfUniqueWordsTag+pattern, htmlSource)
        nOw = int(numberOfWords[0])
        nOuw = int(numberOfUniqueWords[0])
        nOrw = nOw-nOuw
        
        percentageUniqueWords = math.ceil(nOuw*100/nOw)
        print(percentageUniqueWords)
        

        print("Flesch Reading Ease:"+str(flesch))
        print("Gunning Fog:"+str(gunning))
        print("Flesch-Kincaid Grade:"+str(kincaid))
        print("Coleman Liau Index:"+str(colemon))
        print("Smog Index:"+str(smog))
        print("Automatic Readability Index:"+str(ari))
        print("Linsear Write Formula:"+str(linsear))
        print("Consensus based upon all above tests:"+str(consensus))
        print(numberOfWords[0])
        print(numberOfUniqueWords[0])
       
    except:
        print(fileNum)
        
    


for x in range(252,253):
    readFile(x)
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 11:11:49 2018

@author: sabab05
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:31:18 2018

@author: sabab05
""" 

import re
import numpy 

steps=list()

def fileRead(num):    
    res = 0
    N=0
    global steps
    out_path = r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.1 Analysis (using 750 symmetrical data)\5. Feature (Wordcount)\Output\Output.txt'''    
    fileOut =open(out_path,"a",encoding="utf-8")
    #with open(r'''D:\CLoud\Academic\Research\___\Feature Set\Text Data\Tutplus Dataset (A-190, B-190) Steps Only\A ('''+repr(num)+''').txt''','r',encoding="utf8",errors='ignore') as f: 
    with open(r'''D:\CLoud\Academic\Research\___\Analysis (Photoshop)\4.2 Analysis Visualization - pyLDAvis (Using 750 symmetrical data)\1. Feature (Words)\Topic Model\Input\S(''' + repr(num) + ''').txt''','r',encoding="utf8",errors='ignore') as f:            
        lines=f.read()
        lines=lines.split()
        l = len(lines)
    steps.extend([l])
    fileOut.write(repr(l)+"\n")
    print(repr(l))

for x in range(252,253):    
    fileRead(x)
    

print(steps)

results = steps
print(numpy.mean(results))
print(numpy.median(results))
print(numpy.std(results))
#print(numpy.var(results))
#print(len(results))
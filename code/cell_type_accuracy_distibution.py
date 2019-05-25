# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 09:17:12 2018

@author: Marcus Östling, Joakim Lilja

Cell type accuracy distribution

Draw the histograms for each classifier.
Data from ../data/after1000runs.json
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
# Using sypder the file '../data/after1000runs.json' was
# copied into data as code.
with open('../data/after1000runs.json', 'r') as myfile:
    data2=myfile.read().replace('\n', '')

data = eval(data2)
'''

# Print the values
for k,v in data.items():
    print()
    print()
    print(k)
    for k2,v2 in v.items():    
        fir = v2[0]
        sec = v2[1]
        print(k2, "{0:.4f}".format((fir/(fir+sec))*100)+"%")


# Draw the histograms
fig, axes = plt.subplots(nrows=5, ncols=2)
fig.tight_layout()
subplotindex = 1

for k,v in data.items():
    names = list(range(len(v.items())))
    values = []
    for k2,v2 in v.items():
        values.append((v2[0]/(v2[0]+v2[1]))*100)
    plt.xticks(rotation=0)
    plt.ylim(0,105)
    plt.subplot(5, 2, subplotindex)
    plt.bar(names,values, align='center')
    plt.title(k)
    plt.ylabel('%')
    subplotindex += 1
    
plt.show()

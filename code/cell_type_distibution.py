# -*- coding: utf-8 -*-
"""
@author: Marcus Östling, Joakim Lilja

Cell type accuracy distribution

Draw the histograms for each cell type.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

dataset = pd.read_csv('../data/mouse.csv')
cols = dataset.shape[1]
cell_types = dataset.iloc[:, cols-1].values.tolist()

number_of_cells = dict(Counter(cell_types))

names = []
values = []

for k,v in number_of_cells.items():
    names.append(k)
    values.append(v)
    print(k, str(v))

plt.rc('font', size=24)
plt.xticks(rotation=-90)
plt.gcf().subplots_adjust(bottom=0.50)
plt.bar(names, values, align='center')
plt.title("Cell type distribution")
plt.ylabel('Number of cells')
plt.show()
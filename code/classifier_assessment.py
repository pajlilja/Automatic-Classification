# -*- coding: utf-8 -*-
"""
@author: Marcus Östling, Joakim Lilja

Assess the classifiers
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cross_validation import train_test_split

# Importing the dataset
dataset = pd.read_csv('../data/mouse.csv')
cols = dataset.shape[1]
X = dataset.iloc[:, 1:cols-1].values
y = dataset.iloc[:, cols-1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Fitting classifier to the Training set
# Create your classifier here
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

classifiersName = ['Multi-layer perceptor', 'DT CART', 'LDA', 'SVC (Linear kernal)',
                   'SVC (RBF)', 'SVC (Polynomial)', 'Random Forest', 'k-Nearest Neigbors', 
                   'Naive Bayes (Gaussian)', 'Logistic Regression']

classifiers = [MLPClassifier(solver='lbfgs', alpha=1e-5, 
                             hidden_layer_sizes=(100, 100), random_state=1),
                DecisionTreeClassifier(criterion = 'entropy'),
                LinearDiscriminantAnalysis(n_components=50),
                svm.SVC(kernel='linear', C=1.0, tol=1e-3),
                svm.SVC(kernel='rbf', C=100.0, tol=1e-3),
                svm.SVC(kernel='poly', C=100.0, tol=1e-3, degree=2, coef0=100),
                RandomForestClassifier(n_estimators=100),
                KNeighborsClassifier(n_neighbors=5),
                GaussianNB(),
                LogisticRegression(C=100.0)]

# Scores
from sklearn.metrics import accuracy_score
scores = [[] for i in range(len(classifiers))]

scoreRatios = dict()
for i in range(len(classifiersName)):
    scoreRatios[classifiersName[i]] = dict()
    for j in range(len(y)):
        scoreRatios[classifiersName[i]][labelencoder_y.inverse_transform(y[j])] = [0 , 0]

'''
if(True): # TEST JUST ONE
    i = 5
    solo_scores = []
    for it in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)
        classifiers[i].fit(X_train, y_train)
        solo_scores.append(accuracy_score(y_test,classifiers[i].predict(X_test)))
    for s in solo_scores:
        print(s)
    print("mean: ", np.mean(solo_scores))
    import sys
    sys.exit()
  '''     

for it in range(10): # amount of runs for each classifiers.
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)#, random_state = 0)
    for i in range(len(classifiers)):
        classifiers[i].fit(X_train, y_train)
        y_predictions = classifiers[i].predict(X_test)
        scores[i].append(accuracy_score(y_test, y_predictions))
        for j in range(len(y_test)):
            if y_test[j] == y_predictions[j]:
                scoreRatios[classifiersName[i]][labelencoder_y.inverse_transform(y_test[j])][0] += 1
            else:
                scoreRatios[classifiersName[i]][labelencoder_y.inverse_transform(y_test[j])][1] += 1
        

print()
for i in range(len(classifiers)):
    print(classifiersName[i], (float(scores[i][0])*100),'%')

# The box plot
# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

bp = ax.boxplot(scores)

ax.set_xticklabels(classifiersName, rotation=330)

plt.grid()
plt.legend()
plt.title("Model Comparison")
plt.xlabel("Model")
plt.ylabel("Model accuracy (%)")
plt.show()

# The cell type hit ratio diagrams
fig, axes = plt.subplots(nrows=5, ncols=2)
fig.tight_layout() # Or equivalently,  "plt.tight_layout()"

for i in range(len(classifiersName)):
    names = []
    count = 0
    values = []
    for k,v in scoreRatios[classifiersName[i]].items():
        names.append(count)
        count += 1
        if v[0]+v[1] == 0:
            v[1] = 1
        values.append((v[0]/(v[0]+v[1]))*100)
    plt.xticks(rotation=0)
    plt.subplot(5, 2, i+1)
    plt.bar(names, values, align='center')
    plt.title(classifiersName[i])
    plt.ylabel('%')
    
plt.show()

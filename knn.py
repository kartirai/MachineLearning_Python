# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:10:16 2020

@author: raiajay
"""
#breastcancer dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

breast_cancer = load_breast_cancer()
x = pd.DataFrame(breast_cancer.data,columns = breast_cancer.feature_names)
x= x[['mean area','mean compactness']]
y = pd.Categorical.from_codes(breast_cancer.target,breast_cancer.target_names)
y = pd.get_dummies(y,drop_first=True)

X_train,X_test,Y_train,Y_test = train_test_split(x,y, random_state=1)

knn = KNeighborsClassifier(n_neighbors = 5,metric='euclidean')
knn.fit(X_train,Y_train)

Y_pred = knn.predict(X_test)

sns.scatterplot(x='mean area',y = 'mean compactness',hue='benign',data=X_test.join(Y_test,how='outer'))
z=Y_pred
z

plt.scatter(X_test['mean area'],X_test['mean compactness'],z,cmap='coolwarm',alpha = 0.7)
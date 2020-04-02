# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 22:39:30 2020

@author: raiajay
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('C:/Users/raiajay/Desktop/kartikey/titanic.csv')
df.head()

df = df[['Pclass','Sex','Age','Siblings/Spouses Aboard','Parents/Children Aboard','Fare','Survived']]
df
df['Sex'] = df['Sex'].map({'male':0,'female':1})
df = df.dropna()
X = df.drop('Survived',axis=1)
y = df['Survived']
X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=1)
model = tree.DecisionTreeClassifier()
model
model.fit(X_train,y_train)
y_predict = model.predict(X_test)

accuracy_score(y_test, y_predict)
pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted Not Survival', 'Predicted Survival'],
    index=['True Not Survival', 'True Survival']
)
    
from sklearn.tree import export_graphviz
export_graphviz(model,out_file = 'tree.dot',feature_names = X.columns)
from subprocess import call
call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])
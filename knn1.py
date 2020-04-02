# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 00:40:06 2020

@author: raiajay
"""


#with iris dataset
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score


df = pd.read_csv('C:/Users/raiajay/Desktop/kartikey/iris.csv')
df.head(5)
df.shape
df.describe

df.groupby('Species').size()

features_columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = df[features_columns].values
Y = df['Species'].values

le = LabelEncoder()
Y = le.fit_transform(Y)
Y
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from pandas.plotting import parallel_coordinates
plt.figure(figsize=(15,6))
parallel_coordinates(df.drop('Id',axis=1), "Species")
plt.title('Parallel Co-ordinates plot', fontsize=20,fontweight='bold')
plt.xlabel('Features',fontsize=15)
plt.ylabel('Features values',fontsize=15)
plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
plt.show()

classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred)
cm
          
accuracy = accuracy_score(Y_test, Y_pred)*100
print('Accuracy of model = ',str(round(accuracy,2))+'%')












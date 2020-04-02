# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:30:07 2020

@author: raiajay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('C:/Users/raiajay/Desktop/kartikey/winequality.csv')
df.describe()

df.isnull().any()

X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
Y = df['quality'].values

plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(df['quality'])

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=0)

regression = LinearRegression()
regression.fit(X_train,Y_train)

coeff_df = pd.DataFrame(regression.coef_, columns=['Coefficient'])  
coeff_df

Y_pred = regression.predict(X_test)

df1 = pd.DataFrame({'Actual : ':Y_test,'Predicted : ':Y_pred})
df1

df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major',linestyle='-',linewidth='0.5',color='green')
plt.grid(which='minor',linestyle=':',linewidth='0.5',color='black')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:08:05 2020

@author: raiajay
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('C:/Users/raiajay/Desktop/kartikey/Weather.csv')
df.describe()
df.shape

df.plot(x='MinTemp',y='MaxTemp',style='o')
plt.title('MinTemp vs Maxtemp')
plt.xlabel('Min temperature')
plt.ylabel('Max temperature')
plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(df['MaxTemp'])

X = df['MinTemp'].values.reshape(-1,1)
Y = df['MaxTemp'].values.reshape(-1,1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=0)

regression = LinearRegression()
regression.fit(X_train,Y_train)
print(regression.intercept_)
print(regression.coef_)

Y_pred = regression.predict(X_test)
Y_pred

df1 = pd.DataFrame({'Actual':Y_test.flatten(),'Predicted':Y_pred.flatten()})
df1

df2 = df1.head(25)
df2.plot(kind='bar',figsize=(15,10))
plt.grid(which='major',linestyle='-',linewidth='0.5',color='green')
plt.grid(which='minor',linestyle=':',linewidth = '0.5',color='black')
plt.show()

plt.scatter(X_test,Y_test,color='gray')
plt.plot(X_test,Y_pred,color='red',linewidth=2)
plt.show()

print('Mean Absolute Error : ',metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error : ',metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error : ',metrics.mean_squared_error(Y_test, Y_pred))

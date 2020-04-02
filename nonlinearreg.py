# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 00:07:23 2020

@author: raiajay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('C:/Users/raiajay/Desktop/kartikey/temperature.csv')
df

X = df.iloc[:,1:2].values
X
Y = df.iloc[:,2].values
Y

regression = LinearRegression()
regression.fit(X,Y)

poly = PolynomialFeatures()
X_poly = poly.fit_transform(X)
poly.fit(X_poly,X)
regression2 = LinearRegression()
regression2.fit(X_poly,Y)

plt.scatter(X,Y,color='blue')
plt.title('Linear Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')

plt.scatter(X,Y,color='blue')
plt.plot(X,regression2.predict(poly.fit_transform(X)),color='red')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')

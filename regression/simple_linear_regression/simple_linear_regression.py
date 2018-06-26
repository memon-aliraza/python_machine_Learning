# Simple Linear Regression Example

# -*- coding: utf-8 -*-
"""
Created on Fri May 25 14:15:04 2018

@author: Ali R. Memon
@file:   simple_linear_regression.py
@date:   25.05.2018
"""

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing  the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

# No need to apply Feature Scaling in Linear Regression. Algorithms take care of this.

# Filliting model to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Making predictions and compare predictions with y_test dataset
y_predictor = regressor.predict(X_test)
X_predictor = regressor.predict(X_train)

# Visualizing training set results
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train, X_predictor, color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing test set results
plt.scatter(X_test,y_test, color='red')
plt.plot(X_train, X_predictor, color =  'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

 











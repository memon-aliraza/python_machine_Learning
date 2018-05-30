# Multiple Linear Regression Example

# -*- coding: utf-8 -*-
"""
Created on Tues May 29 12:55:04 2018

@author: Ali R. Memon
@file:   multiple_linear_regression.py
@date:   29.05.2018
"""

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing  the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
 
# Encoding the Categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding dummxy variable trap (Linear Regression library automatically take care)

X = X[:, 1:] 

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# Fitting MLR to Dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X,axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X = np.append(arr = np.ones((50, 1)).astype(int), values = X,axis=1)
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X = np.append(arr = np.ones((50, 1)).astype(int), values = X,axis=1)
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X = np.append(arr = np.ones((50, 1)).astype(int), values = X,axis=1)
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X = np.append(arr = np.ones((50, 1)).astype(int), values = X,axis=1)
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


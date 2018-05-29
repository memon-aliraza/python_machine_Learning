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
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
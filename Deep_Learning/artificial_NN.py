# Artificial Neural Network

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:30:16 2018

@author: Ali R. Memon
@file:   artificial_NN.py
@date:   25.05.2018
"""

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing  the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_land = LabelEncoder()
X[:, 1] = labelencoder_X_land.fit_transform(X[:, 1])

labelencoder_X_sex = LabelEncoder()
X[:, 2] = labelencoder_X_sex.fit_transform(X[:, 2])


# Dummy variable for country independent variable
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:13]


# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)


# We need to apply Feature Scaling in Artificial Neural Networks
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# part 2 - Building ANN! 

#Importing ANN libraries
import keras 
from keras.models import Sequential # initialize neural network
from keras.layers import Dense # used to create layers in ANN

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# output_dim = number of hidden layer, input_dim = number of input layers (independent variables)
# How many nodes are we going to add in hidden layer (Odd! In case the data is linearly separable, no need to add hidden layer even not NN) 
# Tip: Choose AVG number of nodes in input layer and the number of nodes in output layer! 11 + 1 = 6 nodes in hidden layer!
# Even k-fold cross validation can be used. 
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim = 11 )) 

# Adding second hidden layer. However, we do not need to add more hidden layers for our dataset.
# But due to deep learning we should have more than 1 hidden layer and also we need to see how to add another.  
#classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))  

# Adding output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid')) 

# Compiling the ANN (Applying Stochastic Gradient Descent)
# Optimizer specifies the SGD. There are several SGD methods. 
# Loss: The SGD is based on loss function. We need to optimize for optimal weights. If dependent variable is binary then binary_ if morethan 2 then category_.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',  metrics = ['accuracy'] )

# Fitting ANN to the Training Set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


# Part 3 - Making Prediction and Model Evaluation

# Predicting test set results.
y_pred = classifier.predict(X_test) # y_pred will return the probabilities of the customers who can leave the bank.
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


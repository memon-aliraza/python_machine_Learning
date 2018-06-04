# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 12:30:19 2018

@author: arm
"""

from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np



seed = 0
np.random.seed(seed)

dataframe = pd.read_csv('BBC.csv')

array = dataframe.values

X = array[:,0:11]
y = array[:, 11]

model = Sequential()
model.add(Dense(11, input_dim=11, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, nb_epoch=50, batch_size=10)

score = model.evaluate(X, y)

print ("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

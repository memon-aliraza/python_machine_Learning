#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 01:30:16 2018

@author: Ali R. Memon
@file:   convolutional_NN.py
"""
# Part - I: Building the CNN
    
# Importing the Keras libraries and packages

from keras.models import Sequential # Initialise Neural Network 
from keras.layers import Convolution2D # For adding convolutional layers (2D for images)
from keras.layers import MaxPooling2D # For adding pooling layer 
from keras.layers import Flatten # Convert all pooled feature maps into feature vector
from keras.layers import Dense # To add fully connected layer

# Initialize Convolutional Neural Network
classifier = Sequential()

# Step I - Convolution 
"""
Image converted into pixel values. Now we have 0 and 1 matrix. Convolutional step applies 
several feature detector on input image by sliding featue detector on image matrix. 
The output will be feature map consisting of multiple numbers where higher number implies 
Feature Detector could detect a specific feature in the input image.  
"""
classifier.add(Convolution2D(32,3,3, input_shape=(64, 64, 3), activation='relu')) # 32 feature detectors with 3X3 dimentions. 

# Step 2 - Pooling 
"""
Reducing the size of feature map by sliding 2x2 matrix over the feature map.
Causes reduction of input nodes, computation time and complexity.  
"""
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
""" 
Taking all pool maps and put them into one single vector. This will be the input layer to ANN.
"""
classifier.add(Flatten())

# Step 4 - Full Connection 
"""
Creating a classic ANN with fully connected layers. 
"""
classifier.add(Dense(output_dim=128, activation='relu'))

# Output layer
classifier.add(Dense(output_dim=1, activation='sigmoid')) # If not binary output we will use softmax

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part - II: Fitting the CNN to the dataset 
"""
Image augmentation is a technique that allows us to enrich the dataset without adding more 
images. Therefore, it alows us to get good performance results with low or no overfitting 
over the small amount of images.  
"""
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_size',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
    
test_set = test_datagen.flow_from_directory('data/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    nb_val_samples=800)



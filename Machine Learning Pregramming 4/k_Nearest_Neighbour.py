
# coding: utf-8

# In[32]:

from __future__ import division
import pandas as pd
import numpy as np
import copy
from operator import itemgetter
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')


# In[33]:

"""scales the data like med=2, high=3, vhigh=4"""
def transform_data(df):
    unique_values = [x for x in df.drop_duplicates()]
    if all(item.isdigit() for item in unique_values):
        unique_values = [int(x) for x in unique_values]
    replacements = [x for x in xrange(1,len(unique_values)+1)]
    df = df.replace(to_replace=unique_values, value=replacements)
    return df


# In[34]:

"""scale_data takes filename as input, reads the 
dataframe from it and scales the values in dataframe 
to be able for processing in K-nearest neighbours implementation"""
def scale_data(filename):
    df = pd.read_csv(filename,header=None,
                     names=["buying","maint","doors","persons","lug_boot","safety","Actual_Classification"])
    attributes_list = [x for x in df.columns if x <> "Actual_Classification"]
    k_df = df.copy(deep=True)
    for attribute in attributes_list:
        k_df[attribute] = transform_data(k_df[attribute])
    return k_df


# In[35]:

"""split_data takes filename as input, reads the dataframe from it
and splits the dataframe in test data and training data"""
def split_data(df):
    training_data = df.sample(frac=0.66) #Samples out 2/3rd of the data on random as training data 
    test_data = df.ix[~df.index.isin(training_data.index)] #The remaining 1/3rd of data as test data
    return training_data, test_data


# In[36]:

"""Calculates Eucladian Distance bw two instances"""
def eucladian_distance(A, B):
    first = [y for x,y in enumerate(A) if x not in [0,len(A)-2,len(A)-1]]
    second = [y for x,y in enumerate(B) if x not in [0,len(B)-1]]
    dist= np.linalg.norm(np.array(first)-np.array(second))
    return dist


# In[37]:

"""classify_testdata takes training data and test data as input
and classify test data on the basis of K-Nearest Neighbours Algorithm
and returns test data frame with and extra column of Learned Classifications
which are the result of application of naive bayes algorithm"""
def classify_testdata(training_data, test_data, k_num):
    #append Learned_Classification column to test data
    test_data["Learned_Classification"] = np.nan
    #find target values from Actual_Classification column
    #iterate over each row of test data
    for rows in test_data.itertuples():
        temp_list = []
        for tupl in training_data.itertuples():
            distance = eucladian_distance(rows, tupl)
            temp_list.append([tupl[0],distance])
        a = sorted(temp_list, key=itemgetter(1))
        a = a[:k_num]
        temp_a = [i for i,j in a]
        neighrest_neighbours = training_data.ix[temp_a]
        classifier = pd.value_counts(neighrest_neighbours["Actual_Classification"]).idxmax()
        test_data.ix[rows[0],"Learned_Classification"] = classifier
    return test_data


# In[38]:

"""main function to call. Inputs filename. 
Calculates error rate for each individual sample.
Calculates mean error rate for 100 samples.
Calculates Confusion Matrix for one sample.
Plots a Stacked Bar Graph illustrating confusion matrix for one samlple."""
def main(filename):
    errorrate_array = []
    mean_errorrate = 0
    plt.figure();
    #how many k-neighbours
    k_num = int(raw_input("Enter value for k: "))
    text_file = open("Output.txt", "w")
    for i in xrange(100):
        #split_data to split data in training and test
        training_data, test_data = split_data(scale_data(filename))
        #call classify_testdata to classify instances in test data
        test_data = classify_testdata(training_data, test_data, k_num)
        #calculate error rate for each individual sample
        errorrate = len(test_data[test_data["Learned_Classification"]<>test_data["Actual_Classification"]])
        if i==0:
            test_data.to_csv("output.csv", sep=',', encoding='utf-8')
            #measure confusion matrix for one sample
            df_confusion = pd.crosstab(test_data["Actual_Classification"], test_data["Learned_Classification"])
            #plots confusion matrix
            df_confusion.plot(kind='bar', stacked=True)
            text_file.write("Confusion Matrix:\n")
            text_file.write(str(df_confusion)+"\n\n")
            plt.savefig('Confusion Matrix Plot.png')
            temp = "Error Rate for Sample: "+str(errorrate)+"\n\n"
            text_file.write(str(temp))
            print "Confusion Matrix Plotted. See File 'Confusion Matrix Plot.png'"
            print "Calculating Mean Error Rate over 100 Samples..."
        mean_errorrate += errorrate
        errorrate_array.append(errorrate)
    #calculate mean error rate over 100 samples
    mean_errorrate = mean_errorrate/100
    temp = "Mean Error Rate over 100 samples: "+str(mean_errorrate)
    text_file.write(str(temp))
    text_file.close()
    print "See Confusion Matrix and other results in Output.txt"


# In[40]:

main("car.data")


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




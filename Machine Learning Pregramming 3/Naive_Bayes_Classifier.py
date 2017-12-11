
# coding: utf-8

# In[36]:

from __future__ import division
import pandas as pd
import numpy as np
import copy
from operator import itemgetter
import matplotlib.pyplot as plt


# In[37]:

"""split_data takes filename as input, reads the dataframe from it
and splits the dataframe in test data and training data"""
def split_data(filename):
    df = pd.read_csv(filename,header=None,
                     names=["buying","maint","doors","persons","lug_boot","safety","Actual_Classification"])
    attributes_list = [x for x in df.columns if x <> "Actual_Classification"]
    attributes_list = ['index'] + attributes_list
    training_data = df.sample(frac=0.66) #Samples out 2/3rd of the data on random as training data 
    test_data = df.ix[~df.index.isin(training_data.index)] #The remaining 1/3rd of data as test data
    return training_data, test_data, attributes_list


# In[38]:

"""classify_testdata takes training data and test data as input
and classify test data on the basis of naive bayes algorithm
and returns test data frame with and extra column of Learned Classifications
which are the result of application of naive bayes algorithm"""
def classify_testdata(training_data, test_data, attributes_list):
    #append Learned_Classification column to test data
    test_data["Learned_Classification"] = np.nan 
    length = len(training_data)
    #find target values from Actual_Classification column
    target_values = training_data["Actual_Classification"].drop_duplicates()
    #find prior probabilities for classifiers by dividing the frequency
    #of their occurance with the total length of data
    target_values_probability = [len(training_data[training_data["Actual_Classification"]==x])/length for x in target_values] 
    target_tuple = zip(target_values, target_values_probability)
    #iterate over each row of test data
    for rows in test_data.itertuples():
        #make deep copy of list to perserve it 
        temp_target_tuple = copy.deepcopy(target_tuple)
        temp = zip(attributes_list, rows)
        #for each classifier find the probability given training data
        for x,y in enumerate(temp_target_tuple):
            classifier, classifier_probability = y
            prod = 1
            required_df = training_data[training_data["Actual_Classification"]==classifier]
            den_length = len(required_df)
            #iterate over attributes in given instance to find their probability given classifier
            for column, attribute in temp:
                if column<>'index':
                    num_length = len(required_df[required_df[column]==attribute])
                    req_probability = num_length/den_length
                    prod = prod * req_probability
            prod = prod * classifier_probability
            temp_target_tuple[x] = temp_target_tuple[x] + (prod,)  
        #find classifier with max probability given training data
        a,b,c = max(temp_target_tuple,key=itemgetter(2))
        test_data.ix[rows[0],"Learned_Classification"] = a
    return test_data


# In[71]:

"""main function to call. Inputs filename. 
Calculates error rate for each individual sample.
Calculates mean error rate for 100 samples.
Calculates Confusion Matrix for one sample.
Plots a Stacked Bar Graph illustrating confusion matrix for one samlple."""
def main(filename):
    errorrate_array = []
    mean_errorrate = 0
    plt.figure();
    text_file = open("Output.txt", "w")
    for i in xrange(1):
        #call split_data to split data in training and test
        training_data, test_data, attributes_list = split_data(filename)
        #call classify_testdata to classify instances in test data based on
        #classifiers probability given training data
        test_data = classify_testdata(training_data, test_data, attributes_list)
        #calculate error rate for each individual sample
        errorrate = len(test_data[test_data["Learned_Classification"]<>test_data["Actual_Classification"]])
        if i==0:
            bar_l = [i+1 for i in range(len(test_data["Learned_Classification"].drop_duplicates()))]
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


# In[72]:

main("car.data")


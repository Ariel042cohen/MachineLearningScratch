# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 19:02:54 2021

@author: ariel
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = np.exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

data = pd.read_csv('diabetes.csv', delimiter=',')
print(data.head(10))
print(data.shape)

result = data['Outcome'].value_counts()/data['Outcome'].count()
print (result)
priorDataPositive = result[1]
priorDataNegative = result[0]

statisticPositive = data[data['Outcome'] == 1].describe()
statisticNegative = data[data['Outcome'] == 0].describe()
print(statisticPositive.describe())
print(statisticNegative.describe())

y = data['Outcome'].to_numpy().reshape(768,1)
data = data.drop(columns=['Outcome'])
X = data.to_numpy()

print(y.shape)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

data_train = np.column_stack( (X_train,y_train) )
data_test = np.column_stack( (X_test,y_test))

class NaiveBayseClassifiction:
    def __init__(self, dataset, label_col_name):
        self.dataset = dataset
        self.label_col_name = label_col_name
        
    def calculate_probability(x, mean, stdev):
        	exponent = np.exp(-((x-mean)**2 / (2 * stdev**2 )))
        	return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent
        
    def predict(self, row):
        pos = priorDataPositive
        neg = priorDataNegative
        for index in range(row.shape[0] - 1):
            value = row[index]
            meanPositive = statisticPositive[data.columns[index]][1]
            stdPositive = statisticPositive[data.columns[index]][2]
            meanNegative = statisticNegative[data.columns[index]][1]
            stdNegative = statisticNegative[data.columns[index]][2]
            pos = pos * calculate_probability(value, meanPositive, stdPositive)
            neg = neg * calculate_probability(value, meanNegative, stdNegative)
        
        if pos > neg:
            return 1 
        else:
            return 0
        
    def train(self):
        data = self.dataset
        result = data[self.label_col_name].value_counts()/data[self.label_col_name].count()
        priorDataPositive = result[1]
        priorDataNegative = result[0]
        statisticPositive = data[data[self.label_col_name] == 1].describe()
        statisticNegative = data[data[self.label_col_name] == 0].describe()
        
        y = data[self.label_col_name].to_numpy().reshape(768,1)
        data = data.drop(columns=[self.label_col_name])
        X = data.to_numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        data_train = np.column_stack( (X_train,y_train) )
        data_test = np.column_stack( (X_test,y_test))
        
        good = 0
        bad = 0
        for row in data_test:
            output = self.predict(row)
            if row[8] == output:
                good = good + 1
            else:
                bad = bad + 1
                
        print ("accuracy : " + str(round(good / (good + bad) * 100)))
        
    
        

def predict(row):
    pos = priorDataPositive
    neg = priorDataNegative
    for index in range(row.shape[0] - 1):
        value = row[index]
        meanPositive = statisticPositive[data.columns[index]][1]
        stdPositive = statisticPositive[data.columns[index]][2]
        meanNegative = statisticNegative[data.columns[index]][1]
        stdNegative = statisticNegative[data.columns[index]][2]
        pos = pos * calculate_probability(value, meanPositive, stdPositive)
        neg = neg * calculate_probability(value, meanNegative, stdNegative)
    
    if pos > neg:
        return 1 
    else:
        return 0

good = 0
bad = 0
for row in data_test:
    output = predict(row)
    if row[8] == output:
        good = good + 1
    else:
        bad = bad + 1


#print ("accuracy : " + str(round(good / (good + bad) * 100)))

data = pd.read_csv('diabetes.csv', delimiter=',')
A = NaiveBayseClassifiction(data, 'Outcome')
A.train()




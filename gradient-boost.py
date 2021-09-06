# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:30:50 2021

@author: ariel
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score

learing_rate = 0.1

def read_and_organize_data(file_name):
    
    dt = pd.read_csv(file_name)
    
    dt = pd.get_dummies(dt, drop_first=True)
    
    y_vals = dt['Weight'].values
    x_train, x_test, y_train, y_test = train_test_split(dt.drop('Weight', 1).values,y_vals , test_size = .25, random_state=10) 
    return x_train, x_test, y_train, y_test

def boosting_classifier(x_train,y_train,max_num_iterations=100,base_model = DecisionTreeRegressor,**params):
    x_train_copy = x_train.copy()
    y_train_copy = y_train.copy()

    estimator_list = []
    average = np.average(y_train)
    y_train -= average
    
    for i in range(max_num_iterations):
        
        # estimator = DecisionTreeClassifier(max_depth=1)
        # estimator = LogisticRegression()
        estimator = base_model(**params)
        
        estimator.fit(x_train_copy, y_train_copy)
        y_predict = estimator.predict(x_train_copy)
        
        #3. terror
        y_train_copy = (y_train_copy - y_predict)
        
        estimator_list.append(estimator)
          
    return estimator_list, average

def predict_by_boosting(estimator_list,x_test,y_test, average):

    #create predictions for test data
    y_test_pred = np.ones(y_test.shape[0])
    y_test_pred *= average
    
    for model in estimator_list:
        y_test_pred += learing_rate * model.predict(x_test)
        
    return y_test_pred

if __name__ == '__main__':
    
    file_name = 'Fish.csv'
    x_train, x_test, y_train, y_test = read_and_organize_data(file_name)
    
    estimator_list, average = boosting_classifier(x_train,y_train,10,DecisionTreeRegressor,max_leaf_nodes=8)
    y_test_pred = predict_by_boosting(estimator_list,x_test,y_test, average)
    print('Primrose adaboost accuracy: ',y_test_pred - y_test)

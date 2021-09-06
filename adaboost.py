# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:16:32 2021

@author: ariel
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def read_and_organize_data(file_name):
    
    dt = pd.read_csv(file_name)
    
    #all these features represent categorical features
    dt['cp'] = dt['cp'].astype(str)
    dt['fbs'] = dt['fbs'].astype(str)
    dt['restecg'] = dt['restecg'].astype(str)
    dt['exang'] = dt['exang'].astype(str)
    dt['slope'] = dt['slope'].astype(str)
    dt['thal'] = dt['thal'].astype(str)
    
    dt = pd.get_dummies(dt, drop_first=True)
    
    y_vals = dt['target'].values
    y_vals = 2*y_vals-1
    x_train, x_test, y_train, y_test = train_test_split(dt.drop('target', 1).values,y_vals , test_size = .25, random_state=10) 
    return x_train, x_test, y_train, y_test

def boosting_classifier(x_train,y_train,max_num_iterations=100,base_model = DecisionTreeClassifier,**params):

    num_samples = len(x_train)

    estimator_list, significance_list = [], []
#    2. initialize w
    sample_weight = np.ones(num_samples) / num_samples
    
    for i in range(max_num_iterations):
        
        # estimator = DecisionTreeClassifier(max_depth=1)
        # estimator = LogisticRegression()
        estimator = base_model(**params)
        
        estimator.fit(x_train, y_train, sample_weight=sample_weight)
        y_predict = estimator.predict(x_train)
        
        #3. terror
        incorrect = (y_predict != y_train)
        
        #4. total error
        estimator_error =  np.average(incorrect, weights=sample_weight, axis=0)
        
        #5. significance
        significance =  np.log((1. - estimator_error) / estimator_error)
        
        #6. update weights
        sample_weight *= np.exp(significance * incorrect)
        sample_weight /= sample_weight.sum()
        
        estimator_list.append(estimator)
        significance_list.append(significance.copy())
          
    return estimator_list,significance_list

def predict_by_boosting(estimator_list,estimator_weight_list,x_test,y_test):

    #create predictions for test data
    y_test_pred_list = []
    for model in estimator_list:
        y_pred = model.predict(x_test)
        y_test_pred_list.append(y_pred)
        
    #organizing arrays for matrix multiplication    
    y_test_pred_list = np.asarray(y_test_pred_list)
    estimator_weight_list = np.array(estimator_weight_list).reshape( len(estimator_weight_list),1)
    
    preds = np.sign(y_test_pred_list.T@estimator_weight_list)
    accuracy = accuracy_score(preds,y_test)
    return preds,accuracy

if __name__ == '__main__':
    
    file_name = 'heart.csv'
    x_train, x_test, y_train, y_test = read_and_organize_data(file_name)
    
    my_dct = DecisionTreeClassifier()
    my_dct.fit(x_train,y_train)
    dct_score = my_dct.score(x_test,y_test)
    print('decision tree accuracy: ',dct_score)
             
    my_adaboost = AdaBoostClassifier()
    my_adaboost.fit(x_train,y_train)
    adaboost_score = my_adaboost.score(x_test,y_test)
    print('scikit learn adaboost accuracy: ',adaboost_score)
    
    estimator_list,estimator_weight_list = boosting_classifier(x_train,y_train,100,DecisionTreeClassifier,max_depth=1)
    preds,adaboost_score = predict_by_boosting(estimator_list,estimator_weight_list,x_test,y_test)
    print('Primrose adaboost accuracy: ',adaboost_score)

    estimator_list,estimator_weight_list = boosting_classifier(x_train,y_train,10,SVC,kernel='linear')
    preds,adaboost_score_with_svm = predict_by_boosting(estimator_list,estimator_weight_list,x_test,y_test)
    print('Primrose adaboost accuracy with svm: ',adaboost_score_with_svm)

    estimator_list,estimator_weight_list = boosting_classifier(x_train,y_train,10,LogisticRegression,C=1)
    preds,adaboost_score_with_logistic = predict_by_boosting(estimator_list,estimator_weight_list,x_test,y_test)
    print('Primrose adaboost accuracy with logistic: ',adaboost_score_with_logistic)

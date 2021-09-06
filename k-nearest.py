# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:14:02 2021

@author: ariel
"""

import pandas as pd
import numpy as np
from math import sqrt

def data_seperate(X):
    X_1 = X[X[:,2] == 1]
    X_0 = X[X[:,2] == 0]
    X_1_SPLIT = np.array_split(X_1, 2)
    X_0_SPLIT = np.array_split(X_0, 2)
    X_TRAIN = np.concatenate((X_1_SPLIT[0], X_0_SPLIT[0]))
    X_TEST = np.concatenate((X_1_SPLIT[1], X_0_SPLIT[1]))
    
    return (X_TRAIN, X_TEST)

def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1) -1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
 
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
 
# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

"""
1
"""
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', delimiter=',')
iris['filter'] = iris['Iris-setosa'] == 'Iris-setosa'
X = iris.iloc[:,0:2].values
Y = iris['filter'].astype(int).to_numpy()
Y = Y.reshape((Y.shape[0], 1))
X = np.column_stack((X,Y))

result = data_seperate(X)
X_TEST = result[0]
X_TRAIN = result[1]

print (X)

"""
2
"""

dist = euclidean_distance(X[0], X[1])


"""
3
"""
num_neighbors = 3
neighbors = get_neighbors(X, X[0], num_neighbors)
print (neighbors)

"""
4 + 5
"""
count_predictd_well = 0
all_count = X_TEST.shape[0]

for i in range(0, all_count):
    prediction = predict_classification(X_TRAIN, X_TEST[i], num_neighbors)
    if X_TEST[i][-1] == prediction:
        count_predictd_well += 1

print ('accuracy is : ' + str(count_predictd_well / all_count))

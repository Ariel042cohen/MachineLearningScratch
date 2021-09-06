# -*- coding: utf-8 -*-
"""
Created on Tue May  4 17:04:58 2021

@author: ariel
"""

import os, sys
import shutil
from imageio import imread,imwrite
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

'''
for file in os.listdir('E:/Primose/faces94'):
    for file2 in os.listdir('E:/Primose/faces94/' + file):
        for file3 in os.listdir('E:/Primose/faces94/' + file + '/' + file2):
            shutil.copyfile('E:/Primose/faces94/' + file + '/' + file2 + '/' + file3 , 'E:/Primose/faces94/data/'+ file3)
            break;
'''
faces = np.array([])
for file in os.listdir('E:/Primose/faces94/data'):
    face = imread('faces94/data/' + file)
    face_reshape = face.reshape(1,108000)
    
    if (faces.shape[0] == 0):
        faces = face_reshape
    else:
        faces = np.vstack([faces,face_reshape])
    
pca = PCA(n_components=10)
pca.fit(faces)
eigenfaces = pca.components_
faces_pca = pca.inverse_transform(pca.transform(faces))
index = 0
print(faces_pca.shape)
for face_pca in faces_pca:
    plt.imshow(reshape_face.reshape(200,180,3), cmap=plt.gray)
    #imwrite('E:/Primose/faces94/pca/' + str(index) + '.jpg' ,face_pca.reshape(200,180,3))
    index +=1
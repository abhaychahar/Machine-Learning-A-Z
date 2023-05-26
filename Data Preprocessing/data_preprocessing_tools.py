# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:12:49 2023

@author: abhay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset=pd.read_csv('Data.csv')

# feature matrix and dependent variable vector
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

print(x)
print(y)
print("------------")

# taking care of missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

print(x)
print("------------")

# encoding categorical data (matrix of features using one hot encoding)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoding',OneHotEncoder(),[0])],remainder='passthrough')
x=np.array(ct.fit_transform(x))

print(x)

# encoding catrgorical data (dependent variable vecto using label encoding)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

print(y)
print("------------")

# splitting data into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=1)

print(x_train)
print(x_test)
print(y_train)
print(y_test)
print("------------")

# feature scaling using standardisation (always done after splitting)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train[:,3:]=ss.fit_transform(x_train[:,3:])
x_test[:,3:]=ss.transform(x_test[:,3:])

print(x_train)
print(x_test)













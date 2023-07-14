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
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values

print(X)
print(y)
print("------------")

# taking care of missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3]=imputer.transform(X[:, 1:3])

print(X)
print("------------")

# encoding categorical data (matrix of features using one hot encoding)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoding', OneHotEncoder(), [0])], remainder='passthrough')
X=np.array(ct.fit_transform(X))

print(X)

# encoding catrgorical data (dependent variable vector using label encoding)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

print(y)
print("------------")

# splitting data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=1)

print(X_train)
print(X_test)
print(y_train)
print(y_test)
print("------------")

# feature scaling using standardisation (always done after splitting)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train[:, 3:]=ss.fit_transform(X_train[:, 3:])
X_test[:, 3:]=ss.transform(X_test[:, 3:])

print(X_train)
print(X_test)













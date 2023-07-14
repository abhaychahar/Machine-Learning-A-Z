# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 14:23:11 2023

@author: abhay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset=pd.read_csv('50_Startups.csv')

# feature matrix and dependent variable vector
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values

# encoding categorical data (matrix of features using one hot encoding)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoding', OneHotEncoder(), [3])], remainder='passthrough')
X=ct.fit_transform(X)

# splitting data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)

# training the multiple linear regression model on the training set
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train, y_train)

# predicting the results
y_pred=reg.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))




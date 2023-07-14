# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:59:14 2023

@author: abhay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset=pd.read_csv('Salary_Data.csv')

# feature matrix and dependent variable vector
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values

# splitting data into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=1)

# training the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train, y_train)

# predicting the results
y_pred_train=reg.predict(x_train)
y_pred_test=reg.predict(x_test)

# visualizing the results for training set
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, y_pred_train, color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience (years)')
plt.ylabel('Salary')
plt.show()

# visualizing the results for test set
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, y_pred_train, color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience (years)')
plt.ylabel('Salary')
plt.show()









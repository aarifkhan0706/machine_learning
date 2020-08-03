
# coding: utf-8

# # Implement Linear Regression problem. For example, based on a dataset comprising of existing set of prices and area/size of the houses, predict the estimated price of a given house

# This data was originally a part of UCI Machine Learning Repository and has been removed now. This data also ships with the scikit-learn library. There are 506 samples and 13 feature variables in this data-set. The objective is to predict the value of prices of the house using the given features.
# 
# The description of all the features is given below:
# 
# CRIM: Per capita crime rate by town
# 
# ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
# 
# INDUS: Proportion of non-retail business acres per town
# 
# CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 
# NOX: Nitric oxide concentration (parts per 10 million)
# 
# RM: Average number of rooms per dwelling
# 
# AGE: Proportion of owner-occupied units built prior to 1940
# 
# DIS: Weighted distances to five Boston employment centers
# 
# RAD: Index of accessibility to radial highways
# 
# TAX: Full-value property tax rate per $10,000
# 
# B: 1000(Bk - 0.63)², where Bk is the proportion of [people of African American descent] by town
# 
# LSTAT: Percentage of lower status of the population
# 
# MEDV: Median value of owner-occupied homes in $1000s
# 
# Import the required Libraries now

# In[16]:


import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.datasets import load_boston

boston_dataset = load_boston()

# boston_dataset is a dictionary
# let's check what it contains
boston_dataset.keys()


# In[4]:


boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()


# In[5]:


boston['MEDV'] = boston_dataset.target


# In[6]:


# check for missing values in all the columns
boston.isnull().sum()


# In[7]:


# set the size of the figure
sns.set(rc={'figure.figsize':(11.7,8.27)})

# plot a histogram showing the distribution of the target values
sns.distplot(boston['MEDV'], bins=30)
plt.show()


# In[8]:


# compute the pair wise correlation for all columns  
correlation_matrix = boston.corr().round(2)


# In[9]:


# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)


# In[10]:


plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')


# In[11]:


X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']


# In[12]:


from sklearn.model_selection import train_test_split

# splits the training and test data set in 80% : 20%
# assign random_state to any value.This ensures consistency.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[13]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[14]:


# model evaluation for training set

y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set

y_test_predict = lin_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[15]:


# plotting the y_test vs y_pred
# ideally should have been a straight line
plt.scatter(Y_test, y_test_predict)
plt.show()


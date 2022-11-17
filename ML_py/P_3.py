#!/usr/bin/env python
# coding: utf-8

# ## Importing the required libraries

# In[53]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# ## Importing the dataset

# In[2]:


dataset = pd.read_csv('temperatures.csv')


# In[3]:


dataset.head()


# ## Data preprocessing

# In[4]:


nrows = dataset.shape[0]
ncols = dataset.shape[1]
print('The given dataset has {} rows and {} columns'.format(nrows,ncols))


# In[5]:


# Check for missing values
dataset.isnull().any()


# In[6]:


dataset.dtypes


# In[7]:


# dataset statistics
dataset.describe()


# ## Separate the independent variable (x) and dependent variable (y) 

# In[8]:


x = np.array(dataset[['YEAR']])
# x set must be a 2D^^ array


# In[9]:


x


# In[10]:


# Here we choose January month to predict results using regression
y = np.array(dataset['JAN'])
#y must be 1D array


# In[11]:


y


# ## Spliting the dataset into training set and test set

# In[12]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# Here we keep the test set size as 25%


# In[13]:


ntest = len(x_test)
ntrain = len(x_train)
print('Length of training set is {} and of test set is {}'.format(ntrain,ntest))


# ## Create linear regression model

# In[14]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# ### Train the model on training set

# In[15]:


regressor.fit(x_train,y_train)


# ### Predict the test set results

# In[16]:


y_pred = regressor.predict(x_test)


# In[17]:


y_pred


# In[25]:


print('Predicted  Vs  Actual value')
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ### Evaluating the model performance

# In[44]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
MAE = mean_absolute_error(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)
r2score = r2_score(y_test,y_pred)
print('The MAE of the Linear Regression model is {}'.format(MAE))
print('The MSE of the Linear Regression model is {}'.format(MSE))
print('The R2 score of the Linear Regression model is {}'.format(r2score))


# ## Plotting the regression line

# The equation of line is y = mx + c where m is the slope and c is intercept

# In[45]:


m = regressor.coef_


# In[46]:


m


# In[47]:


c = regressor.intercept_


# In[48]:


c


# In[51]:


eqn = 'y = {} x + {}'.format(float(m),float(c))
print('The equation of line is {}'.format(eqn))


# In[62]:


plt.plot(x_test, regressor.predict(x_test), color='blue', linewidth=3)
plt.scatter(x_test, y_test, color='red')
plt.title("Temperature vs Year")
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.show()


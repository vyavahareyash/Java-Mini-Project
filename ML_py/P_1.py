#!/usr/bin/env python
# coding: utf-8

# ## Importing the required libraries

# In[67]:


import numpy as np 
import pandas as pd


# ## Importing the dataset

# In[2]:


dataset = pd.read_csv('Heart.csv')


# In[4]:


dataset.head()


# ### a) Find Shape of Data

# In[8]:


dataset.shape


# In[9]:


nrows = dataset.shape[0]
ncols = dataset.shape[1]
print('The given dataset has {} rows and {} columns'.format(nrows,ncols))


# ### b) Find Missing Values

# In[15]:


dataset.isnull().any()


# In[17]:


dataset.isnull().sum()


# In[20]:


# Column 'Ca' has 4 missing values and 'Thal' has 2 missing values
# As the number of missing values is very less we can crop the rows containing the missing values

dataset.dropna(inplace=True)


# In[21]:


dataset.isnull().any()


# In[22]:


# Now the dataset does not contain any missing value


# ### c) Find datatype of each column

# In[23]:


dataset.dtypes


# ### d) Finding out zero's

# In[35]:


# Iterate through columns
for column_name in dataset.columns:
    column = dataset[column_name]
    # Get the count of Zeros in column 
    count = (column == 0).sum()
    print('Count of zeros in column ', column_name, ' is : ', count)


# ### e) Find Mean age of patients

# In[39]:


meanAge = dataset['Age'].mean()
print('Mean age of patients is {}'.format(meanAge))


# ### f) Now extract only Age, Sex, ChestPain, RestBP, Chol. Randomly divide dataset in training (75%) and testing (25%).

# In[43]:


dataset2 = dataset[['Age','Sex','ChestPain','RestBP','Chol']]


# In[45]:


dataset2.head()


# In[48]:


# spliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dataset2,test_size=0.25)


# In[49]:


train_set


# In[51]:


test_set


# =======================================================================================================================
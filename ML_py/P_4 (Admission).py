#!/usr/bin/env python
# coding: utf-8

# ## Importing the required libraries

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# ## Importing the dataset

# In[3]:


dataset = pd.read_csv('Admission_Predict.csv')


# In[4]:


dataset.head()


# In[5]:


dataset.dtypes


# In[6]:


# Check for Missing values
dataset.isnull().any()


# ## Separating the independent variable (x) and dependent variable (y)

# In[7]:


x = dataset.iloc[:,1:-1]


# In[8]:


x


# In[9]:


y = dataset.iloc[:,-1]


# In[10]:


y


# #### Convert the dependent variable y into 0 and 1 for classification

# In[11]:


y[y>=0.5] = 1


# In[12]:


y[y<0.5] = 0


# ## Spliting the dataset into training set and test set

# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[14]:


x_train


# In[15]:


x_test


# In[16]:


y_train


# In[17]:


y_test


# ## Applying Feature Scaling

# In[18]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[19]:


x_train


# In[20]:


x_test


# ## Training the Decision Tree model on the Training set

# In[21]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy' ,random_state = 0)
classifier.fit(x_train, y_train)


# ## Predicting the Test set result

# In[22]:


y_pred = classifier.predict(x_test)


# ## Making the Confusion Matrix

# In[27]:


from sklearn.metrics import plot_confusion_matrix, accuracy_score, classification_report
print(f'Accuracy Score = {round(accuracy_score(y_test, y_pred),2)*100}%')
print(classification_report(y_test,y_pred))
plot_confusion_matrix(classifier, x_test, y_test)
plt.show()


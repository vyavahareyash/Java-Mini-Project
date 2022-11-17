#!/usr/bin/env python
# coding: utf-8

# ## Importing the required libraries

# In[1]:


import numpy as np 

# In[18]:


# =======================================================================================================================

# Through the diagnosis test I predicted 100 report as COVID positive, but only 45 of those were actually positive. Total 50 people in my sample were actually COVID positive. I have total 500
# samples.<br>Create confusion matrix based on above data and find<br>I. Accuracy<br>II. Precision<br>III. Recall<br>IV. F-1 score

# In[19]:


predicted_set = np.array(list(np.ones(45))+list(np.zeros(55)))


# In[20]:


predicted_set


# In[21]:


actual_set = np.array(list(np.ones(40))+list(np.zeros(52))+list(np.ones(8)))


# In[22]:


actual_set


# In[23]:


# Plotting confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(actual_set,predicted_set)


# In[24]:


from sklearn.metrics import classification_report
print(classification_report(actual_set,predicted_set))


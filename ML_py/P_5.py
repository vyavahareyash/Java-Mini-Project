#!/usr/bin/env python
# coding: utf-8

# ## Importing the libraries

# In[1]:


get_ipython().system('pip install nltk')


# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# ## Importing the dataset

# In[3]:


dataset = pd.read_csv('SMSSpamCollections.unknown', sep='\t', names=['label','text'])


# In[4]:


dataset


# In[5]:


dataset.shape


# In[6]:


# Representing spam with 1 and ham with 0
dataset['label'].replace({'spam':1 , 'ham':0} , inplace = True)


# In[7]:


dataset.head()


# ## Cleaning the text

# In[19]:


import re # Regular Expression
import nltk
nltk.download('stopwords')
# irrelevant words^
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #to derive the root word
corpus = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
#     substitute^      ^not including
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)


# In[20]:


corpus


# ## Creating the Bag of Words model

# In[21]:


from sklearn.feature_extraction.text import CountVectorizer  
cv = CountVectorizer(max_features = 1500)
#removes less frequent words^
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,0].values


# In[22]:


print(x)
print(y)


# In[23]:


len(x[0])


# ## Spliting the dataset into Training set and Test set

# In[24]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) 


# ## Training the Naive Bayes model on the Training set

# In[25]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)


# ## Predicting the Test set results

# In[26]:


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Making the Confusion Matrix

# In[29]:


from sklearn.metrics import plot_confusion_matrix, accuracy_score
plot_confusion_matrix(classifier, x_test, y_test)
print(f'Accuracy Score = {round(accuracy_score(y_test, y_pred)*100,2)}%')
plt.show()


# ## Applying K-Fold Cross Validation

# In[30]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)


# In[31]:


print('Accuracy: {:.2f} %'.format(accuracies.mean()*100))
print('Standard Deviation: {:.2f} %'.format(accuracies.std()*100))


#!/usr/bin/env python
# coding: utf-8

# ## Importing the libraries

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# ## Importing the dataset

# In[2]:


dataset = pd.read_csv('Mall_Customers.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.info()


# In[5]:


# Check for null values
dataset.isnull().any()


# In[6]:


#Rename column name
dataset.rename(columns = {'Genre':'Gender'} , inplace = True)


# In[7]:


dataset.head()


# In[8]:


#Find Gender Counts
print(dataset['Gender'].value_counts())


# In[9]:


# Representing male with 1 and female with 0
dataset['Gender'].replace({'Male':1 , 'Female':0} , inplace = True)


# In[10]:


dataset.head()


# ## Feature Scaling

# In[11]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset_scaled = sc.fit_transform(dataset)


# ## Dimensionality reduction

# In[12]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
dataset_pca = pca.fit_transform(dataset)
print("data shape after PCA :",dataset_pca.shape)
print("dataset_pca is:",dataset_pca)


# In[22]:


from sklearn.cluster import KMeans

#Minimum no. of clusters & squared distance
wcss_list = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i , init = 'k-means++'  , random_state = 1)
    kmeans.fit(dataset_pca) 
    wcss_list.append(kmeans.inertia_)

#Draw Elbow plot
#X & Y axis range
plt.plot(range(1,15) , wcss_list) 
plt.plot([4,4] , [0 , 500] , linestyle = '--' , alpha = 0.7)
#Elbow line
plt.text(4.2 , 300 , 'Elbow = 4')
#X & Y axis labels
plt.xlabel('K')
plt.ylabel('WCSS')
plt.show()


# ## Training the K-Means model on the dataset

# In[24]:


kmeans = KMeans(n_clusters = 4 , init = 'k-means++'  , random_state = 1)
kmeans.fit(dataset_pca)
cluster_id = kmeans.predict(dataset_pca)


# In[26]:


result_data = pd.DataFrame()
result_data['PC1'] = dataset_pca[:,0]
result_data['PC2'] = dataset_pca[:,1]
result_data['ClusterID'] = cluster_id


# In[27]:


#KMeans clustered ploting features
#cluster colors & tab details
cluster_colors = {0:'tab:red' , 1:'tab:green' , 2:'tab:blue' , 3:'tab:pink'}
cluster_dict = {'Centroid':'tab:orange','Cluster0':'tab:red' , 'Cluster1':'tab:green'
                , 'Cluster2':'tab:blue' , 'Cluster3':'tab:pink'}


# In[29]:


from matplotlib.lines import Line2D
#Scatter data 
#X & Y Value, result & cluster colors
plt.scatter(x = result_data['PC1'] , y = result_data['PC2'] 
                , c = result_data['ClusterID'].map(cluster_colors))

handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k, markersize=8) 
           for k, v in cluster_dict.items()]
plt.legend(title='color', handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k, markersize=8) 
           for k, v in cluster_dict.items()]
plt.legend(title='color', handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.scatter(x = kmeans.cluster_centers_[:,0] , y = kmeans.cluster_centers_[:,1] , 
            marker = 'o' , c = 'tab:orange', s = 150 , alpha = 1)

#Heading details
plt.title("Clustered by KMeans" , fontdict = plt_font)
plt.xlabel("PC1" , fontdict = plt_font)
plt.ylabel("PC2" , fontdict = plt_font)

#Show all data
plt.show()


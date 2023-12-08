#!/usr/bin/env python
# coding: utf-8

# # Workshop - 1: K- Means Clustering
# This notebook will walk through some of the basics of K-Means Clustering.
# 

# # Author: Claudio A. A. Mikhael

# In[2]:


import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns= iris.feature_names)
iris_df.head(12)


# In[3]:


ca = iris_df.iloc[:,[0,1,2,3]].values
from sklearn.cluster import KMeans
WCSS=[]

for i in range(1,11):
    Kmeans= KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    Kmeans.fit(ca)
    WCSS.append(Kmeans.inertia_)
    
plt.plot(range(1, 11), WCSS)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show


# In[4]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(ca)


# In[13]:


plt.scatter(ca[y_kmeans == 0, 0], ca[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(ca[y_kmeans == 1, 0], ca[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(ca[y_kmeans == 2, 0], ca[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:





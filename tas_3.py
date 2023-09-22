#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
data=pd.read_csv(r'C:/Users/ROWIDA/Documents/dataset_3/IRIS.csv')
data


# In[8]:


data.describe()


# In[9]:


data.dtypes


# In[10]:


data.info()


# In[11]:


data.isnull().values.any()


# In[12]:


data.isnull().sum()


# In[13]:


data.nunique()


# In[14]:


data.dropna(axis=0,inplace=True)
data


# In[15]:


data.duplicated().any()


# In[16]:


data.drop_duplicates()


# In[17]:


data.duplicated().any()


# In[19]:


data['species'].value_counts()


# In[31]:


data.corr(numeric_only=True)


# In[23]:


data['sepal_width'].hist()


# In[24]:


data['sepal_length'].hist()


# In[26]:


data['petal_length'].hist()


# In[27]:


data['petal_width'].hist()


# In[34]:


corr = data.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(corr, annot=True, ax=ax)


# In[36]:


sns.pairplot(data,hue='species')
plt.show()


# In[38]:


le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])
data


# In[39]:


data.dtypes


# In[40]:


X = data.drop(columns=['species'])
Y = data['species']


# In[41]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[45]:


lg = LogisticRegression(max_iter=1000)
lg.fit(x_train,y_train)

print("Accuracy: ",lg.score(x_test, y_test) * 100)


# In[47]:


Des = DecisionTreeClassifier()
Des.fit(x_train, y_train)
print("Accuracy: ",Des.score(x_test, y_test) * 100)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





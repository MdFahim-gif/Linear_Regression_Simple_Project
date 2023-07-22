#!/usr/bin/env python
# coding: utf-8

# In[7]:


#importlibraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[8]:


dataset = pd.read_csv('50_SupermarketBranches.csv')


# In[9]:


dataset


# In[10]:


dataset.isnull().sum()


# In[11]:


#for droop dependent variable 
x=dataset.drop(['Profit'],axis=1)


# In[12]:


#insert profit in new variable
y=dataset['Profit']


# In[13]:


#one-hot-encoding to conver string data into numerical data 
city=pd.get_dummies(x['State'],drop_first=True)


# In[14]:


city


# In[15]:


#now drop for the string coulmn from the dataset
x=x.drop('State',axis=1)


# In[16]:


x


# In[17]:


#now concatenation to add the converted string value in dataset 
x=pd.concat([x,city],axis=1)


# In[18]:


x


# In[19]:


#Now seperate or declear train and test data from data set 
#for this we need to import a library called train_test_split from sklearn.model_selection

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)


# In[20]:


#now ,for appling linear regration we need to import LinearRegression from sklearn.linear_model
from sklearn.linear_model import LinearRegression
#now, we need to create an object for this LinearRegression library

Module=LinearRegression()


# In[21]:


#now we need to fit x and y values in this model
Module.fit(xtrain,ytrain)


# In[22]:


xtest


# In[23]:


ytest


# In[24]:


#predicting the test result
predict=Module.predict(xtest)


# In[25]:


predict


# In[26]:


y_predict_train=Module.predict(xtrain)


# In[27]:


#result for model and findings the accurecy
Module.score(xtest,ytest)


# In[28]:


#for R2 value calculation(which is also use as accurecy)
from sklearn.metrics import r2_score
score=r2_score(ytest,predict)


# In[29]:


score


# In[30]:


plt.scatter(ytrain,y_predict_train)
plt.xlabel("actual profit")
plt.ylabel("predict profit")
plt.show()


# In[ ]:





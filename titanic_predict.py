#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sbn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
titanic_data = sbn.load_dataset('titanic')


# In[2]:


titanic_data.isnull().sum()


# In[3]:


titanic_data.info()


# In[4]:


sbn.heatmap(titanic_data.isnull(), yticklabels='auto')


# In[5]:


titanic_data["age"].plot.hist()


# In[6]:


sbn.boxplot(x="pclass", y="age",data=titanic_data)


# In[7]:


titanic_data.dropna(inplace=True)


# In[8]:


titanic_data.head(2)


# In[9]:


titanic_data.isnull().sum()


# In[10]:


import pandas as pd


# In[11]:


Sex = pd.get_dummies(titanic_data['sex'])


# In[12]:


Sex.head(2)


# In[13]:


Sex = pd.get_dummies(titanic_data['sex'],drop_first=True)


# In[14]:


Sex.head(2)


# In[15]:


titanic_data['embarked'].head(2)


# In[16]:


embark = pd.get_dummies(titanic_data['embarked'])


# In[18]:


embark.head(2)


# In[17]:


embark = pd.get_dummies(titanic_data['embarked'], drop_first=True)


# In[19]:


embark.head(2)


# In[21]:


Pc1.head(4)


# In[20]:


Pc1 = pd.get_dummies(titanic_data['pclass'])


# In[22]:


Pc1 = pd.get_dummies(titanic_data['pclass'],drop_first=True)


# In[23]:


Pc1.head(4)


# In[24]:


titanic_data = pd.concat([titanic_data,Sex,embark,Pc1],axis=1)


# In[25]:


titanic_data.head(3)


# In[26]:


titanic_data.drop(['sex','embark_town','embarked','who','class','deck','embark_town','alive'], axis=1,inplace=True)


# In[27]:


titanic_data.head(3)


# In[28]:


titanic_data.drop(['pclass'],axis =1, inplace=True)


# In[29]:


titanic_data.head(4)


# In[30]:


X = titanic_data.drop('survived', axis=1)


# In[31]:


y = titanic_data['survived']


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)


# In[34]:


from sklearn.linear_model import LogisticRegression


# In[35]:


logmodel = LogisticRegression()


# In[36]:


X_train.columns = X_train.columns.astype(str)


# In[37]:


logmodel.fit(X_train, y_train)


# In[42]:


X_test.columns = X_test.columns.astype(str)


# In[43]:


predictions = logmodel.predict(X_test)


# In[38]:


from sklearn.metrics import classification_report


# In[44]:


classification_report(y_test, predictions)


# In[46]:


final = classification_report(y_test, predictions)


# In[50]:


new = '              precision    recall  f1-score   support\n\n           0       0.50      0.65      0.56        17\n           1       0.82      0.71      0.76        38\n\n    accuracy                           0.69        55\n   macro avg       0.66      0.68      0.66        55\nweighted avg       0.72      0.69      0.70        55\n'


# In[51]:


len(new)


# In[52]:


new[0:100]


# In[53]:


new[101:200]


# In[54]:


new[201:]


# In[ ]:





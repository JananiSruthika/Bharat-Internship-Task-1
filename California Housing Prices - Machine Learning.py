#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#LOADING DATASET
data = pd.read_csv("housing.csv")


# In[3]:


data


# In[4]:


#INFORMATION ABOUT THE DATA
data.info()


# In[5]:


data.dropna(inplace=True)


# In[6]:


data.info()


# In[7]:


#MODEL TRAINING
from sklearn.model_selection import train_test_split
X = data.drop(['median_house_value'],axis=1)
y = data['median_house_value']


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[9]:


train_data = X_train.join(y_train)


# In[10]:


train_data


# In[11]:


#HISTOGRAMS
train_data.hist(figsize=(15, 8))


# In[12]:


#CORRELATION MATRIX
train_data.corr()


# In[13]:


plt.figure(figsize=(15, 8))
sns.heatmap(train_data.corr(),annot=True, cmap="YlGnBu")


# In[14]:


train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)


# In[15]:


train_data.hist(figsize = (15, 8))


# In[16]:


train_data.ocean_proximity.value_counts()


# In[17]:


train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis = 1)
train_data


# In[18]:


train_data.corr()


# In[19]:


plt.figure(figsize=(15, 8))
sns.heatmap(train_data.corr(),annot=True, cmap="YlGnBu")


# In[20]:


#SCATTERPLOT
plt.figure(figsize=(15, 8))
sns.scatterplot(x="latitude", y="longitude", data=train_data, hue="median_house_value", palette="coolwarm")


# In[21]:


train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']


# In[22]:


plt.figure(figsize=(15, 8))
sns.heatmap(train_data.corr(),annot=True, cmap="YlGnBu")


# In[32]:


#TRAINING AND TESTING THE MODELS
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train, y_train = train_data.drop(['median_house_value'], axis = 1),train_data['median_house_value']
X_train_s = scaler.fit_transform(X_train)

reg = LinearRegression()
reg.fit(X_train_s, y_train)


# In[24]:


test_data = X_test.join(y_test)


# In[25]:


test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)


# In[26]:


test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'], axis = 1)


# In[27]:


test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']


# In[28]:


test_data


# In[33]:


X_test, y_test = test_data.drop(['median_house_value'], axis = 1),test_data['median_house_value']


# In[34]:


X_test


# In[35]:


X_test_s = scaler.transform(X_test)


# In[36]:


#PERFORMANCE METRICS
reg.score(X_test_s, y_test)


# In[40]:


from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(X_train_s, y_train)


# In[41]:


forest.score(X_test_s, y_test)


# In[50]:


#CROSS VALIDATION OF MODELS
from sklearn.model_selection import GridSearchCV


# In[51]:


forest = RandomForestRegressor()

param_grid = {
    "n_estimators": [3, 10, 30],
    "max_features": [2, 4, 6, 8],
}

grid_search = GridSearchCV(forest, param_grid, cv=5,
                          scoring="neg_mean_squared_error",
                          return_train_score=True)
grid_search.fit(X_train_s, y_train)


# In[52]:


grid_search.best_estimator_


# In[53]:


grid_search.best_estimator_.score(X_test_s, y_test)


# In[ ]:





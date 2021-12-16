#!/usr/bin/env python
# coding: utf-8

# ## Import library

# In[1]:


import seaborn as sns

import yfinance as yf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import ta 
import plotly.graph_objects as go
import stock_lib as stlib


# ## Import data

# In[2]:


data = yf.download("TSLA", period='60d', interval='5m')
data = data.reset_index()

macd = ta.trend.MACD(close=data['Close'])
bb = ta.volatility.BollingerBands(close=data['Close'])
cmf = ta.volume.ChaikinMoneyFlowIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'])
rsi = ta.momentum.RSIIndicator(close=data['Close'])

data['rsi'] = rsi.rsi()
data['macd'] = macd.macd_diff()
data['cmf'] = cmf.chaikin_money_flow()
data['bb_width'] = bb.bollinger_wband()

data['rsi_5'] = data['rsi'].diff(periods=5)
data['macd_5'] = data['macd'].diff(periods=5)
data['cmf_5'] = data['cmf'].diff(periods=5)

data['close_5'] = data['Close'].diff(periods=-5)
data['close_5'] = (data['close_5'] / data['Close']) * 100

data['close_10'] = data['Close'].diff(periods=10)
data['close_10'] = (data['close_10'] / (data['Close'] - data['close_10'])) * 100

data = data.dropna()
# data1 = data[['rsi', 'rsi_diff', 'close_5', 'close_10', 'macd', 'cmf', 'bb_width']]
data1 = data[['rsi', 'close_5', 'macd', 'cmf', 'bb_width', 'rsi_5', 'macd_5', 'cmf_5']]


# In[23]:


# data2 = data1[data1['close_5'] > 0]
data2 = data1
data2 = data2.reset_index()
data2 = data2.drop('index', axis=1)
data2


# In[4]:


label = []

for i in range(0, len(data2)):
    
    if data2['close_5'][i] < -3:
        label.append(0)
    elif data2['close_5'][i] < -2:
        label.append(1)
    elif data2['close_5'][i] < -0.5:
        label.append(2)
    elif data2['close_5'][i] < 0.5:
        label.append(3)
    elif data2['close_5'][i] < 2:
        label.append(4)
    else:
        label.append(5)
        
data2['label'] = label


# ## Create Model

# In[5]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

clf2 = RandomForestRegressor(random_state=0)

X = data2.drop(['close_5', 'label'], axis=1).to_numpy()
y = data2[['close_5']].to_numpy()

X = np.array(X)
y = np.array(y).ravel()

print(X.shape)
print(y.shape)


# ## Regression

# In[6]:


from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

clf2.fit(X_train, y_train)

print(clf2.score(X, y))


# In[7]:


prediction = clf2.predict(X_test)

print("prediction:   ", prediction)
print()
print("actual value: ", y_test)
print()
diff = prediction - y_test
print("different:    ", np.absolute(diff))
print()


# In[8]:


metrics.r2_score(y_test, prediction)


# ## Classification

# In[9]:


from sklearn.ensemble import RandomForestClassifier

clf1 = RandomForestClassifier(random_state=0)

X = data2.drop(['close_5', 'label'], axis=1).to_numpy()
y = data2[['label']].to_numpy()

X = np.array(X)
y = np.array(y).ravel()

print(X.shape)
print(y.shape)


# In[10]:


scores = cross_val_score(clf1, X, y, cv=10)


# In[11]:


scores


# In[12]:


data2


# In[13]:


data2[data2['close_5'] < -3]


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

clf1.fit(X_train, y_train)


# In[15]:


prediction = clf1.predict(X_test)

print("prediction:   ", prediction)
print()
print("actual value: ", y_test)
print()


# In[19]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

confusion_matrix(y_test, prediction)


# In[22]:


plot_confusion_matrix(clf1, X, y)  
plt.show() 


# In[ ]:





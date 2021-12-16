import streamlit as st
import time
import yfinance as yf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import ta 
import plotly.graph_objects as go

from scipy.signal import find_peaks

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

def regular_bearish(close_peaks, ind_peaks):
    sum1 = 0
    sum2 = 0
    if len(close_peaks) > 0:
        sum1 = close_peaks[0] - close_peaks[-1]
    if len(ind_peaks) > 0:
        sum2 = ind_peaks[0] - ind_peaks[-1]
    
    if sum1 > 0 and sum2 < 0:
        return 1
    return 0

def hidden_bearish(close_peaks, ind_peaks):
    sum1 = 0
    sum2 = 0
    if len(close_peaks) > 0:
        sum1 = close_peaks[0] - close_peaks[-1]
    if len(ind_peaks) > 0:
        sum2 = ind_peaks[0] - ind_peaks[-1]
    
    if sum1 < 0 and sum2 > 0:
        return 1
    return 0


# Web app title:
st.write("""
# Simple Stock Predictor App
""")

# Description:
st.markdown("Our Group: Khuong Tran, Deniz Jasarbasic, Eric Karpovits, Raz Levi")
st.markdown("Language: Python")
st.markdown("Technologies:")
st.markdown("Method & Project Description: Our stock prediction model is using a technical analysis approach at predicting the daily price movement of a stock and uses this data to attempt to predict future price movements.")
st.markdown("Demo:")

data = yf.download("TSLA", period='60d', interval='5m')

macd = ta.trend.MACD(close=data['Close'])
bb = ta.volatility.BollingerBands(close=data['Close'])
cmf = ta.volume.ChaikinMoneyFlowIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'])
rsi = ta.momentum.RSIIndicator(close=data['Close'])

data['rsi'] = rsi.rsi()
data['macd'] = macd.macd_diff()
data['cmf'] = cmf.chaikin_money_flow()
data['bb_width'] = bb.bollinger_wband()

data['close_3'] = data['Close'].diff(periods=-5)
data['close_3'] = (data['close_3'] / data['Close']) * 100

data['close_8'] = data['Close'].diff(periods=-8)
data['close_8'] = (data['close_8'] / data['Close']) * 100

data['close_13'] = data['Close'].diff(periods=-13)
data['close_13'] = (data['close_13'] / data['Close']) * 100
data['close_21'] = data['Close'].diff(periods=-21)
data['close_21'] = (data['close_21'] / data['Close']) * 100
data = data.dropna()
data = data.reset_index()

# data['macd'] = macd.macd_diff()
# data['cmf'] = cmf.chaikin_money_flow()
# data['bb_width'] = bb.bollinger_wband()

label = []

for i in range(0, len(data)):
    num1 = i
    num2 = i + 20
    
    close_data = data.iloc[num1:num2][['Close']].to_numpy()
    close_data = np.array(close_data).flatten()

    index1, _ = find_peaks(close_data, width=1)
    close_peaks = close_data[index1]

    rsi_data = data.iloc[num1:num2][['cmf']].to_numpy()
    rsi_data = np.array(rsi_data).flatten()

    index2, _ = find_peaks(rsi_data, width=1)
    rsi_peaks = rsi_data[index2]
    
    label.append(regular_bearish(close_peaks, rsi_peaks))
    
data['regular_bearish'] = label

label = []

for i in range(0, len(data)):
    num1 = i
    num2 = i + 20
    
    close_data = data.iloc[num1:num2][['Close']].to_numpy()
    close_data = np.array(close_data).flatten()

    index1, _ = find_peaks(close_data, width=1)
    close_peaks = close_data[index1]

    rsi_data = data.iloc[num1:num2][['cmf']].to_numpy()
    rsi_data = np.array(rsi_data).flatten()

    index2, _ = find_peaks(rsi_data, width=1)
    rsi_peaks = rsi_data[index2]
    
    label.append(hidden_bearish(close_peaks, rsi_peaks))
    
data['hidden_bearish'] = label

data = data.drop(['High', 'Low', 'Adj Close'], axis=1)
data = data.drop(['Datetime', 'Open', 'Close', 'Volume'], axis=1)

data

clf = RandomForestRegressor(random_state=1)

X = data.drop(['close_3', 'close_8', 'close_13', 'close_21'], axis=1).to_numpy()
y = data[['close_3', 'close_8', 'close_13', 'close_21']].to_numpy()

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=1)

clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
score = metrics.r2_score(y_test, prediction, multioutput='uniform_average')
# score1 = metrics.r2_score(y_test[0], prediction[0])
# score2 = metrics.r2_score(y_test[1], prediction[1])
# score3 = metrics.r2_score(y_test[2], prediction[2])
# score4 = metrics.r2_score(y_test[3], prediction[3])

st.write("""
# First Model
""")
st.markdown("Actual Value:")
y_test

st.markdown("Prediction Value:")
prediction

st.markdown("R^2 score:")
score

st.write("""
# Second Model using divergence
""")
data1 = data[(data['regular_bearish'] == 1) | ((data['hidden_bearish'] == 1))]

clf = RandomForestRegressor(random_state=1)

X = data1.drop(['close_3', 'close_8', 'close_13', 'close_21'], axis=1).to_numpy()
y = data1[['close_3', 'close_8', 'close_13', 'close_21']].to_numpy()

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=1)

clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
score = metrics.r2_score(y_test, prediction, multioutput='uniform_average')

st.markdown("Actual Value:")
y_test

st.markdown("Prediction Value:")
prediction

st.markdown("R^2 score:")
score



# Visualization of the Confusion Matrix:

# Reruns the web app:
st.button("Re-run")
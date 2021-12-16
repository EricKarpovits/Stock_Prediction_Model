#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import plotly as plt
import yfinance as yf

import plotly.express as px

# class Position
class Position:
    def __init__(self, t_buy=0, t_sell=0, t_avg=1):
        self.buy = t_buy
        self.sell = t_sell
        self.close_buy = 0
        self.close_sell = 0
        self.avg = t_avg
        self.prev_close = t_buy
        self.end = False
        self.profit = 0
        if self.buy > 0:
            self.stop_loss = self.buy - self.avg
        else:
            self.stop_loss = self.sell + self.avg
    def should_stop(self, close):
        if self.buy > 0 and close < self.stop_loss:
            return True
        if self.sell > 0 and close > self.stop_loss:
            return True
        return False
    def update_stop_loss(self, close):
        if self.buy > 0:
            if close > self.prev_close:
                self.prev_close = close
                self.stop_loss = close - self.avg
        else:
            if close < self.prev_close:
                self.prev_close = close
                self.stop_loss = close + self.avg
        return self.stop_loss            
    def isEnd(self):
        return self.end
    def close_position(self, close):
        if self.buy > 0:
            self.end = True
            self.profit = close - self.buy
            self.close_sell = close
        else:
            self.end = True
            self.profit = self.sell - close
            self.close_buy = close
        return self.profit
    def isEmpty(self):
        if self.buy == 0 and self.sell == 0:
            return True
        return False
    def isBuy(self):
        return self.buy > 0
    def isSell(self):
        return self.sell > 0
    
    
class BuySellHist:
    def __init__(self):
        self.buy = []
        self.sell = []
        self.profit = []
        self.stop = []
        self.datetime = []
    def update_hist(self, t_buy, t_sell, t_profit, t_stop, t_dtime):
        self.buy.append(t_buy)
        self.sell.append(t_sell)
        self.profit.append(t_profit)
        self.stop.append(t_stop)
        self.datetime.append(t_dtime)
        
        

# return simple moving average data
def sma(close, number=9):
    temp_ma = []
    for i in range(0, number - 1):
        temp_ma.append(0)
    for i in range(number-1, len(close)):
        # temp_ma.append(1)
        ma_value = close[i-number:i].sum()
        ma_value = ma_value / number
        temp_ma.append(ma_value)
    return temp_ma


# return typical price
def typical_price(df):
    tp = df[['Close', 'High', 'Low']].mean(axis=1)
    return tp


# return standard deviation for the typical price in the dataset
def bb_std(df, number=20):
    temp = []
    for i in range(number):
        temp.append(0)
    for i in range(number, len(df)):
        bb_std = np.std(df['TP'][i - number:i])
        temp.append(bb_std)
    return temp


# return bollinger bands for the dataset
def bb(df, number=20):
    df['TP'] = typical_price(df)
    df['SMA20'] = sma(df, number, name='TP')
    df['std'] = bb_std(df, number)
    df['Upper'] = df['SMA20'] + 2 * df['std']
    df['Lower'] = df['SMA20'] - 2 * df['std']
    
    
# return weight multiplier
def weight_multiplier(number):
    wm = 2 / (number + 1)
    return wm


# return exponential moving average 
def ema(df, number=10, name="Close"):
    temp_ema = []
    k = weight_multiplier(number)
    for i in range(number):
        temp_ema.append(0)
    for i in range(number, len(df[name])):
        ma_value = df['TP'][i] * k + temp_ema[i - 1] * (1 - k)
        temp_ema.append(ma_value) 
    return temp_ema

def obv(df):
    OBV = []
    OBV.append(0)

    for i in range(1, len(df.Close)):
        if df.Close[i] > df.Close[i-1]:
            obv_value = OBV[i-1] + df.Volume[i]
            OBV.append(obv_value)
        elif df.Close[i] < df.Close[i-1]:
            obv_value = OBV[i-1] - df.Volume[i]
            OBV.append(obv_value)
        else:
            OBV.append(OBV[i-1])
        return OBV
    
# return buy signal and sell signal 
# receive df, first sma and second sma as parameters
def cross_strategy(close, datetime, col1, col2, ignore=1, number=100):
    pos = Position()
    temp_diff = col1 - col2
    hist = BuySellHist()
    for i in range(ignore, len(close)):
        if temp_diff[i] > 0 and temp_diff[i-1] < 0:
            # buy signal
            if pos.isEmpty():
                pos = Position(t_buy=close[i], t_avg=number)
                hist.update_hist(pos.buy, pos.sell, pos.profit, pos.stop_loss, datetime[i])
            else:
                pos.close_position(close[i])
                hist.update_hist(pos.close_buy, pos.close_sell, pos.profit, pos.stop_loss, datetime[i])
                pos = Position(t_buy=close[i], t_avg=number)
                hist.update_hist(pos.buy, pos.sell, pos.profit, pos.stop_loss, datetime[i])
        elif temp_diff[i] < 0 and temp_diff[i-1] > 0:
            # sell signal
            if pos.isEmpty():
                pos = Position(t_sell=close[i], t_avg=number)
                hist.update_hist(pos.buy, pos.sell, pos.profit, pos.stop_loss, datetime[i])
            else:
                pos.close_position(close[i])
                hist.update_hist(pos.close_buy, pos.close_sell, pos.profit, pos.stop_loss, datetime[i])
                pos = Position(t_sell=close[i], t_avg=number)
                hist.update_hist(pos.buy, pos.sell, pos.profit, pos.stop_loss, datetime[i])
        else:
            if pos.should_stop(close[i]):
                pos.close_position(close[i])
                hist.update_hist(pos.close_buy, pos.close_sell, pos.profit, pos.stop_loss, datetime[i])
                pos = Position()
            else:
                pos.update_stop_loss(close[i])
    return hist
    

def fig_setting(fig, number):
    fig.update_xaxes(
        type='category', 
        showticklabels=False,
    )
    fig.update_layout(
        xaxis_range=[number, number + 200],
        xaxis_rangeslider_visible=False,
        dragmode='pan',
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
        ),
        showlegend=True,
    )
    
    
def average_diff(close, number=10):
    avg = 0
    
    diff = close.diff(periods=1)
    diff = diff.abs()
    avg = diff.mean()
    
    return avg
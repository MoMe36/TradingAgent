# ============================================
# https://medium.com/swlh/free-historical-market-data-download-in-python-74e8edd462cf
# ============================================


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from mplfinance.original_flavor import candlestick_ohlc
import mplfinance as mpf
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from yahoofinancials import YahooFinancials

def display(stock_data): 

    if(isinstance(stock_data, str)):
        stock_data = pd.read_csv(stock_data)
        stock_data.Date = pd.to_datetime(stock_data.Date)
        stock_data = stock_data.set_index('Date')

    scaler = StandardScaler()
    stock_data_tf = pd.DataFrame(scaler.fit_transform(stock_data), 
                            columns = stock_data.columns, 
                            index = stock_data.index)


    f, axes = plt.subplots(2,1, figsize = (8,8))
    axes = axes.flatten()


    mpf.plot(stock_data, ax = axes[0])
    mpf.plot(stock_data_tf, ax = axes[1])
    plt.show()

def get_data(stock_name): 

    data = yf.download(stock_name, 
                    start = '2000-01-01', end = '2021-03-20')
    data.to_csv('trading_env/envs/{}.csv'.format(stock_name))
    display(data)
if __name__ == "__main__": 

    # get_data('MCD')
    display('trading_env/envs/MCD.csv')
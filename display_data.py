import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from mplfinance.original_flavor import candlestick_ohlc
import mplfinance as mpf


def display(): 
    data = pd.read_csv('price.csv').drop('Volume', axis =1)
    data.Date = pd.to_datetime(data.Date)
    data = data.set_index('Date')
    input(data.head())
    mpf.plot(data)
    plt.show()

if __name__ == "__main__": 

    display()
from pandas_datareader import data as web
import pandas as pd
from datetime import date


def get_data(ticker):
    """
    INPUT: ticker - (str) ticker of a stock to be predicted
    
    OUTPUT: data - (pandas dataframe) dataframe with stock price info. 
            Includes following columns: 
                - Open
                - High
                - Low
                - Close
                - Adj Close
                - Volume
    """
    
    start_date = "2017-01-01"
    today = date.today()
    
    data = web.get_data_yahoo(ticker, end = today)
    
    return data

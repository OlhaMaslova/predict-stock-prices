import sys

from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from datetime import date
import yfinance as yf
import seaborn as sns
from math import sqrt
import pandas as pd
import numpy as np

from lib import get_data

sns.set_style('whitegrid')
yf.pdr_override()

def get_ticker():
    """
    
    """
    # get ticker from user input. ex: "AAPL", "DOW", "GOOG"
    print('Please enter ticker: ')
    ticker = input()
    
    return ticker

def load_data(ticker):
    """
    
    """
    try:
        df = get_data(ticker.upper())
        print(df.head())
    except Exception:
        print('Data can not be loaded. Please check your ticker and try again')
        
    return df
    

def save_data(df):
    """
    
    """
    engine = create_engine('sqlite:///Prices.db')
    df.to_sql(ticker, engine, index=False)
    

if __name__ == "__main__":
    
    # get ticker
    ticker = get_ticker()
    
    # load data from Yahoo finance
    print('Loading data for:', ticker, '...')
    df = load_data(ticker)
    
    print('Saving data ...')
    save_data(df)

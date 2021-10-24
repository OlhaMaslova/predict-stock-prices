# import sys

from sqlalchemy import create_engine
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np

from lib import get_data

tickers = ("AAPL", "GOOG", "NFLX", "FB", "AMZN")


def load_data(ticker):
    """

    """
    df = pd.Dataframe()
    for ticker in tickers:
        try:
            df = pd.concat(df, get_data(ticker.upper()))
            print(df.head())
        except Exception:
            print('Data can not be loaded. Please check your ticker and try again')

    return df


# def save_data(df):
#     """

#     """
#     engine = create_engine('sqlite:///Prices.db')
#     df.to_sql(ticker, engine, index=False)


# if __name__ == "__main__":

#     # get ticker
#     ticker = get_ticker()

#     # load data from Yahoo finance
#     print('Loading data for:', ticker, '...')
#     df = load_data(ticker)

#     print('Saving data ...')
#     save_data(df)

import streamlit as st
from datetime import date
from plotly import graph_objs as go
import yfinance as yf
import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn import preprocessing

from WindowGenerator import WindowGenerator

START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "NFLX", "FB", "AMZN")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

period = st.slider("Days to predict", 1, 30)


def load_data(ticker):
    """

    """
    try:
        data = yf.download(ticker.upper(), START, TODAY)
        data.reset_index(inplace=True)
    except Exception:
        print('Data can not be loaded. Please check your ticker and try again')

    return data


def ma_5(x): return x.rolling(5).mean()
def ma_200(x): return x.rolling(200).mean()
def ema_50(x): return x.ewm(span=50).mean()
def plus_minus_fn(x): return x.rolling(20).sum()


def engineer_features(df):
    df.loc[:, 'ticker'] = selected_stock
    df.loc[:, 'HL_PCT'] = ((df['High'] - df['Low']) /
                           df['Adj Close'] * 100).values
    df.loc[:, 'PCT_change'] = (
        (df['Adj Close'] - df['Open']) / df['Open'] * 100.0).values

    # Logarithmic transformation
    df.loc[:, 'Volume'] = df['Volume'].apply(np.log)

    # Differencing
    df.loc[:, 'Change_1'] = df['Volume'].diff()
    df.loc[:, 'Change_50'] = df['Volume'].diff(50)

    # Moving Averages log of 5 day ma of volume
    df.loc[:, 'ma_5'] = df.groupby(
        by='ticker')['Volume'].apply(ma_5).apply(np.log)

    # daily volume vs 200 day ma
    df.loc[:, 'ma_200'] = df['Volume'] / \
        df.groupby(by='ticker')['Volume'].apply(ma_200) - 1

    # daily closing price vs 50 day exponential ma
    df.loc[:, 'ema_50'] = df['Adj Close'] / \
        df.groupby(by='ticker')['Adj Close'].apply(ema_50) - 1

    # signing: volume increased or decreased?
    df.loc[:, 'volume_sign'] = df['PCT_change'].apply(np.sign)

    # how many days in a raw a value has increased / decreased
    df.loc[:, 'days_vol_increased'] = df.groupby(
        by='ticker')['volume_sign'].apply(plus_minus_fn)

    return df


def engineer_date_features(df):
    df = df.set_index('Date')
    # One-Hot Encoding for month
    month_of_year = df.index.get_level_values(level='Date').month
    one_hot_frame = pd.DataFrame(pd.get_dummies(month_of_year))
    one_hot_frame.index = df.index

    # create column names
    columns = month = ["Jan", "Feb", "Mar", "Apr",
                       "May", "Jun", "Jul", "Aug",
                       "Sep", "Oct", "Nov", "Dec"]

    one_hot_frame.columns = columns
    df = df.join(one_hot_frame)

    # Weekday features
    weekdays = pd.Series(df.index)
    dummy = pd.get_dummies(weekdays.dt.dayofweek.values)
    dummy.index = df.index

    dummy.columns = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

    df = df.join(dummy)

    return df


def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Adj Close'], name='Adjusted Close'))
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Close'], name='Close'))
    fig.layout.update(title_text='Time Series Data',
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


def main():
    data_load_state = st.text("Load data ...")
    data = load_data(selected_stock)
    data_load_state.text("Load data ... done!")

    st.subheader('Raw Data')
    st.write(data.tail())
    plot_raw_data(data)

    # feature engineering
    df = engineer_features(data)
    df = engineer_date_features(df)
    df = df.drop(columns=['Open', 'Low', 'High', 'Volume', 'Close', 'ticker'])

    # Scaling
    scaler = preprocessing.StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index)
    df_scaled.columns = df.columns

    # load a model
    conv_model = tf.keras.models.load_model('models/conv_model')

    multi_window = WindowGenerator(input_width=100, label_width=period, shift=period,
                                   data=df_scaled, label_columns=['Adj Close'])

    predictions = conv_model.predict(multi_window.data_set)
    forecast = scaler.inverse_transform(predictions)
    forecast = forecast[-1:, :, 0]

    #forecast = pd.DataFrame(predictions)

    st.write('Forecast data')
    st.write(forecast)
    # plot_raw_data(predictions)


if __name__ == "__main__":
    main()

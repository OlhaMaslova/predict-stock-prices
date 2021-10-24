import streamlit as st
from datetime import date
from plotly import graph_objs as go
import yfinance as yf
import pandas as pd

from fbprophet.plot import plot_plotly
from fbprophet import Prophet


START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "NFLX", "FB", "AMZN")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of Prediction", 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    """

    """
    try:
        data = yf.download(ticker.upper(), START, TODAY)
        data.reset_index(inplace=True)
    except Exception:
        print('Data can not be loaded. Please check your ticker and try again')

    return data


data_load_state = st.text("Load data ...")
data = load_data(selected_stock)
data_load_state.text("Load data ... done!")


st.subheader('Raw Data')
st.write(data.tail())


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Adj Close'], name='Adjusted Close'))
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Close'], name='Close'))
    fig.layout.update(title_text='Time Series Data',
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('Forecast data')
fig_forecast = plot_plotly(m, forecast)
st.plotly_chart(fig_forecast)


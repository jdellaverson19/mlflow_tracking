import pandas as pd
import numpy as np
import os
import sys
import joblib

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()

# For time stamps
from datetime import datetime, timedelta

# argList = []
# for arg in sys.argv[1:]:
#    argList.append(arg)

# predictStock = argList[0]


def getPrediction(predictStock, yesterday=False):
    a = os.listdir(path="./models")
    predictStock = predictStock.upper()
    if f"{predictStock}.keras" in a:
        print("tis in there")
        scaler = joblib.load(rf"models/{predictStock}.gz")
        model = keras.models.load_model(rf"models/{predictStock}.keras")
    else:
        print("tis not")
        scaler = joblib.load(rf"models/scaler.gz")
        model = keras.models.load_model(rf"models/model.keras")

    # Now to do a specific prediction
    stock_quote = 0
    if yesterday == False:
        stock_quote = pdr.get_data_yahoo(
            predictStock, start="2024-01-01", end=datetime.now()
        )
    else:
        stock_quote = pdr.get_data_yahoo(
            predictStock, start="2024-01-01", end=(datetime.now() - timedelta(1))
        )

    new_df = stock_quote.filter(["Close"])
    last_60_days = new_df[-60:].values
    # Scale the data to be values between 0
    last_60_days_scaled = scaler.transform(last_60_days)

    # Create an empty list
    pred_list = []
    # Append the past 60days
    pred_list.append(last_60_days_scaled)

    # Convert the pred_list data into numpy array
    pred_list = np.array(pred_list)

    # Reshape the data
    pred_list = np.reshape(pred_list, (pred_list.shape[0], pred_list.shape[1], 1))
    # Get predicted scaled price
    pred_price = model.predict(pred_list)
    # undo the scaling
    pred_price = scaler.inverse_transform(pred_price)
    print(f"Price of {predictStock} tomorrow:{pred_price}")
    return pred_price


if __name__ == "__main__":
    print(getPrediction("AAPL", True))
# a = data.tail(1)["Close"]
# time_match = a.index[0]
# nextPred = pd.Series(
#     predictions[-1],
#     index=[time_match],
#     name="pred_Close",
# )

# print(nextPred)
# todayPred = pd.concat([a, nextPred], axis=1)
# print("potat2")
# print((todayPred))

# print(len(predictions))

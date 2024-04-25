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
import mlflow

yf.pdr_override()

# For time stamps
from datetime import datetime, timedelta


def makeModel(trainStock, trainStartDate):
    mlflow.autolog()
    df2 = pdr.get_data_yahoo(trainStock, start=trainStartDate, end=datetime.now())
    # Show the data
    # Create a new dataframe with only the 'Close column
    data = df2.filter(["Close"])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) * 0.95))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0 : int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60 : i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:
            print("61")

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model with mlflow
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(f"{trainStock}_{trainStartDate}_experiment")
    with mlflow.start_run() as run:
        model.fit(
            x_train,
            y_train,
            batch_size=1,
            epochs=2,
            callbacks=[mlflow.keras.MlflowCallback()],
        )

        # model.fit(x_train, y_train, batch_size=1, epochs=3)
        test_data = scaled_data[training_data_len - 60 :, :]
        # Create the data sets x_test and y_test
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60 : i, 0])

        # Convert the data to a numpy array
        x_test = np.array(x_test)

        # Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Get the models predicted price values
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Get the root mean squared error (RMSE)
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        mlflow.log_metric("rmse", rmse)

        model.save(
            rf".\models\{trainStock}.keras",
        )
        joblib.dump(
            scaler,
            rf".\models\{trainStock}.gz",
        )

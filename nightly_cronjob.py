from lstm_preds import getPrediction
from lstm_model import makeModel
from createModels import cronModels
import sqlite3
from datetime import datetime, timedelta, date
import yfinance as yf

stock_list = [
    "TSLA",
    "AMD",
    "GRAB",
    "NVDA",
    "INTC",
    "AAPL",
    "AAL",
    "SOFI",
    "RIVN",
    "MU",
    "AMZN",
    "MSFT",
    "CSX",
]


def add_prediction(conn, prediction):
    sql = """ INSERT INTO prediction(date, company, predicted, actual, error)
              VALUES(?,?,?,?,?) """
    cur = conn.cursor()
    cur.execute(sql, prediction)
    conn.commit()
    return cur.lastrowid


if __name__ == "__main__":
    predCon = sqlite3.connect("daily_pred.db")
    cur = predCon.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS prediction(date, company, predicted, actual, error)"
    )
    # For every stock in the stock list, compare yesterday's prediction with today's price. Put that in a table
    for i in stock_list:
        toAdd = []
        toAdd.append(date.today())
        toAdd.append(i)
        pred = getPrediction(i, True)[0][0]
        toAdd.append(pred)
        tickerData = yf.Ticker(i)
        todayClose = tickerData.history(period="1d")["Close"][0]
        actualValue = todayClose
        toAdd.append(actualValue)
        toAdd.append((actualValue - pred) / actualValue)

        prediction_id = add_prediction(predCon, toAdd)
    cronModels()

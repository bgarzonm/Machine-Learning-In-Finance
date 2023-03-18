import pandas as pd
import yfinance as yf
import sys

sys.path.append(r"ML_finance_python\utility")
import plot_settings


data = yf.download(tickers="BTC-USD", period="10mo", interval="1d")
data.to_csv(r"ML_finance_python\data\BTC-USD.csv")


data = data[["Adj Close", "Volume"]]

data.describe()
data["Adj Close"].plot(label="Bitcoin", legend=True)

data['Adj Close'].pct_change().plot.hist(bins=50)
plt.title('adjusted close 1-day percent change')
plt.show()


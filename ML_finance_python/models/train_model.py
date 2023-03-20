# Importing required libraries
import numpy as np                      # For numerical calculations
import pandas as pd                     # For data manipulation and analysis
import matplotlib.pyplot as plt         # For data visualization
import statsmodels.api as sm            # For statistical modeling and analysis
from statsmodels.regression.linear_model import OLS   # For Ordinary Least Squares regression analysis
import sys
sys.path.append('../../ML_finance_python/utility')
import plot_settings

df = pd.read_csv('../../ML_finance_python/data/BTC-USD.csv', index_col='Date', parse_dates=True)
df.info()
# Calculate the typical price
df['Typical Price'] = (df['High'] + df['Low'] + df['Close']) / 3

# Calculate the Volume-Weighted Average Price (VWAP)
df['Volume * Typical Price'] = df['Typical Price'] * df['Volume']
df['Cumulative Volume * Typical Price'] = df['Volume * Typical Price'].cumsum()
df['Cumulative Volume'] = df['Volume'].cumsum()
df['VWAP'] = df['Cumulative Volume * Typical Price'] / df['Cumulative Volume']

# Calculate the On-Balance Volume (OBV)
df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), 
                               df['Volume'], 
                               np.where(df['Close'] < df['Close'].shift(1), 
                                        -df['Volume'], 
                                        0)).cumsum()



plt.scatter(x=df['OBV'], y=df['VWAP'], alpha=0.1, color='darkblue')

# Add axis labels and a title
plt.xlabel('On-Balance Volume (OBV)')
plt.ylabel('Volume-Weighted Average Price (VWAP)')
plt.title(' VWAP vs. OBV')
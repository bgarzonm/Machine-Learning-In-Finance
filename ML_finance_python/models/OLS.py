import numpy as np                      # For numerical calculations
import pandas as pd                     # For data manipulation and analysis
import matplotlib.pyplot as plt         # For data visualization
import statsmodels.api as sm            # For statistical modeling and analysis
from statsmodels.regression.linear_model import OLS   # For Ordinary Least Squares regression analysis
import seaborn as sns                   # For data visualization
import talib                            # For technical analysis
import sys                              # For system-specific parameters and functions
sys.path.append('../../ML_finance_python/utility')
import plot_settings                    # Plot settings

# Load the data
df = pd.read_csv('../../ML_finance_python/data/BTC-USD.csv', index_col='Date', parse_dates=True)

# CORRELATION
def calculate_correlation(df):
    # Create 5-day % changes of Adj Close for the current day, and 5 days in the future
    df['5d_future_close'] = df['Adj Close'].shift(-5)
    df['5d_close_future_pct'] = df['5d_future_close'].pct_change(5)
    df['5d_close_pct'] = df['Adj Close'].pct_change(5)

    # Calculate the correlation matrix between the 5d close percentage changes (current and future)
    corr = df[['5d_close_pct', '5d_close_future_pct']].corr()

    # Scatter the current 5-day percent change vs the future 5-day percent change
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(df['5d_close_pct'], df['5d_close_future_pct'])
    plt.show()

    return corr

calculate_correlation(df)

feature_names = ['5d_close_pct']  # a list of the feature names for later

# Create moving averages and rsi for timeperiods of 14, 30, 50, and 200
for n in [14, 30, 50, 200]:

    # Create the moving average indicator and divide by Adj_Close
    df['ma' + str(n)] = talib.SMA(df['Adj Close'].values,
                              timeperiod=n) / df['Adj Close']
    # Create the RSI indicator
    df['rsi' + str(n)] = talib.RSI(df['Adj Close'].values, timeperiod=n)
    
    # Add rsi and moving average to the feature name list
    feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]
    
print(feature_names)

# Drop all na values
df = df.dropna()

# Create features and targets
# use feature_names for features; '5d_close_future_pct' for targets
features = df[feature_names]
targets = df['5d_close_future_pct']

# Create DataFrame from target column and feature columns
feature_and_target_cols = ['5d_close_future_pct'] + feature_names
feat_targ_df = df[feature_and_target_cols]

# Calculate correlation matrix
corr = feat_targ_df.corr()
# Plot heatmap of correlation matrix
fig = plt.figure(figsize=(10, 8))

sns.heatmap(corr , annot= True, annot_kws = {"size": 10})
plt.yticks(rotation=0, size = 14); plt.xticks(rotation=90, size = 14)  # fix ticklabel directions and size
plt.tight_layout()  # fits plot area to the plot, "tightly"
plt.show()  # show the plot

plt.scatter(df['rsi30'], df['ma50'])
plt.show()

# Add a constant to the features
linear_features = sm.add_constant(features)

# Create a size for the training set that is 85% of the total number of samples
train_size = int(0.85 * targets.shape[0])
train_features = linear_features[:train_size]
train_targets = targets[:train_size]
test_features = linear_features[train_size:]
test_targets = targets[train_size:]
print(linear_features.shape, train_features.shape, test_features.shape)

# Create the linear model and complete the least squares fit
model = sm.OLS(train_targets, train_features)
results = model.fit()  # fit the model
print(results.summary())

# examine pvalues
# Features with p <= 0.05 are typically considered significantly different from 0
print(results.pvalues)

# Make predictions from our model for train and test sets
train_predictions = results.predict(train_features)
test_predictions = results.predict(test_features)

# Evaluate our results
fig = plt.figure(figsize=(10, 8))
# Scatter the predictions vs the targets with 20% opacity
plt.scatter(train_predictions, train_targets, alpha=0.2, color='b', label='train')
plt.scatter(test_predictions, test_targets, alpha = 0.2, color='r', label='test')
# Plot the perfect prediction line
xmin, xmax = plt.xlim()
plt.plot(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01), c='k')

# Set the axis labels and show the plot
plt.xlabel('predictions')
plt.ylabel('actual')
plt.legend()
plt.show()


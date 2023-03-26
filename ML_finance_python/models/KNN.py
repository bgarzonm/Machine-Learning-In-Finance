import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import talib
import sys 

sys.path.append("../../ML_finance_python/utility")
import plot_settings  
import random

df = pd.read_csv(
    "../../ML_finance_python/data/apple.csv", index_col="Date", parse_dates=True
)

# CORRELATION
def calculate_correlation(df):
    # Create 5-day % changes of Adj Close for the current day, and 5 days in the future
    df["5d_future_close"] = df["Adj Close"].shift(-5)
    df["5d_close_future_pct"] = df["5d_future_close"].pct_change(5)
    df["5d_close_pct"] = df["Adj Close"].pct_change(5)

    # Calculate the correlation matrix between the 5d close percentage changes (current and future)
    corr = df[["5d_close_pct", "5d_close_future_pct"]].corr()

    # Scatter the current 5-day percent change vs the future 5-day percent change
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(df["5d_close_pct"], df["5d_close_future_pct"])
    plt.show()

    return corr


calculate_correlation(df)
feature_names = ["5d_close_pct"]  # a list of the feature names for later

# Create moving averages and rsi for timeperiods of 14, 30, 50, and 200
for n in [14, 30, 50, 200]:

    # Create the moving average indicator and divide by Adj_Close
    df["ma" + str(n)] = (
        talib.SMA(df["Adj Close"].values, timeperiod=n) / df["Adj Close"]
    )
    # Create the RSI indicator
    df["rsi" + str(n)] = talib.RSI(df["Adj Close"].values, timeperiod=n)

    # Add rsi and moving average to the feature name list
    feature_names = feature_names + ["ma" + str(n), "rsi" + str(n)]

print(feature_names)

# Drop all na values
df = df.dropna()

# Create 2 new volume features, 1-day % change and 5-day SMA of the % change
# here the data don't have a volume column, so I use volumen instead
# new_features = ['Adj Volume_1d_change', 'Adj Volume_1d_change_SMA']
# feature_names.extend(new_features)
# df['Adj Volume_1d_change'] = df['Adj Volume'].pct_change()
# df['Adj Volume_1d_change_SMA'] = talib.SMA(df['Adj Volume_1d_change'].values,timeperiod=5)


new_features = ["Volume_1d_change", "Volume_1d_change_SMA"]
feature_names.extend(new_features)
df["Volume_1d_change"] = df["Volume"].pct_change()
df["Volume_1d_change_SMA"] = talib.SMA(df["Volume_1d_change"].values, timeperiod=5)
df = df.dropna()
# Plot histogram of volume % change data
df[new_features].plot(kind="hist", sharex=False, bins=50)
plt.show()

# Create day-of-week features

# Use pandas' get_dummies function to get dummies for day of the week
days_of_week = pd.get_dummies(df.index.dayofweek, prefix="weekday", drop_first=True)

# Set the index as the original dataframe index for merging
days_of_week.index = df.index

# Join the dataframe with the days of week dataframe
df = pd.concat([df, days_of_week], axis=1)

# Add days of week to feature names
feature_names.extend([f"weekday_{str(i)}" for i in range(1, 5)])
df.dropna(inplace=True)  # drop missing values in-place
df.head()

# Examine correlations of the new features

# Add the weekday labels to the new_features list
new_features.extend([f"weekday_{str(i)}" for i in range(1, 5)])

# Plot the correlations between the new features and the targets
fig = plt.figure(figsize=(10, 8))
sns.heatmap(df[new_features + ["5d_close_future_pct"]].corr(), annot=True)
plt.yticks(rotation=0)  # ensure y-axis ticklabels are horizontal
plt.xticks(rotation=90)  # ensure x-axis ticklabels are vertical
plt.tight_layout()
plt.show()

# define features
# use feature_names for features; '5d_close_future_pct' for targets

features = df[feature_names]
targets = df["5d_close_future_pct"]
# Add a constant to the features

# Create a size for the training set that is 85% of the total number of samples
train_size = int(0.85 * targets.shape[0])
train_features = features[:train_size]
train_targets = targets[:train_size]
test_features = features[train_size:]
test_targets = targets[train_size:]
print(features.shape, train_features.shape, test_features.shape)

from sklearn.preprocessing import scale

# Remove unimportant features (weekdays)
train_features = train_features.iloc[:, :-4]
test_features = test_features.iloc[:, :-4]

# Standardize the train and test features
scaled_train_features = scale(train_features)
scaled_test_features = scale(test_features)

# Plot histograms of the 14-day SMA RSI before and after scaling
f, ax = plt.subplots(nrows=2, ncols=1)
train_features.iloc[:, 2].hist(ax=ax[0])
ax[1].hist(scaled_train_features[:, 2])
plt.show()


# optimize the number of neighbors
from sklearn.neighbors import KNeighborsRegressor

for n in range(2, 13):
    # Create and fit the KNN model
    knn = KNeighborsRegressor(n_neighbors=n)
    
    # Fit the model to the training data
    knn.fit(scaled_train_features, train_targets)
    
    # Print number of neighbors and the score to find the best value of n
    print("n_neighbors =", n)
    print('train, test scores')
    print(knn.score(scaled_train_features, train_targets))
    print(knn.score(scaled_test_features, test_targets))
    print()  # prints a blank line
    
    
# Evaluate KNN performance
# Create the model with the best-performing n_neighbors of 5
knn = KNeighborsRegressor(n_neighbors=5)

# Fit the model
knn.fit(scaled_train_features, train_targets)

# Get predictions for train and test sets
train_predictions = knn.predict(scaled_train_features)
test_predictions = knn.predict(scaled_test_features)

# Plot the actual vs predicted values
plt.scatter(train_predictions, train_targets, label='train')
plt.scatter(test_predictions, test_targets, label='test')
plt.legend()
plt.show()











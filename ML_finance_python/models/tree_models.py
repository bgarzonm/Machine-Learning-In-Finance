import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm            # For statistical modeling and analysis
import sys                              # For system-specific parameters and functions
sys.path.append('../../ML_finance_python/utility')
import plot_settings                    # Plot settings
import random
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

# Create 2 new volume features, 1-day % change and 5-day SMA of the % change
# here the data don't have a volume column, so I use volumen instead
# new_features = ['Adj Volume_1d_change', 'Adj Volume_1d_change_SMA']
# feature_names.extend(new_features)
# df['Adj Volume_1d_change'] = df['Adj Volume'].pct_change()
# df['Adj Volume_1d_change_SMA'] = talib.SMA(df['Adj Volume_1d_change'].values,timeperiod=5)


new_features = ['Volume_1d_change', 'Volume_1d_change_SMA']
feature_names.extend(new_features)
df['Volume_1d_change'] = df['Volume'].pct_change()
df['Volume_1d_change_SMA'] = talib.SMA(df['Volume_1d_change'].values,timeperiod=5)
df = df.dropna()
# Plot histogram of volume % change data
df[new_features].plot(kind='hist', sharex=False, bins=50)
plt.show()

# Create day-of-week features

# Use pandas' get_dummies function to get dummies for day of the week
days_of_week = pd.get_dummies(df.index.dayofweek,
                              prefix='weekday',
                              drop_first=True)

# Set the index as the original dataframe index for merging
days_of_week.index = df.index

# Join the dataframe with the days of week dataframe
df = pd.concat([df, days_of_week], axis=1)

# Add days of week to feature names
feature_names.extend([f'weekday_{str(i)}' for i in range(1, 5)])
df.dropna(inplace=True)  # drop missing values in-place
df.head()

# Examine correlations of the new features

# Add the weekday labels to the new_features list
new_features.extend([f'weekday_{str(i)}' for i in range(1, 5)])

# Plot the correlations between the new features and the targets
fig = plt.figure(figsize=(10, 8))
sns.heatmap(df[new_features + ['5d_close_future_pct']].corr(), annot=True)
plt.yticks(rotation=0)  # ensure y-axis ticklabels are horizontal
plt.xticks(rotation=90)  # ensure x-axis ticklabels are vertical
plt.tight_layout()
plt.show()

# define features 
# use feature_names for features; '5d_close_future_pct' for targets

features = df[feature_names]
targets = df['5d_close_future_pct']
# Add a constant to the features

# Create a size for the training set that is 85% of the total number of samples
train_size = int(0.85 * targets.shape[0])
train_features = features[:train_size]
train_targets = targets[:train_size]
test_features = features[train_size:]
test_targets = targets[train_size:]
print(features.shape, train_features.shape, test_features.shape)

# Create a decision tree regression model with default arguments

decision_tree = DecisionTreeRegressor()

# Fit the model to the training features and targets
decision_tree.fit(train_features, train_targets)

# Check the score on train and test
print(decision_tree.score(train_features, train_targets))
print(decision_tree.score(test_features, test_targets))

# Loop through a few different max depths and check the performance

for d in [3, 5, 10]:
    # Create the tree and fit it
    decision_tree = DecisionTreeRegressor(max_depth=d)
    decision_tree.fit(train_features, train_targets)

    # Print out the scores on train and test
    print('max_depth=', str(d))
    print(decision_tree.score(train_features, train_targets))
    print(decision_tree.score(test_features, test_targets), '\n')

decision_tree = DecisionTreeRegressor(max_depth=5)
decision_tree.fit(train_features, train_targets)

# Predict values for train and test
train_predictions = decision_tree.predict(train_features)
test_predictions = decision_tree.predict(test_features)

# Scatter the predictions vs actual values
plt.scatter(train_predictions, train_targets, label='train')
plt.scatter(test_predictions, test_targets, label='test')
plt.show()


from sklearn.ensemble import RandomForestRegressor

# Create the random forest model and fit to the training data
rfr = RandomForestRegressor(n_estimators=200)
rfr.fit(train_features, train_targets)

# Look at the R^2 scores on train and test
print(rfr.score(train_features, train_targets))
print(rfr.score(test_features, test_targets))

from sklearn.model_selection import ParameterGrid

# Create a dictionary of hyperparameters to search
grid = {'n_estimators': [200],
        'max_depth': [3],
        'max_features': [4, 8],
        'random_state': [42]}
test_scores = []

# Loop through the parameter grid, set the hyperparameters, and save the scores
for g in ParameterGrid(grid):
    rfr.set_params(**g)  # ** is "unpacking" the dictionary
    rfr.fit(train_features, train_targets)
    test_scores.append(rfr.score(test_features, test_targets))

# Find best hyperparameters from the test score and print
best_idx = np.argmax(test_scores)
print(test_scores[best_idx], ParameterGrid(grid)[best_idx])

# Use the best hyperparameters from before to fit a random forest model
rfr = RandomForestRegressor(n_estimators=200, max_depth=3, max_features=8, random_state=42)
rfr.fit(train_features, train_targets)

# Make predictions with our model
train_predictions = rfr.predict(train_features)
test_predictions = rfr.predict(test_features)

# Create a scatter plot with train and test actual vs predictions
plt.scatter(train_targets, train_predictions, label='train')
plt.scatter(test_targets, test_predictions, label='test')
plt.legend()
plt.show()

# Get feature importances from our random forest model
importances = rfr.feature_importances_

# Get the index of importances from greatest importance to least
sorted_index = np.argsort(importances)[::-1]
x = range(len(importances))

# Create tick labels 
labels = np.array(feature_names)[sorted_index]
plt.bar(x, importances[sorted_index], tick_label=labels)

# Rotate tick labels to vertical
plt.xticks(rotation=90)
plt.show()
# we conclude thta the weeks are useless for this dataset

# Now we are going to use Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

# Create GB model -- hyperparameters have already been searched for you
gbr = GradientBoostingRegressor(max_features=4,
                                learning_rate=0.01,
                                n_estimators=200,
                                subsample=0.6,
                                random_state=17)
gbr.fit(train_features, train_targets)

print(gbr.score(train_features, train_targets))
print(gbr.score(test_features, test_targets))


# Extract feature importances from the fitted gradient boosting model
feature_importances = gbr.feature_importances_

# Get the indices of the largest to smallest feature importances
sorted_index = np.argsort(feature_importances)[::-1]
x = range(features.shape[1])

# Create tick labels 
labels = np.array(feature_names)[sorted_index]

plt.bar(x, feature_importances[sorted_index], tick_label=labels)

# Set the tick lables to be the feature names, according to the sorted feature_idx
plt.xticks(rotation=90)
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA


apple = pd.read_csv("../data/apple.csv")
apple = apple[["Date", "Close"]]

apple.set_index("Date", inplace=True)

# Calculate the first difference and drop the nans
apple_diff = apple.diff()
apple_diff = apple_diff.dropna()

fig, ax = plt.subplots()
apple_diff.plot(ax=ax)
plt.show()

# Run test and print
result_diff = adfuller(apple_diff["Close"])
print(result_diff)

# Calculate log-return and drop nans
apple_log = np.log(apple / apple.shift(1))
apple_log = apple_log.dropna()

fig, ax = plt.subplots()
apple_log.plot(ax=ax)
plt.show()

# Run test and print
result_log = adfuller(apple_log["Close"])
print(result_log)


# -----------------------------------------------------
# Forecasting with ARIMA
# -----------------------------------------------------

model = ARIMA(apple["Close"], order=(1, 1, 1))

# Fit the model
results = model.fit()

# Print summary
print(results.summary())

# Generate predictions
one_step_forecast = results.get_prediction(start=-30)

# Extract prediction mean
mean_forecast = one_step_forecast.predicted_mean

# Get confidence intervals of predictions
confidence_intervals = one_step_forecast.conf_int()

# Select lower and upper confidence limits
lower_limits = confidence_intervals.loc[:, "lower Close"]
upper_limits = confidence_intervals.loc[:, "upper Close"]

# Print best estimate predictions
print(mean_forecast)


# plot the amazon data

plt.plot(apple.index, apple, label="observed")
plt.plot(mean_forecast.index, mean_forecast, color="r", label="forecast")

# shade the area between your confidence limits
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color="pink")
# set labels, legends and show plot
plt.xlabel("Date")
plt.ylabel("apple Stock Price - Close USD")
plt.legend()
plt.show()


# Generate predictions
dynamic_forecast = results.get_prediction(start=-30, dynamic=True)

# Extract prediction mean
mean_forecast = dynamic_forecast.predicted_mean

# Get confidence intervals of predictions
confidence_intervals = dynamic_forecast.conf_int()
lower_limits = confidence_intervals.loc[:, "lower Close"]
upper_limits = confidence_intervals.loc[:, "upper Close"]

# Print best estimate predictions
print(mean_forecast)

# plot the amazon data

plt.plot(apple.index, apple, label="observed")
plt.plot(mean_forecast.index, mean_forecast, color="r", label="forecast")
# shade the area between your confidence limits
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color="pink")
# set labels, legends and show plot
plt.xlabel("Date")
plt.ylabel("Amazon Stock Price - Close USD")
plt.legend()
plt.show()


# Take the first difference of the data

apple_diff = apple.diff().dropna()

# Create ARMA(2,2) model
arma = ARIMA(apple_diff, order=(2, 0, 2))

# Fit model
arma_results = arma.fit()

# Print fit summary
print(arma_results.summary())


# Make arma forecast of next 10 differences

arma_diff_forecast = arma_results.get_forecast(steps=10).predicted_mean

# Integrate the difference forecast
arma_int_forecast = np.cumsum(arma_diff_forecast)

# Make absolute value forecast
arma_value_forecast = arma_int_forecast + apple.iloc[-1, 0]

# Print forecast
print(arma_value_forecast)


# Create ARIMA(2,1,2) model

arima = ARIMA(apple, order=(2, 1, 2))

# Fit ARIMA model
arima_results = arima.fit()

# Make ARIMA forecast of next 10 values
arima_value_forecast = arima_results.get_forecast(steps=10).predicted_mean

# Print forecast
print(arima_value_forecast)


# Plot the original time series
plt.plot(apple.index, apple["Close"], label="Original")
# Plot the ARIMA forecasted values
plt.plot(arima_value_forecast.index, arima_value_forecast, label="ARIMA Forecast")

# Add a legend and axis labels
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")

# Show the plot
plt.show()

# -----------------------------------------------------
# ACF and PACF
# -----------------------------------------------------

# Import
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot the ACF of df
plot_acf(apple, lags=10, zero=False, ax=ax1)

# Plot the PACF of df
plot_pacf(apple, lags=10, zero=False, ax=ax2)

plt.show()


# Create empty list to store search results
order_aic_bic = []

# Loop over p values from 0-2
for p in range(3):
    # Loop over q values from 0-2
    for q in range(3):
        # create and fit ARMA(p,q) model
        model = ARIMA(apple, order=(p, 0, q))
        results = model.fit()

        # Append order and results tuple
        order_aic_bic.append((p, q, results.aic, results.bic))

# Construct DataFrame from order_aic_bic
order_df = pd.DataFrame(order_aic_bic, columns=["p", "q", "AIC", "BIC"])

# Print order_df in order of increasing AIC
print(order_df.sort_values("AIC"))

# Print order_df in order of increasing BIC
print(order_df.sort_values("BIC"))


# Loop over p values from 0-2
for p in range(3):
    # Loop over q values from 0-2
    for q in range(3):

        try:
            # create and fit ARMA(p,q) model
            model = ARIMA(apple, order=(p, 0, q))
            results = model.fit()

            # Print order and results
            print(p, q, results.aic, results.bic)

        except:
            print(p, q, None, None)


# -----------------------------------------------------
# Model diagnostics
# -----------------------------------------------------

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf


# Plot autocorrelation function of residuals
plot_acf(results.resid, lags=20)
plt.show()

# Plot partial autocorrelation function of residuals
plot_pacf(results.resid, lags=20)
plt.show()


results.plot_diagnostics(figsize=(15, 12))
plt.show()


# -----------------------------------------------------
# Seasonal time series
# -----------------------------------------------------

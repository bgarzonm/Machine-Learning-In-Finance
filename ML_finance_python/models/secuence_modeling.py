import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
import sys  # For system-specific parameters and functions

sys.path.append("../../ML_finance_python/utility")
import plot_settings

df = pd.read_excel(
    "../../ML_finance_python/data/Total_Share_prices.xls",
    index_col=0,
    skiprows=range(10),
)


def run_time_series_analysis(title, start_date):
    
    ts = pd.Series(
        df.iloc[:,0].values, index=pd.date_range(start=start_date, periods=len(df), freq="QS")
    )

    plt.plot(ts.index, ts.values)
    plt.title(f"{title}")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.show()

    lags = 25
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    plot_acf(
        ts,
        ax=ax[0],
        lags=lags,
        alpha=0.05,
        title=f"ACF {title}",
        linewidth=2,
    )
    plot_pacf(
        ts,
        ax=ax[1],
        lags=lags,
        alpha=0.05,
        title=f"PACF {title}",
        linewidth=2,
    )
    plt.tight_layout()
    plt.show()

    result = adfuller(ts)
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print("\t%s: %.3f" % (key, value))


run_time_series_analysis(title="Total Share prices", start_date="1960-01-01")


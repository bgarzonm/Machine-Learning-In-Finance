from fredapi import Fred
import pandas as pd

with open("api_key.txt", "r") as f:
    api_key = f.read().strip()
    
fred = Fred(api_key= api_key)

cpi = "CPIAUCSL"
df = fred.get_series(cpi)
df.to_csv("../../ML_finance_python/data/CPI_US.csv")


import matplotlib.pyplot as plt
plt.plot(df)
plt.title("US Consumer Price Index")
plt.xlabel("Year")
plt.ylabel("Index")
plt.show()


from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt

with open("api_key.txt", "r") as f:
    api_key = f.read().strip()
    
fred = Fred(api_key= api_key)

#cpi united states
cpi = "CPIAUCSL"
df = fred.get_series(cpi)
df.to_csv("../../ML_finance_python/data/CPI_US.csv")

# plot the data
plt.plot(df)
plt.title("US Consumer Price Index")
plt.xlabel("Year")
plt.ylabel("Index")
plt.show()



# Residential Property Prices for Colombia 
rrpc = 'QCON628BIS'
df_2 = fred.get_series(rrpc)
df_2.to_csv("../../ML_finance_python/data/Residential_Property_Prices_Colombia.csv")

# plot the data
plt.plot(df_2)
plt.title("Residential Property Prices for Colombia ")
plt.xlabel("Year")
plt.ylabel("Index")
plt.show()
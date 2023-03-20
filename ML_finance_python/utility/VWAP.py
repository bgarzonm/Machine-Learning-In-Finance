class VWAPCalculator:
    
    def __init__(self, df):
        self.df = df.copy()
    
    def calculate_vwap(self):
        self.df['Typical Price'] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['Volume * Typical Price'] = self.df['Typical Price'] * self.df['Volume']
        self.df['Cumulative Volume * Typical Price'] = self.df['Volume * Typical Price'].cumsum()
        self.df['Cumulative Volume'] = self.df['Volume'].cumsum()
        self.df['VWAP'] = self.df['Cumulative Volume * Typical Price'] / self.df['Cumulative Volume']
        return self.df['VWAP'].iloc[-1]
    
    def calculate_obv(self):
        self.df['OBV'] = np.where(self.df['Close'] > self.df['Close'].shift(1), 
                                  self.df['Volume'], 
                                  np.where(self.df['Close'] < self.df['Close'].shift(1), 
                                           -self.df['Volume'], 
                                           0)).cumsum()
        return self.df['OBV'].iloc[-1]
    
    def plot_vwap_vs_obv(self):
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(x=self.df['OBV'], y=self.df['VWAP'], alpha=1, color='darkblue')
        plt.xlabel('On-Balance Volume (OBV)')
        plt.ylabel('Volume-Weighted Average Price (VWAP)')
        plt.title('VWAP vs. OBV')
        plt.show()
    
df = pd.read_csv('../../ML_finance_python/data/BTC-USD.csv', index_col='Date', parse_dates=True)
df.info()
# VWAP
vwap_calc = VWAPCalculator(df)
vwap = vwap_calc.calculate_vwap()
obv = vwap_calc.calculate_obv()
vwap_calc.plot_vwap_vs_obv()

print("VWAP:", vwap)
print("OBV:", obv)
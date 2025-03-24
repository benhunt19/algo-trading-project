from src.forecastModelBase import ForecastModel
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
from src.globals import SPTL_DATA_PATH

class GarchModel(ForecastModel):
    """
    Description:
        ARCH (Autoregressive Conditional Heteroskedasticity) based model built upon arch library
        
    Parameters:
        data (float[]): History of timeseries data to base forecast upon
        timeseries (datetime[]): Timeseries index for the data
        p (int): Order of the lag of the squared residuals
        q (int): Order of the lag of the conditional variance
    """
    def __init__(self, data, timeseries, p=2, q=2) -> None:
        # Initialize the parent class
        super().__init__(data=data, timeseries=timeseries) # self.data, self.timeseries, self.results, self.forecastData, self.name
        self.name = 'ARCH'
        
        self.p = p
        self.q = q
        self.model = arch_model(self.data, vol='ARCH', p=self.p, q=self.q)
        
        # Initialize with a fit
        self.fitModel()
        
    def fitModel(self) -> pd.Series:
        """
        Description:
            Fit model using arch inbuilt fitting algo
        """
        self.results = self.model.fit(disp='off')
        return self.results
        
    def runChecks(self):
        """
        Description:
            Perform statistical tests to find best parameters to use
        """
        # 1. Check for stationarity using ADF test
        def adf_test(series):
            result = adfuller(series)
            print(f"ADF Statistic: {result[0]}")
            print(f"p-value: {result[1]}")
            return result[1] < 0.05  # If True, series is stationary

        print("Before differencing:")
        is_data_stationary = adf_test(self.data)
        print(is_data_stationary)
        
        # 2. Differencing if necessary
        if not is_data_stationary:
            time_series_diff = self.data.diff().dropna()
            print("\nAfter differencing:")
            is_differenced_stationary = adf_test(time_series_diff)
            print(is_differenced_stationary)

        # 3. Identify ARCH(p,q) orders using ACF and PACF
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(time_series_diff, ax=axes[0])
        plot_pacf(time_series_diff, ax=axes[1])
        plt.show()
        
    def forecast(self, steps=10) -> pd.Series:
        """
        Description:
            Forecast using historical data based on arch inbuilt forecasting algo
        """
        self.forecastData = self.results.forecast(horizon=steps)
        return self.forecastData.mean.iloc[-1]
    
    def __str__(self) -> str:
        return f"ARCH Model: p({self.p}), q({self.q})"

## Remove once built
if __name__ == "__main__":
    
    train_pct = 80
    
    data = pd.read_csv(SPTL_DATA_PATH)
    # print(data)
    data_length = len(data)
    
    p = 2
    q = 2
    
    model1 = GarchModel(
        data=data['Close'],
        timeseries=data['Date'],
        p=p,
        q=q
    )
    
    # model1.fitModel()
    
    f = model1.forecast()
    print(f)
    
    model1.runChecks()
    
    model1.plot()
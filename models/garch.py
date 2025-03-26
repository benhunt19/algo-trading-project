from models.baseModel import ForecastModel
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from src.globals import SPTL_DATA_PATH
import seaborn as sns
from statsmodels.tsa.stattools import q_stat
from statsmodels.stats.diagnostic import het_arch
import numpy as np
class GarchModel(ForecastModel):
    """
    Description:
        GARCH (Generalized Autoregressive Conditional Heteroskedasticity) based model built upon arch library
        
    Parameters:
        data (float[]): History of timeseries data to base forecast upon
        timeseries (datetime[]): Timeseries index for the data
        p (int): Order of the GARCH terms (α)
        q (int): Order of the ARCH terms (β)
    """
    def __init__(self, data, timeseries, p=2, q=2, lookForwardOverride=None) -> None:
        super().__init__(data=data, timeseries=timeseries) # self.data, self.timeseries, self.results, self.forecastData, self.name
        self.name = 'GARCH'

        self.p = p
        self.q = q
        self.model = arch_model(self.data, vol='Garch', p=self.p, q=self.q)
        self.lookForwardOverride = lookForwardOverride
        
        # Initialize with a fit
        self.fitModel()
        
    def fitModel(self) -> pd.Series:
        """
        Description:
            Fit model using arch inbuilt fitting algo
        """
        self.results = self.model.fit(disp='off')
        return self.results
        
    def runChecks(self) -> None:
        """
        Perform statistical tests to determine the best parameters for a GARCH model.
        """
        
        # 1. Convert to log returns (if needed)
        self.data = np.log(self.data).diff().dropna()

        # 2. Check stationarity with ADF test
        def adf_test(series):
            result = adfuller(series)
            print(f"ADF Statistic: {result[0]}")
            print(f"p-value: {result[1]}")
            return result[1] < 0.05  # If True, series is stationary
        
        print("Checking stationarity of log returns:")
        is_data_stationary = adf_test(self.data)
        
        # 3. Check for ARCH effects using Engle's test
        arch_test_pval = het_arch(self.data)[1]
        print(f"\nEngle's ARCH test p-value: {arch_test_pval}")
        
        if arch_test_pval > 0.05:
            print("No significant ARCH effect detected. GARCH modeling might not be necessary.")
    
        # 4. Compare different GARCH orders using AIC/BIC
        print("\nComparing different GARCH orders:")
        orders = [(1,1), (1,2), (2,1), (2,2)]
        results = {}
        
        for p, q in orders:
            try:
                model = arch_model(self.data, vol='Garch', p=p, q=q)
                fitted = model.fit(disp='off')
                results[(p,q)] = {
                    'AIC': fitted.aic,
                    'BIC': fitted.bic
                }
                print(f"GARCH({p},{q}): AIC={fitted.aic:.2f}, BIC={fitted.bic:.2f}")
            except:
                print(f"GARCH({p},{q}) failed to converge")

        # 5. Plot ACF and PACF
        squared_returns = self.data ** 2

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.set_style("darkgrid")
        plot_acf(self.data, ax=axes[0])
        axes[0].set_title('ACF of Returns')
        axes[0].set_xlabel('Lags')
        axes[0].set_ylabel('Correlation')
        axes[0].legend(['ACF'])
        
        plot_pacf(self.data, ax=axes[1])
        axes[1].set_title('PACF of Returns')
        axes[1].set_xlabel('Lags')
        axes[1].set_ylabel('Correlation')
        axes[1].legend(['PACF'])
        
        plt.close()
        
        
    def forecast(self, steps=10) -> pd.Series:
        """
        Description:
            Forecast using historical data based on arch inbuilt forecasting algo
        """
        self.forecastData = self.results.forecast(horizon=steps)
        return self.forecastData.mean.iloc[-1]
        
    # Review - turned off as data format is different
    def plot(self) -> None:
        pass
    
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
    
    f = model1.forecast()
    print(f)
    
    model1.runChecks()
    
    model1.plot()
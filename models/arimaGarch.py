from models.baseModel import ForecastModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic, adfuller, kpss
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ModelWarning
from arch import arch_model  # Import for GARCH modeling
from src.globals import SPTL_DATA_PATH

class ArimaGarchModel(ForecastModel):
    """
    Description:
        ARIMA (AutoRegressive Integrated Moving Average) and GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model.
        
    Parameters:
        data (float[]): History of timeseries data to base forecast upon
        timeseries (datetime[]): Timeseries index for the data
        AR_order (int): Auto Regressive look back window, AR(1), AR(2) etc...
        differencing_order (int): Differencing (integrating) order
        MA_order (int): Moving Average lookback window MA(1), MA(2) etc...
    """
    
    def __init__(self, data, timeseries, AR_order=1, differencing_order=1, MA_order=1, lookForwardOverride=None) -> None:
        super().__init__(data=data, timeseries=timeseries)
        self.name = 'ARIMA-GARCH'
        
        self.AR_order = AR_order
        self.differencing_order = differencing_order
        self.MA_order = MA_order
        self.model = ARIMA(self.data, order=(self.AR_order, self.differencing_order, self.MA_order))
        self.selectionInfo = None
        self.lookForwardOverride = lookForwardOverride
        
        # Fit the model initially
        self.fitModel()
        
    def fitModel(self) -> pd.Series:
        """
        Description:
            Fit the ARIMA model first, then fit the GARCH model to the residuals
        """
        self.results = self.model.fit()  # Fit ARIMA model
        self.residuals = self.results.resid  # Extract residuals
        self.garch_model = arch_model(self.residuals, vol='Garch', p=1, q=1)  # Fit GARCH(1,1) model to residuals
        self.garch_results = self.garch_model.fit()  # Fit the GARCH model
        return self.results, self.garch_results
        
    def runChecks(self):
        """
        Description:
            Perform statistical tests to find best parameters to use
        """
        def adf_test(series):
            adf_result = adfuller(series)
            print("=== ADF Test ===")
            print(f"ADF Statistic: {adf_result[0]}")
            print(f"p-value: {adf_result[1]}")
            kpss_result = kpss(series)
            print("\n=== KPSS Test ===")
            print(f"KPSS Statistic: {kpss_result[0]}")
            print(f"p-value: {kpss_result[1]}")
            return adf_result[1] < 0.05 and kpss_result[1] > 0.05

        print("Before differencing:")
        is_data_stationary = adf_test(self.data)
        print(is_data_stationary)
        
        # Differencing if necessary
        if not is_data_stationary:
            time_series_diff = self.data.diff().dropna()
            print("\nAfter differencing:")
            is_differenced_stationary = adf_test(time_series_diff)
            print(is_differenced_stationary)

        # ACF and PACF plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(time_series_diff, ax=axes[0])
        plot_pacf(time_series_diff, ax=axes[1])
        plt.show()
        
        max_ar = 4
        max_ma = 4
        info_criteria = ['aic', 'bic']
        data = self.data if is_data_stationary else time_series_diff
        self.orderSelection(data=data)
        
        for criteria in info_criteria:
            print(criteria)
            print(self.selectionInfo[criteria])

    def forecast(self, steps=10) -> pd.Series:
        """
        Description:
            Forecast using the ARIMA and GARCH models
        """
        forecast_data = self.results.forecast(steps)
        garch_forecast = self.garch_results.forecast(horizon=steps)
        # return forecast_data, garch_forecast.variance[-1:]
        # return garch_forecast.variance[-steps:-1]
        # print(garch_forecast.mean.iloc[-1])
        return garch_forecast.mean.iloc[-1]

    def __str__(self) -> str:
        return f"ARIMA-GARCH Model: AR({self.AR_order}), I({self.differencing_order}), MA({self.MA_order})"
        
    def orderSelection(self, data, max_ar=4, max_ma=4, info_criteria=['aic', 'bic']):
        self.selectionInfo = arma_order_select_ic(data, max_ar, max_ma, info_criteria)
        return self.selectionInfo


## Example usage
if __name__ == "__main__":
    train_pct = 80
    
    data = pd.read_csv(SPTL_DATA_PATH)
    data_length = len(data)
    
    AR_order = 2
    differencing_order = 1
    MA_order = 2
    
    model1 = ArimaGarchModel(
        data=data['Close'],
        timeseries=data['Date'],
        AR_order=AR_order,
        differencing_order=differencing_order,
        MA_order=MA_order
    )
    
    model1.runChecks()

    # Forecast
    forecast_data, garch_forecast = model1.forecast(steps=10)
    print(f"Forecast: {forecast_data}")
    print(f"GARCH Forecasted Volatility: {garch_forecast}")

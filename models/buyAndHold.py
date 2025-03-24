from models.baseModel import ForecastModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ModelWarning
from src.globals import SPTL_DATA_PATH

class BuyAndHoldModel(ForecastModel):
    """
    Description:
        ARIMA (AutoRegressive Integrated Moving Average) based model built upon statsmodels.tsa library
        
    Parameters:
        data (float[]): History of timeseries data to base forecast upon
        timeseries (datetime[]): Timeseries index for the data
        AR_order (int): Auto Regressive look back window, AR(1), AR(2) etc...
        differencing_order (int): Differencing (integrating) order
        MA_order (int): Moving Average lookback window MA(1), MA(2) etc...
    """
    def __init__(self, data, timeseries) -> None:
        # Initialize the parent class
        super().__init__(data=data, timeseries=timeseries) # self.data, self.timeseries, self.results, self.forecastData, self.name
        self.name = 'BUY_AND_HOLD'
        self.model = None
        self.results
    
    def fitModel(self) -> None:
        pass
    
    def forecast(self, steps=10) -> pd.Series:
        """
        Description:
            Return positive returns to signal to hold
        """
        final_value = self.data.iloc[-1]
        self.forecastData = pd.Series([final_value + 1] * steps, index=[self.data.index[-1] + i for i in range(steps)])
        return self.forecastData
    
    def __str__(self) -> str:
        return f"Buy And Hold"

## Remove once built
if __name__ == "__main__":
        
    data = pd.read_csv(SPTL_DATA_PATH)
    data_length = len(data)
        
    model1 = BuyAndHoldModel(
        data=data['Close'],
        timeseries=data['Date']
    )
    
    model1.fitModel()
    
    f = model1.forecast()
    print(f)
    
    model1.plot()
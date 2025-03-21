from src.forecastModelBase import ForecastModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic
import matplotlib.pyplot as plt
from src.globals import SPTL_DATA_PATH

class ArimaModel(ForecastModel):
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
    def __init__(self, data, timeseries, AR_order=1, differencing_order=1, MA_order=1) -> None:
        # Initialize the parent class
        super().__init__(data=data, timeseries=timeseries) # self.data, self.timeseries, self.results, self.forecastData
        
        self.AR_order = AR_order
        self.differencing_order = differencing_order
        self.MA_order = MA_order
        self.model = ARIMA(self.data, order=(self.AR_order, self.differencing_order, self.MA_order))
        self.selectionInfo = None
        
        # Initialize with a fit
        self.fitModel()
        
    def fitModel(self) -> pd.Series:
        """
        Description:
            Fit model using statsmodels.tsa inbuilt fitting algo
        """
        self.results = self.model.fit()
        return self.results
        
    def forecast(self, steps=10) -> pd.Series:
        """
        Description:
            Forecaset using historical data based on statsmodels.tsa inbuilt forecasting algo
        """
        self.forecastData = self.results.forecast(steps)
        return self.forecastData
    
    def __str__(self) -> str:
        return f"ARIMA Model: AR({self.AR_order}), I({self.differencing_order}), MA({self.MA_order})"
        
    # REVISIT THIS
    def orderSelection(self, max_ar=5, max_ma=5, info_criteria=['aic', 'bic']):
        self.selectionInfo = arma_order_select_ic(self.data, max_ar, max_ma, info_criteria)
        return self.selectionInfo


## Remove once built
if __name__ == "__main__":
    
    train_pct = 80
    
    data = pd.read_csv(SPTL_DATA_PATH)
    # print(data)
    data_length = len(data)
    
    AR_order = 2
    differencing_order = 1
    MA_order = 2
    
    model1 = ArimaModel(
        data=data['Close'],
        timeseries=data['Date'],
        AR_order=AR_order,
        differencing_order=differencing_order,
        MA_order=MA_order
    )
    
    model1.fitModel()
    
    f = model1.forecast()
    print(f)
    
    # model1.plot()
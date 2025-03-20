from modelMeta.forecastModel import ForecastModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic
import matplotlib.pyplot as plt

# ~~~ INFO ~~~
# Arima based model built upon statsmodels.tsa library

class ArimaModel(ForecastModel):
    
    def __init__(self, data, AR_order=1, differencing_order=1, MA_order=1):
        
        super().__init__(data=data) # self.data, self.results, self.forecastData
        
        self.AR_order = AR_order
        self.differencing_order = differencing_order
        self.MA_order = MA_order
        self.model = ARIMA(self.data, order=(self.AR_order, self.differencing_order, self.MA_order))
        self.selectionInfo = None
        
        # Initialize with a fit
        self.fitModel()
        
    def fitModel(self):    
        self.results = self.model.fit()
        return self.results
        
    def forecast(self, steps=10):
        self.forecastData = self.results.forecast(steps)
        return self.forecastData
    
    def __str__(self):
        return self.model.__str__()
        
    # REVISIT THIS
    def orderSelection(self, max_ar=5, max_ma=5, info_criteria=['aic', 'bic']):
        self.selectionInfo = arma_order_select_ic(self.data, max_ar, max_ma, info_criteria)
        return self.selectionInfo


## Remove once built
if __name__ == "__main__":
    
    train_pct = 80
    
    file_name = '../data/SPTL_2023.csv'
    data = pd.read_csv(file_name)
    print(data)
    data_length = len(data)
    
    AR_order = 2
    differencing_order = 1
    MA_order = 2
    
    model1 = ArimaModel(
        data['Close'],
        AR_order==AR_order,
        differencing_order=differencing_order,
        MA_order=MA_order
    )
    print(model1)
    model1.fitModel()
    print(model1.results.summary())
    
    f = model1.forecast()
    print(f)
    
    model1.plot()
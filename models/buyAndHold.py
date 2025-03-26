from models.baseModel import ForecastModel
import pandas as pd
from src.globals import SPTL_DATA_PATH

class BuyAndHoldModel(ForecastModel):
    """
    Description:
        Buy and Hold - Signals to only ever keep the stock
        
    Parameters:
        data (float[]): History of timeseries data to base forecast upon
        timeseries (datetime[]): Timeseries index for the data
    """
    def __init__(self, data, timeseries) -> None:
        # Initialize the parent class
        super().__init__(data=data, timeseries=timeseries) # self.data, self.timeseries, self.results, self.forecastData, self.name
        self.name = 'BUY_AND_HOLD'
        self.model = None
    
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
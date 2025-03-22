from src.forecastModelBase import ForecastModel
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from src.globals import SPTL_DATA_PATH

# ~~~ INFO ~~~
# GAM (Generalized Additive Models) based model built upon Facebook's prophet model


class GamModel(ForecastModel):
    """
    Description:
        GAM (Generalized Additive Models) based model built upon Facebook's prophet model
        
    Parameters:
        data (float[]): History of timeseries data to base forecast upon
        timeseries (datetime[]): Timeseries index for the data
    """
    def __init__(self, data, timeseries) -> None:
        # Initialize the parent class
        super().__init__(data=data, timeseries=timeseries) # self.data, self.timeseries, self.results, self.forecastData
        self.name = 'GAM'

        self.model= Prophet(changepoint_prior_scale=0.05, weekly_seasonality=False, daily_seasonality=True)
        
        # Prophet specific data
        self.formattedData = pd.DataFrame({
            'ds': self.timeseries,
            'y': self.data
        })
        
        # print(self.formattedData) # DELETE THIS
        
        self.formattedData.astype({'ds': 'datetime64[s]'})
        self.forecastResponse = None
                
        # Initialize with a fit
        self.fitModel()
        
    def fitModel(self) -> pd.Series:    
        self.results = self.model.fit(self.formattedData)
        return self.results
        
    def forecast(self, steps=10) -> pd.Series:
        # Create a DataFrame for future dates
        future = self.model.make_future_dataframe(periods=steps)
        YHAT = 'yhat' # date frame column with forecast value
        self.forecastResponse = self.model.predict(future)
        self.forecastData = self.forecastResponse[YHAT][-steps:]
        return self.forecastData
    
    def __str__(self) -> str:
        return f"GAM Prophet"
    
    # Method Overridign to use facebook default plotting
    def plot(self, maxLookback=50) -> None:
        fig = self.model.plot(self.forecastResponse)
        # if self.forecastData is not None:
            # Create combi dataframe and plot
            # plt.plot(self.data, color='blue')
            # plt.plot(self.forecastData, color='black')
        if self.actualForwardData is not None:
            last_date = pd.to_datetime(self.timeseries.iloc[-1])
            print('last_date', last_date)
            print('self.actualForwardData', self.actualForwardData)
            forward_dates = pd.date_range(start=last_date, periods=len(self.actualForwardData), freq='D')
            self.actualForwardData.index = forward_dates
            plt.plot(self.actualForwardData, color='red')
            legend = ['Actual']
            plt.legend(legend)           
            # plt.show()
        plt.show()


if __name__ == "__main__":

    train_pct = 0.5
    
    # file_name = '../data/SPTL_2023.csv'
    data = pd.read_csv(SPTL_DATA_PATH)
    print("\nDataset Info:")
    print(data.info())
    print("\nDataset Description:")
    print(data.describe())
    data_length = len(data)
    
    data = data.iloc[ : int(train_pct * len(data)), :]
    
    model2 = GamModel(
        data=data['Close'],
        timeseries=data['date_string']
    )
    
    print(model2)
    steps = 20
    f = model2.forecast(steps=steps)
    print(f)
    
    model2.plot()
    
    # model2.forecastData.iloc[-steps:].to_csv("model2_test.csv")
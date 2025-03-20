from modelMeta.forecastModel import ForecastModel
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

class GamModel(ForecastModel):
    
    def __init__(self, data):
        super().__init__(data=data) # self.data, self.results, self.forecastData

        self.model= Prophet()
                
        # Initialize with a fit
        self.fitModel()
        
    def fitModel(self):    
        self.results = self.model.fit(self.data)
        return self.results
        
    def forecast(self, steps=10):
        self.forecastData = self.results.forecast(steps)
        return self.forecastData
    
    def __str__(self):
        return self.model.__str__()
    
    def plot(self):
        plt.plot(self.data)
        plt.show()


if __name__ == "__main__":

    train_pct = 80
    
    file_name = '../data/SPTL_2023.csv'
    data = pd.read_csv(file_name)
    print(data[['Date', 'Close']])
    data_length = len(data)
    
    
    # model2 = GamModel(
        # data[['Date', 'Close']],
    # )
    
    # print(model2)
    # model2.fitModel()
    # f = model2.forecast()
    # print(f)
    
    # model2.plot()

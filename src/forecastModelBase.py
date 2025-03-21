import matplotlib.pyplot as plt

class ForecastModel:
    """
    Description:
        Parent (base) class for all strategies, defining the base data that all 
        models must output as well as some boiler plate methods to apply to the data
    
    Parameters:
        data (pd.Series): Timeseries data
        timeseries (pd.Series): Timeseries index for the data provided
    """
    def __init__(self, data, timeseries) -> None:
        self.data = data
        self.timeseries = timeseries
        self.results = None
        self.forecastData = None
        
    # Overridable by child classs if needed
    def plot(self) -> None:
        if self.forecastData is None:
            plt.plot(self.data)
            plt.show()
            
        else:
            # Create combi dataframe and plot
            plt.plot(self.data, color='blue')
            plt.plot(self.forecastData, color='black')
            plt.show()
        
    # Potentially add a data cleaning method to ensure there is a column for the timesseries and one for the value
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
        self.data = data                    # Historic timeseries data
        self.timeseries = timeseries        # Historic timeseries indexess (days etc.)
        self.results = None                 # Output of the model being fitted
        self.forecastData = None            # Future forecast data from the model
        self.actualForwardData = None       # Actual look forward data for model comparison
        self.name = None                    # Name of the model to be overriden in the child class
        self.lookForwardOverride = None     # Look forward override to be used in forecasting
        self.useLookForwardDiff = False     # Look at future trend diffs rather than their absolute values
        
    # Overridable by child classs if needed
    def plot(self) -> None:
        
        plt.plot(self.data, color='blue')
        legend = ['Historical Data']
        
        if self.forecastData is not None:
            # Create combi dataframe and plot
            plt.plot(self.forecastData, color='black')
            legend.append('Forecast')
        
        if self.actualForwardData is not None:
            plt.plot(self.actualForwardData, color='red')
            legend.append('Actual Forward Data')
        
        plt.legend(legend)             
        plt.title(self.name)
        plt.show()
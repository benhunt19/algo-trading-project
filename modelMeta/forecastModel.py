import matplotlib.pyplot as plt

# Generic class to forecast data

class ForecastModel:
    def __init__(self, data):
        self.data = data
        self.results = None
        self.forecastData = None
        
    def plot(self):
        plt.plot(self.data)
        plt.show()
        
    # Potentially add a data cleaning method to ensure there is a column for the timesseries and one for the value
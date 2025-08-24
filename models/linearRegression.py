from models.baseModel import ForecastModel
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
from src.globals import SPTL_DATA_PATH


import warnings
warnings.filterwarnings('ignore')

class LinearRegressionModel(ForecastModel):
    """
    Description:
        Linear Regression based model built upon sklearn.linear_modela library
        
    Parameters:
        data (float[]): History of timeseries data to base forecast upon
        timeseries (datetime[]): Timeseries index for the dat
    """
    def __init__(self, data, timeseries, lookbackTrainWindow , lookForwardHorizon=None) -> None:
        # Initialize the parent class
        super().__init__(data=data, timeseries=timeseries) # self.data, self.timeseries, self.results, self.forecastData, self.name
        self.name = 'LinearRegression'

        self.selectionInfo = None
        self.lookForwardHorizon = lookForwardHorizon      # Number of steps to look forward over
        self.lookbackTrainWindow = lookbackTrainWindow      # Window to look back and train over, ie window size to fit data
        
        self.model = LinearRegression()
        
        self.lags = [1, 2, 3, 7, 14, 30, 100]  # Common choices
        
        # Initialize with a fit
        self.fitModel()
        
    def fitModel(self):
        """
        Description:
            Fit model using statsmodels.tsa inbuilt fitting algo
        """
        # Ensure data is numpy array for faster operations
        data_array = np.array(self.data)
        n = len(data_array)

        # Determine the maximum lag to ensure we have enough history
        max_lag = np.max(self.lags)

        # Create features (X) and targets (y) for training
        X_list = []; y_list = []

        # Loop from the maximum lag to leave room for the forecast horizon
        for i in range(max_lag, n - self.lookForwardHorizon):
            # For each data point, get all the lag features
            features = [data_array[i - lag] for lag in self.lags]
            X_list.append(features)
            # Target is the value 'forecast_horizon' steps ahead
            y_list.append(data_array[i + self.lookForwardHorizon])

        # Convert lists to numpy arrays
        X_train = np.array(X_list); y_train = np.array(y_list)
        
        
        self.model.fit(X_train, y_train)
        
   
    def forecast(self, x_test : np.array) -> pd.Series:
        """
        Description:
            Forecaset using historical data based on statsmodels.tsa inbuilt forecasting algo
        
        """
        x_test_formatted = np.array([[x_test[len(x_test) - lag] for lag in self.lags]])
        
        self.forecastData = self.model.predict(x_test_formatted)
        return self.forecastData
    
    def __str__(self) -> str:
        return f"Linear Regression Model"
        

if __name__ == "__main__":
    pass
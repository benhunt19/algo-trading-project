from strategies.strategy1 import ArimaModel
from strategies.strategy2 import GamModel
import numpy as np
from src.globals import (SPTL_DATA_PATH, SPTL_DATA_PATH_LOOKBACK)
import pandas as pd

# Framework for testing and exporting results for the models

class ModelTestingFramework:
    """
    Description:
        The framework for testing and applying statistical tests on various models based on test data
    
    Parameters:
        leverage (float): Leverage multiplier to
        starting_cap (float): Starting capital
        models (ForecastModel[]): Forecasting models to test
        data (pd.Series): data to test the models on
        timeseries (pd.Series): timeseries index for the data
    
    """
    def __init__(self, leverage, starting_cap, models, data, timeseries):
        self.theta = np.array([])
        self.leverage = leverage
        self.starting_cap = starting_cap
        self.models = models
        self.starting_leveraged_cap = self.leverage * self.starting_cap
        self.initialised_models = []
        self.data = data
        self.timeseries = timeseries
        
        # Initialize the models
        self.initialiseModels()
    
    def initialiseModels(self):
        for model in self.models:
            self.initialised_models.append(model(data=self.data, timeseries=self.timeseries))
            
    def createDayPortfolio(self, lookback=True):
        """
        Description:
            Run each model over the timeseries data to create a long/short portfolio strategy
        """        
        for model in self.initialised_models:
            
        
if __name__ == "__main__":
    
    leverage = 10
    starting_cap = 100_000
    models = [ArimaModel, GamModel]
    
    train_pct = 0.5
    
    # file_name = '../data/SPTL_2023.csv'
    data = pd.read_csv(SPTL_DATA_PATH_LOOKBACK)
    data_length = len(data)
    
    data = data.iloc[ : int(train_pct * len(data)), :]
    
    mft = ModelTestingFramework(
        leverage=leverage,
        starting_cap=starting_cap,
        models=models,
        data=data['Close'],
        timeseries=data['date_string'],
    )
    
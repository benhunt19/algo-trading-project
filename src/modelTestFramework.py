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
        modelSpecificKwargs (list[dict]): Model specific keyword arguments when defining the model
        data (pd.Series): data to test the models on
        timeseries (pd.Series): timeseries index for the data
    
    """
    def __init__(self, leverage, starting_cap, models, modelSpecificKwargs, data, timeseries):
        self.thetas = np.array([])
        self.leverage = leverage
        self.starting_cap = starting_cap
        self.models = models
        self.modelSpecificKwargs = modelSpecificKwargs
        self.starting_leveraged_cap = self.leverage * self.starting_cap
        self.initialised_models = []
        self.data = data
        self.timeseries = timeseries
                    
    def createDayPortfolio(self, lookbackWindow=250, startIndex=250, endIndex=300, shortLookForward=1, longLookForward=10):
        """
        Description:
            Run each model over the timeseries data to create a long/short portfolio strategy
        """        
        
        # How many total days to run algorithm over
        forcastLength = endIndex - startIndex
        
        self.thetas = np.zeros(forcastLength)
        
        # For each model
        for m_index, model in enumerate(self.models):
            
            print(f"Running For Model {model}")

            # For each forward looking timestep
            for i in range(forcastLength):
                tmpStartIndex = startIndex + i - lookbackWindow
                tmpEndIndex = startIndex + i
                tmpTrainData = self.data.iloc[ tmpStartIndex : tmpEndIndex]
                tmpTrainTimeseries = self.timeseries.iloc[ tmpStartIndex : tmpEndIndex]
                
                print(f'Training standard deviation: {tmpTrainData.std()}')
                
                # Create model and fit to lookback data
                m = model(
                    data=tmpTrainData,
                    timeseries=tmpTrainTimeseries,
                    **self.modelSpecificKwargs[m_index]
                )
                
                # Foreast Lookback windows
                shortLookForwardData = m.forecast(steps=shortLookForward)
                longLookForwardData = m.forecast(steps=longLookForward)

                
                if i % 20 == 0:
                    m.plot()
                
                print('longLookForwardData', longLookForwardData)
                print('shortLookForwardData', shortLookForwardData)
                
                # Clear memory just incase
                del m
        
        
if __name__ == "__main__":
    
    leverage = 10
    starting_cap = 100_000
    
    models = [
        ArimaModel,
        GamModel
    ]
    
    # Change this to be iterable (to find the best performing model!!
    modelSpecificKwargs = [
        {
            'AR_order': 3,
            'differencing_order': 1,
            'MA_order': 20
        },
        {}
    ]
    
    train_pct = 0.8
    
    # file_name = '../data/SPTL_2023.csv'
    data = pd.read_csv(SPTL_DATA_PATH_LOOKBACK)
    data_length = len(data)
    
    data = data.iloc[ : int(train_pct * len(data)), :]
    
    print(data)
    
    mft = ModelTestingFramework(
        leverage=leverage,
        starting_cap=starting_cap,
        models=models,
        modelSpecificKwargs=modelSpecificKwargs,
        data=data['Close'],
        timeseries=data['date_string'],
    )
    
    portfolioDict = {
        'startIndex': 250,
        'endIndex': 350
    }
    
    mft.createDayPortfolio(**portfolioDict)
    
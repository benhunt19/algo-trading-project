from strategies.strategy1 import ArimaModel
from strategies.strategy2 import GamModel
from src.globals import (SPTL_DATA_PATH, SPTL_DATA_PATH_LOOKBACK)
from src.portfolio import Portfolio
import pandas as pd
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning


warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Framework for testing and exporting results for the models

class ModelTestingFramework:
    """
    Description:
        The framework for testing and applying statistical tests on various models based on test data
    
    Parameters:
        leverage (float): Leverage multiplier to
        starting_cap (float): Starting capital
        models (dict{ForecastModel}): Forecasting models to test
        data (pd.Series): data to test the models on
        timeseries (pd.Series): timeseries index for the data
    
    """
    def __init__(self, leverage, starting_cap, models, data, timeseries, riskNeutral):
        self.leverage = leverage
        self.starting_cap = starting_cap
        self.models = models
        self.data = data
        self.timeseries = timeseries
        self.riskNeutral = riskNeutral
                    
    def testModels(self, lookbackWindow=np.inf, startIndex=250, endIndex=300, shortLookForward=1, longLookForward=10, plotOnModuloIndex=10):
        """
        Description:
            Run each model over the timeseries data to create a long/short portfolio strategy
        Parameters:
            lookbackWindow (int): Number of days for the model to lookback over when fitting, (np.inf for maximum on data)
        """        
        
        # How many total days to run algorithm over
        forcastLength = endIndex - startIndex
        
        # For each model
        for m_index, modelName in enumerate(self.models):
            
            # Create portfolio instance and start live plot
            portfolio = Portfolio(starting_cap=self.starting_cap, leverage=self.leverage, length=forcastLength)
            portfolio.startLivePlot()
            
            ratingArray = np.zeros(forcastLength)
            
            # early skip if not enabled
            if not self.models[modelName]['enabled']:
                continue
            
            print(f"Running For Model {modelName}")

            # For each forward looking timestep
            for i in range(forcastLength):
                
                print('\n')
                
                tmpStartIndex = startIndex + i - lookbackWindow if lookbackWindow != np.inf else 0
                tmpEndIndex = startIndex + i
                tmpTrainData = self.data.iloc[ tmpStartIndex : tmpEndIndex]
                tmpTrainTimeseries = self.timeseries.iloc[ tmpStartIndex : tmpEndIndex]
                
                # print(f'tmpEndIndex = {tmpEndIndex}')
                
                dayStdDev = tmpTrainData.std()
                print(f'Training standard deviation: {dayStdDev}')
                
                # Create model and fit to lookback data
                m = self.models[modelName]['model'](
                    data=tmpTrainData,
                    timeseries=tmpTrainTimeseries,
                    **self.models[modelName]['kwargs']
                )
                
                # Foreast Lookback windows
                # shortLookForwardData = m.forecast(steps=shortLookForward)
                longLookForwardData = m.forecast(steps=longLookForward)

                deltaAfterN = longLookForwardData.iloc[-1] - tmpTrainData.iloc[-1]
                
                actualDeltaAfterN = self.data.iloc[tmpEndIndex + longLookForward] - tmpTrainData.iloc[-1]
                
                m.actualForwardData = self.data.iloc[tmpEndIndex : tmpEndIndex + longLookForward]
                                
                print("DELTA: ", deltaAfterN)
                print("ACTUAL DELTA: ", actualDeltaAfterN)
                
                # Get the sign of the delta after N days
                if np.sign(deltaAfterN) == np.sign(actualDeltaAfterN):
                    ratingArray[i] = 1
                
                if i % plotOnModuloIndex == 0:
                    pass
                    # m.plot()
                
                # Find realised returns going into the day (as a percentage)
                realisedReturns = ( self.data.iloc[ tmpEndIndex + 1 ] - self.data.iloc[ tmpEndIndex ] ) / self.data.iloc[ tmpEndIndex ]
                print('realisedReturns: ', realisedReturns)
                    
                # Process day in the portfolio
                portfolio.processDay(
                    returns=realisedReturns,
                    nextDayPredictedReturns=deltaAfterN,
                    riskFreeRate=self.riskNeutral[tmpEndIndex],
                    standardDeviation=dayStdDev
                )
                
                del m
                portfolio.stockData[i] = self.data.iloc[tmpEndIndex + 1]
                portfolio.updatePlot()
            
            # Calculate how many signs were correct
            print("Model rating")
            print(ratingArray.mean())
            portfolio.stockData = self.data.iloc[startIndex : endIndex]
            # portfolio.plot()
            portfolio.plot()

if __name__ == "__main__":
    
    leverage = 10
    starting_cap = 100_000
    
    modelTestMeta = {
        
        'ArimaModel': {
            'model': ArimaModel,
            'kwargs': {
                'AR_order': 1,
                'differencing_order': 1,
                'MA_order': 5
            },
            'enabled': False
        },
        
        'GamModel': {
            'model': GamModel,
            'kwargs': {},
            'enabled': True,
            'deltaThreshold': 0.3 # review
        }
        
    }
    
    data = pd.read_csv(SPTL_DATA_PATH_LOOKBACK)
    data_length = len(data)
    
    print(data)
    
    mft = ModelTestingFramework(
        leverage=leverage,
        starting_cap=starting_cap,
        models=modelTestMeta,
        data=data['Close'],
        timeseries=data['date_string'],
        riskNeutral=data['daily_risk_free']
    )
    
    testModelDics = {
        'lookbackWindow': 200, 
        'startIndex': 250,
        'endIndex': 400,
        'plotOnModuloIndex': 40,
        'longLookForward': 10
    }
    
    mft.testModels(**testModelDics)
    
from strategies.strategy1 import ArimaModel
from strategies.strategy2 import GamModel
from src.globals import (SPTL_DATA_PATH, SPTL_DATA_PATH_LOOKBACK)
from src.portfolio import Portfolio
import pandas as pd
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ModelWarning
from pprint import pprint


warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=ModelWarning)
warnings.filterwarnings('ignore')

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
    def __init__(self, leverage, starting_cap, models, data, timeseries, riskNeutral) -> None:
        self.leverage = leverage
        self.starting_cap = starting_cap
        self.models = models
        self.data = data
        self.timeseries = timeseries
        self.riskNeutral = riskNeutral
                    
    def testModels(self, lookbackWindow, startIndex, endIndex, shortLookForward=1, longLookForward=10, plotOnModuloIndex=10, verbose=True, plot=True, livePlot=False):
        """
        Description:
            Run each model over the timeseries data to create a long/short portfolio strategy
        Parameters:
            lookbackWindow (int): Number of days for the model to lookback over when fitting, (np.inf for maximum on data)
            startIndex (int): The starting index of the data to forecast from
            endIndex (int): The final index of the data to forecast up until
        """        
        
        # How many total days to run algorithm over
        forcastLength = endIndex - startIndex
        
        # For each model
        for m_index, modelName in enumerate(self.models):
            
            # Create portfolio instance and start live plot
            portfolio = Portfolio(starting_cap=self.starting_cap, leverage=self.leverage, length=forcastLength)
            if livePlot:
                portfolio.startLivePlot()
            
            ratingArray = np.zeros(forcastLength)
            
            # early skip if not enabled
            if not self.models[modelName]['enabled']:
                continue
            
            pprint(f"Running For Model:")
            pprint(self.models[modelName])

            # For each forward looking timestep
            for i in range(forcastLength):
                
                if verbose:
                    print('\n')
                
                tmpStartIndex = startIndex + i - lookbackWindow if lookbackWindow != np.inf else 0
                tmpEndIndex = startIndex + i
                tmpTrainData = self.data.iloc[ tmpStartIndex : tmpEndIndex]
                tmpTrainTimeseries = self.timeseries.iloc[ tmpStartIndex : tmpEndIndex]
                
                dayStdDev = tmpTrainData.std()
                if verbose:
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
                
                if verbose:                
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
                
                if verbose:
                    print('realisedReturns: ', realisedReturns)
                    
                # Process day in the portfolio
                portfolio.processDay(
                    returns=realisedReturns,
                    nextDayPredictedReturns=deltaAfterN,
                    riskFreeRate=self.riskNeutral[tmpEndIndex] / 100,
                    standardDeviation=dayStdDev,
                    threshold=self.models[modelName]['deltaThreshold'],
                    verbose=False
                )
                
                # Clear portfolio data
                del m
                portfolio.stockData[i] = self.data.iloc[tmpEndIndex + 1]
                
                if livePlot:
                    portfolio.updatePlot()
            
            # Calculate how many signs were correct
            print(f"Direction Correctness: ", ratingArray.mean())
            portfolio.stockData = self.data.iloc[startIndex : endIndex]
            
            print("Final value:")
            print(portfolio.totalCapitalOnDay(portfolio.currentDayIndex - 1))
            
            print('sharpeRatio: ', portfolio.sharpeRatio())
            
            if plot:
                portfolio.plot()
    
    @staticmethod
    def modelMetaBuilder(model, thresholds, kwargs={}) -> list[dict]:
        """
        Description:
            Builds model meta to be passed into self.testModels
        """
        return { 
            'model_' + str(i) + str(model): {
                'deltaThreshold': threshold,
                'model': model,
                'kwargs': kwargs,
                'enabled': True
            } for i, threshold in enumerate(thresholds)
        }
        
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
            'deltaThreshold': 0.2, # review
            'enabled': True
        },
        
        'GamModel': {
            'model': GamModel,
            'kwargs': {},
            'deltaThreshold': 0.3, # review
            'enabled': True,
        }
        
    }
    
    modelTestMeta1 = ModelTestingFramework.modelMetaBuilder(
        model=ArimaModel,
        thresholds=np.linspace(0, 0.4, 10),
        kwargs={
            'AR_order': 1,
            'differencing_order': 1,
            'MA_order': 5
        }
    )
    
    modelTestMeta2 = ModelTestingFramework.modelMetaBuilder(
        model=ArimaModel,
        thresholds=np.linspace(0, 0.4, 10),
        kwargs={
            'AR_order': 2,
            'differencing_order': 1,
            'MA_order': 4
        }
    )
    
    modelTestMeta3 = ModelTestingFramework.modelMetaBuilder(
        model=GamModel,
        thresholds=np.linspace(0.15, 1.5, 12),
        kwargs={
            'weeklySeasonality': False,
            'dailySeasonality': False,
        }
    )
    
    # combiMeta = modelTestMeta1 + modelTestMeta2 + modelTestMeta3
    
    data = pd.read_csv(SPTL_DATA_PATH_LOOKBACK)
    data_length = len(data)
    
    # print(data)
    
    mft = ModelTestingFramework(
        leverage=leverage,
        starting_cap=starting_cap,
        models=modelTestMeta3,
        data=data['Close'],
        timeseries=data['date_string'],
        riskNeutral=data['daily_risk_free']
    )
    
    testModelDicts = {
        'lookbackWindow': np.inf, 
        'startIndex': 250,
        'endIndex': 496,
        'plotOnModuloIndex': 40,
        'longLookForward': 3,
        'verbose': False,
        'plot': False,
        'livePlot': False
    }
    
    mft.testModels(**testModelDicts)
    
from src.modelTestFramework import ModelTestingFramework
from models.arima import ArimaModel
from models.prophet import GamModel
from models.garch import GarchModel
from models.linearRegression import LinearRegressionModel
from models.lstm import LSTMModel
from models.buyAndHold import BuyAndHoldModel
from src.globals import (SPTL_DATA_PATH_LOOKBACK, RESULTS_PATH, AAPL_TRANSFORMED)
from src.modelTestFramework import ModelTrainType

import pandas as pd
import numpy as np

# Example code to test and run the framework
if __name__ == "__main__":
    
    leverage = 1
    starting_cap = 100_000
    
    modelTestMeta1 = ModelTestingFramework.modelMetaBuilder(
        model=GarchModel,
        thresholds=[0.15],
        kwargs={
            'p': 2,
            'q': 2,
            'lookForwardOverride': 20
        }
    )
    
    # modelTestMeta2 = ModelTestingFramework.modelMetaBuilder(
    #     model=BuyAndHoldModel,
    #     thresholds=[0.1],
    #     kwargs={}
    # )
    
    modelTestMeta3 = ModelTestingFramework.modelMetaBuilder(
        model=GamModel,
        thresholds=[0.05],
        kwargs={
            'weeklySeasonality': False,
            'dailySeasonality': False,
            'lookForwardOverride': 5,
            'useLookForwardDiff': True,
            'changepointPriorScale': 0.1
        }
    )
    
    modelTestMeta4 = ModelTestingFramework.modelMetaBuilder(
        model=LinearRegressionModel,
        thresholds=[1e-7],
        kwargs={
            'lookForwardHorizon': 20,
            'lookbackTrainWindow': 1000
        },
        modelTrainType=ModelTrainType.ML_RETRAIN
    )
    
    
    modelTestMeta5 = ModelTestingFramework.modelMetaBuilder(
        model=LSTMModel,
        thresholds=[0],
        kwargs={
            'lookForwardHorizon': 20,
            'lookback': 100,
            'epochs': 5,
            'batch_size':128
        },
        modelTrainType=ModelTrainType.ML_TRAIN_ONCE
    )
    
    # combiMeta =  modelTestMeta1 + modelTestMeta3
    # combiMeta =  modelTestMeta4
    # combiMeta = modelTestMeta1 + modelTestMeta4 + modelTestMeta5
    combiMeta = modelTestMeta5
    
    # data = pd.read_csv(SPTL_DATA_PATH_LOOKBACK)
    # data_length = len(data)
    
    data = pd.read_csv(AAPL_TRANSFORMED)
    
    # print(data)
    
    mft = ModelTestingFramework(
        leverage=leverage,
        starting_cap=starting_cap,
        models=combiMeta,
        data=data['midprice'],
        timeseries=data.iloc[:,0],
        riskNeutral=data['daily_risk_free']
    )
    
    testModelDicts = {
        'lookbackWindow': 1000, 
        'startIndex': 10_000,
        'endIndex': 20_000,
        'plotOnModuloIndex': 10_000,
        'longLookForward': 20,
        'verbose': False,
        'plot': True,
        'livePlot': False
    }
    
    portfolios = mft.testModels(**testModelDicts)
    
    names = ['garch', 'buy_n_hold', 'gam']
    
    saveData = False
    
    if saveData:
        print('Saving results')
        for i, portfolio in enumerate(portfolios):
            # Save portfolio metrics to CSV
            portfolio_data = pd.DataFrame({
                'value': portfolio.value,
                'thetas': portfolio.thetas,
                'thetaPrime': portfolio.thetaPrime,
                'PnL': portfolio.PnL,
                'capitalGains': portfolio.capitalGains,
            })
            portfolio_path = f"{RESULTS_PATH}/portfolio_{names[i]}.csv"
            portfolio_data.to_csv(portfolio_path, index=False)
            print(f"Saved portfolio data to {portfolio_path}")
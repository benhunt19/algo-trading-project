from src.modelTestFramework import ModelTestingFramework
from models.arima import ArimaModel
from models.prophet import GamModel
from models.garch import GarchModel
from models.buyAndHold import BuyAndHoldModel
from src.globals import (SPTL_DATA_PATH_LOOKBACK, RESULTS_PATH)
import pandas as pd
import numpy as np

# Example code to test and run the framework
if __name__ == "__main__":
    
    leverage = 10
    starting_cap = 100_000
    
    modelTestMeta1 = ModelTestingFramework.modelMetaBuilder(
        model=GarchModel,
        thresholds=[0.15],
        kwargs={
            'p': 2,
            'q': 2,
            'lookForwardOverride': 10
        }
    )
    
    modelTestMeta2 = ModelTestingFramework.modelMetaBuilder(
        model=BuyAndHoldModel,
        thresholds=[0.1],
        kwargs={}
    )
    
    modelTestMeta3 = ModelTestingFramework.modelMetaBuilder(
        model=GamModel,
        thresholds=[0.05],
        lookbackWindowOverride=150,
        kwargs={
            'weeklySeasonality': False,
            'dailySeasonality': False,
            'lookForwardOverride': 5,
            'useLookForwardDiff': True,
            'changepointPriorScale': 0.1
        }
    )
    
    combiMeta =  modelTestMeta1 + modelTestMeta2 + modelTestMeta3
    
    data = pd.read_csv(SPTL_DATA_PATH_LOOKBACK)
    data_length = len(data)
    
    # print(data)
    
    mft = ModelTestingFramework(
        leverage=leverage,
        starting_cap=starting_cap,
        models=combiMeta,
        data=data['Close'],
        timeseries=data['date_string'],
        riskNeutral=data['daily_risk_free']
    )
    
    testModelDicts = {
        'lookbackWindow': 150, 
        'startIndex': 250,
        'endIndex': 500,
        'plotOnModuloIndex': 40,
        'longLookForward': 6,
        'verbose': False,
        'plot': False,
        'livePlot': False
    }
    
    portfolios = mft.testModels(**testModelDicts)
    
    # names = [str(meta['model']) for meta in combiMeta]
    # print(names)
    names = ['garch', 'buy_n_hold', 'gam']
    
    print('Saving results')
    
    saveData = False
    
    if saveData:
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
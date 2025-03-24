from src.modelTestFramework import ModelTestingFramework
from models.arima import ArimaModel
from models.prophet import GamModel
from models.garch import GarchModel
from src.globals import (SPTL_DATA_PATH_LOOKBACK)
import pandas as pd
import numpy as np

# Example code to test and run the framework
if __name__ == "__main__":
    
    leverage = 10
    starting_cap = 100_000
    
    modelTestMeta = ModelTestingFramework.modelMetaBuilder(
        model=GarchModel,
        thresholds=np.linspace(0.15, 1.5, 12),
        kwargs={
            'p': 2,
            'q': 2
        }
    )
    
    data = pd.read_csv(SPTL_DATA_PATH_LOOKBACK)
    data_length = len(data)
    
    # print(data)
    
    mft = ModelTestingFramework(
        leverage=leverage,
        starting_cap=starting_cap,
        models=modelTestMeta,
        data=data['Close'],
        timeseries=data['date_string'],
        riskNeutral=data['daily_risk_free']
    )
    
    testModelDicts = {
        'lookbackWindow': np.inf, 
        'startIndex': 250,
        'endIndex': 490,
        'plotOnModuloIndex': 40,
        'longLookForward': 3,
        'verbose': False,
        'plot': True,
        'livePlot': False
    }
    
    mft.testModels(**testModelDicts)
    
from models.baseModel import ForecastModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from src.globals import SPTL_DATA_PATH

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

class LSTMModel(ForecastModel):
    """
    Description:
        LSTM (Long Short-Term Memory) neural network for time series forecasting.
        
    Parameters:
        data (float[]): History of timeseries data to base forecast upon
        timeseries (datetime[]): Timeseries index for the data
        lookback (int): Number of previous timesteps to use as input features
        lookForwardHorizon (int): Number of steps to predict ahead
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    """
    def __init__(self, data, timeseries, lookback=100, lookForwardHorizon=20, epochs=5, batch_size=32) -> None:
        super().__init__(data=data, timeseries=timeseries)
        self.name = 'LSTM'
        self.lookback = lookback
        self.lookForwardHorizon = lookForwardHorizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None

        self.fitModel()

    def create_sequences(self, data, lookback, lookForwardHorizon):
        X, y = [], []
        for i in range(len(data) - lookback - lookForwardHorizon):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback+lookForwardHorizon-1])
        return np.array(X), np.array(y)

    def fitModel(self):
        # Normalize data
        data_array = np.array(self.data).reshape(-1, 1)
        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(data_array).flatten()

        # Prepare sequences
        X, y = self.create_sequences(data_scaled, self.lookback, self.lookForwardHorizon)
        X = X.reshape((X.shape[0], X.shape[1], 1))  # LSTM expects 3D input

        # Build LSTM model
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.lookback, 1)),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

        # Early stopping for better generalization
        es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

        # Train model
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1, callbacks=[es])

    def forecast(self, x_test: np.array) -> pd.Series:
        """
        Forecast using the trained LSTM model.
        x_test: The raw training data to predict on
        Returns: Array of predictions
        """
        # Normalize input
        x_test_scaled = self.scaler.transform(np.array(x_test).reshape(-1, 1)).flatten()
        
        # Create sequences from the test data
        X_test, y_test = self.create_sequences(x_test_scaled, self.lookback, self.lookForwardHorizon)
        
        print(f"X_test.shape: {X_test.shape}")
        
        # Generate predictions
        predictions_scaled = self.model.predict(X_test, verbose=0)
        predictions = self.scaler.inverse_transform(predictions_scaled).flatten()
        
        # Store forecast data
        self.forecastData = predictions
        return predictions

    def __str__(self) -> str:
        return f"LSTM Neural Network Model"

if __name__ == "__main__":
    pass
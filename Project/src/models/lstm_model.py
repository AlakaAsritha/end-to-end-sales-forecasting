import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, n_steps=4):
        super().__init__("LSTM")
        self.n_steps = n_steps
        self.scaler = MinMaxScaler()
        self.model = None

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data)):
            end_ix = i + self.n_steps
            if end_ix > len(data) - 1:
                break
            seq_x, seq_y = data[i:end_ix], data[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def train(self, data):
        """
        data: Series with datetime index
        """
        # LSTM needs scaling
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        X, y = self.create_sequences(scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.n_steps, 1)),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X, y, epochs=20, verbose=0)
        
        self.last_sequence = scaled_data[-self.n_steps:]
        return self.model

    def predict(self, steps):
        predictions = []
        current_seq = self.last_sequence.copy()
        
        for _ in range(steps):
            x_input = current_seq.reshape((1, self.n_steps, 1))
            yhat = self.model.predict(x_input, verbose=0)
            predictions.append(yhat[0, 0])
            
            # Update sequence
            current_seq = np.append(current_seq[1:], yhat)
            
        # Inverse transform
        predictions = np.array(predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(predictions).flatten()

    def evaluate(self, train_data, test_data):
        self.train(train_data)
        predictions = self.predict(len(test_data))
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mae = mean_absolute_error(test_data, predictions)
        return {"rmse": rmse, "mae": mae}

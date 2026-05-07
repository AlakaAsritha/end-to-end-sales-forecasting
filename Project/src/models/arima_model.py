import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .base_model import BaseModel

class ArimaModel(BaseModel):
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
        super().__init__("ARIMA/SARIMA")
        self.order = order
        self.seasonal_order = seasonal_order

    def train(self, data):
        """
        data: Series with datetime index
        """
        # We use a simple SARIMA model. 
        # In a real production system, we might use pmdarima's auto_arima.
        self.model = SARIMAX(data, 
                             order=self.order, 
                             seasonal_order=self.seasonal_order,
                             enforce_stationarity=False,
                             enforce_invertibility=False)
        self.model_fit = self.model.fit(disp=False)
        return self.model_fit

    def predict(self, steps):
        forecast = self.model_fit.get_forecast(steps=steps)
        return forecast.predicted_mean

    def evaluate(self, train_data, test_data):
        self.train(train_data)
        predictions = self.predict(len(test_data))
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mae = mean_absolute_error(test_data, predictions)
        return {"rmse": rmse, "mae": mae}

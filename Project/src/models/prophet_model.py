import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .base_model import BaseModel

class ProphetModel(BaseModel):
    def __init__(self):
        super().__init__("Prophet")
        self.model = None

    def train(self, data):
        """
        data: Series with datetime index
        """
        df_prophet = data.reset_index()
        df_prophet.columns = ['ds', 'y']
        
        self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        self.model.fit(df_prophet)
        return self.model

    def predict(self, steps):
        # Prophet handles steps by creating a future dataframe
        future = self.model.make_future_dataframe(periods=steps, freq='W-SAT')
        forecast = self.model.predict(future)
        return forecast['yhat'].tail(steps).values

    def evaluate(self, train_data, test_data):
        self.train(train_data)
        predictions = self.predict(len(test_data))
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mae = mean_absolute_error(test_data, predictions)
        return {"rmse": rmse, "mae": mae}

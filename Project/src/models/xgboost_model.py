import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .base_model import BaseModel
from ..feature_engineer import FeatureEngineer

class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__("XGBoost")
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        self.fe = FeatureEngineer()
        self.feature_cols = None

    def train(self, data):
        """
        data: Series with datetime index
        """
        df_feat = self.fe.create_features(data.to_frame())
        df_train = self.fe.prepare_supervised_data(df_feat)
        
        X = df_train.drop(columns=['Total'])
        y = df_train['Total']
        
        self.feature_cols = X.columns
        self.model.fit(X, y)
        
        # Save the last window of data for future predictions
        self.last_data = df_feat.tail(40) # Keep enough for lags/rolling
        return self.model

    def predict(self, steps):
        """
        Recursive prediction for multi-step forecast.
        """
        predictions = []
        current_data = self.last_data.copy()
        
        for _ in range(steps):
            # Create features for the next step
            # We need to append a row with the next date
            next_date = current_data.index[-1] + pd.Timedelta(weeks=1)
            
            # Temporary row with NaN for Total
            next_row = pd.DataFrame(index=[next_date], columns=current_data.columns)
            current_data = pd.concat([current_data, next_row])
            
            # Update features for the new row
            # Note: FeatureEngineer needs the full series to calculate lags/rolling correctly
            # We update only the last row
            feat_df = self.fe.create_features(current_data)
            X_next = feat_df.tail(1).drop(columns=['Total'])
            X_next = X_next[self.feature_cols]
            
            pred = self.model.predict(X_next)[0]
            predictions.append(pred)
            
            # Fill the Total value so next iteration can use it for lags
            current_data.loc[next_date, 'Total'] = pred
            
        return np.array(predictions)

    def evaluate(self, train_data, test_data):
        self.train(train_data)
        predictions = self.predict(len(test_data))
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mae = mean_absolute_error(test_data, predictions)
        return {"rmse": rmse, "mae": mae}

import pandas as pd
import numpy as np
import holidays

class FeatureEngineer:
    def __init__(self, country='US'):
        self.country_holidays = holidays.CountryHoliday(country)

    def create_features(self, df, lags=[1, 2, 4, 8], rolling_windows=[4, 12]):
        """
        Creates features for time series forecasting.
        Assumes df has a datetime index and a 'Total' column.
        """
        df = df.copy()
        
        # 1. Temporal Features
        df['month'] = df.index.month
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter
        
        # Holiday flag
        df['is_holiday'] = df.index.map(lambda x: 1 if x in self.country_holidays else 0)
        
        # 2. Lag Features
        for lag in lags:
            df[f'lag_{lag}'] = df['Total'].shift(lag)
            
        # 3. Rolling Statistics
        for window in rolling_windows:
            df[f'rolling_mean_{window}'] = df['Total'].shift(1).rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['Total'].shift(1).rolling(window=window).std()
            
        # 4. Ensure all features are numeric
        for col in df.columns:
            if col != 'Total':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def prepare_supervised_data(self, df, target_col='Total'):
        """Drops NaNs to prepare data for supervised learning models."""
        return df.dropna()

if __name__ == "__main__":
    # Test
    from data_manager import DataManager
    import os
    file_path = r'c:\Users\AGhantasala\Downloads\Project\Forecasting Case- Study.xlsx'
    dm = DataManager(file_path)
    df_raw = dm.preprocess_data()
    
    fe = FeatureEngineer()
    # Test for one state
    state_series = dm.get_state_series('Alabama').to_frame()
    df_feat = fe.create_features(state_series)
    print(df_feat.head(15))
    print(df_feat.columns)

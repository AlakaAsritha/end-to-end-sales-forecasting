import pandas as pd
import numpy as np

class DataManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None

    def load_data(self):
        """Loads data from the Excel file."""
        self.raw_data = pd.read_excel(self.file_path)
        return self.raw_data

    def preprocess_data(self):
        """Cleans and resamples the data to weekly frequency."""
        if self.raw_data is None:
            self.load_data()
        
        df = self.raw_data.copy()
        
        # 1. Clean Date column
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'Total', 'State'])
        
        # 2. Aggregate by State and Date (sum Total if multiple entries exist)
        df = df.groupby(['State', 'Date'])['Total'].sum().reset_index()
        
        # 3. Resample each state to weekly frequency
        processed_states = []
        for state in df['State'].unique():
            state_df = df[df['State'] == state].set_index('Date').sort_index()
            
            # Resample to Weekly (starting Saturday as per sample data observation)
            # 'W-SAT' means weekly on Saturdays
            state_weekly = state_df['Total'].resample('W-SAT').sum().reset_index()
            
            # Interpolate missing values if any (though sum() might put 0s)
            # Actually, if a week is missing, resample will put NaN or 0 depending on method.
            # Using sum() will put 0 for missing weeks if they are within the range.
            # We should probably interpolate if there are gaps.
            state_weekly['Total'] = state_weekly['Total'].replace(0, np.nan).interpolate(method='linear')
            state_weekly['State'] = state
            processed_states.append(state_weekly)
            
        self.processed_data = pd.concat(processed_states, ignore_index=True)
        return self.processed_data

    def get_state_series(self, state_name):
        """Returns a time series for a specific state."""
        if self.processed_data is None:
            self.preprocess_data()
        
        state_df = self.processed_data[self.processed_data['State'] == state_name].copy()
        state_df = state_df.set_index('Date').sort_index()
        return state_df['Total']

if __name__ == "__main__":
    # Test
    import os
    file_path = r'c:\Users\AGhantasala\Downloads\Project\Forecasting Case- Study.xlsx'
    dm = DataManager(file_path)
    df = dm.preprocess_data()
    print(df.head())
    print(df['State'].unique())
    print(f"Total rows: {len(df)}")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from .data_manager import DataManager
from .trainer import ModelTrainer

class Visualizer:
    def __init__(self, data_manager, trainer, output_dir='plots'):
        self.dm = data_manager
        self.trainer = trainer
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_forecast(self, state):
        """
        Plots historical data and the 8-week forecast for a state.
        """
        series = self.dm.get_state_series(state)
        model_obj = self.trainer.load_model(state)
        
        if model_obj is None:
            print(f"No model found for {state}. Train it first.")
            return
            
        forecast = model_obj.predict(8)
        
        # Create forecast series
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=8, freq='W-SAT')
        forecast_series = pd.Series(forecast, index=future_dates)
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=series.index[-52:], y=series.values[-52:], label='History (Last 1 year)')
        sns.lineplot(x=forecast_series.index, y=forecast_series.values, label='8-Week Forecast', color='red', linestyle='--')
        
        plt.title(f"Sales Forecast for {state} ({model_obj.name})")
        plt.xlabel("Date")
        plt.ylabel("Total Sales")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.output_dir, f"{state.replace(' ', '_')}_forecast.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")
        return output_path

if __name__ == "__main__":
    # Test
    DATA_FILE = r'c:\Users\AGhantasala\Downloads\Project\Forecasting Case- Study.xlsx'
    dm = DataManager(DATA_FILE)
    dm.preprocess_data()
    trainer = ModelTrainer(models_dir='models_storage')
    
    vis = Visualizer(dm, trainer)
    vis.plot_forecast('Alabama')

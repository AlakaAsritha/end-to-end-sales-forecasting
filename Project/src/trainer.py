import pandas as pd
import numpy as np
import pickle
import os
from .models.arima_model import ArimaModel
from .models.prophet_model import ProphetModel
from .models.xgboost_model import XGBoostModel
from .models.lstm_model import LSTMModel

class ModelTrainer:
    def __init__(self, models_dir='models_storage'):
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        self.best_models = {} # State -> (ModelObject, ModelName, Metrics)

    def select_best_model(self, state, series):
        """
        Trains multiple models for a state and selects the best one based on RMSE.
        """
        # Split: Last 8 weeks for validation
        train_size = len(series) - 8
        if train_size < 10: # Minimum data requirement
            print(f"Not enough data for state {state}")
            return None
            
        train_data = series.iloc[:train_size]
        val_data = series.iloc[train_size:]
        
        models_to_test = [
            ArimaModel(),
            ProphetModel(),
            XGBoostModel(),
            LSTMModel()
        ]
        
        results = []
        for model in models_to_test:
            try:
                print(f"Training {model.name} for {state}...")
                metrics = model.evaluate(train_data, val_data)
                results.append((model, metrics))
                print(f"Done. RMSE: {metrics['rmse']:.2f}")
            except Exception as e:
                print(f"Error training {model.name} for {state}: {e}")
                
        if not results:
            return None
            
        # Select best based on RMSE
        best_model_obj, best_metrics = min(results, key=lambda x: x[1]['rmse'])
        
        # Retrain best model on full data
        print(f"Best model for {state} is {best_model_obj.name}. Retraining on full data...")
        best_model_obj.train(series)
        
        self.best_models[state] = {
            'model': best_model_obj,
            'name': best_model_obj.name,
            'metrics': best_metrics
        }
        
        # Save model
        self.save_model(state, best_model_obj)
        
        return self.best_models[state]

    def save_model(self, state, model_obj):
        file_path = os.path.join(self.models_dir, f"{state.replace(' ', '_')}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(model_obj, f)

    def load_model(self, state):
        file_path = os.path.join(self.models_dir, f"{state.replace(' ', '_')}.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return None

    def train_all(self, data_manager):
        states = data_manager.processed_data['State'].unique()
        for state in states:
            series = data_manager.get_state_series(state)
            self.select_best_model(state, series)
        
        # Save summary
        summary = {state: {'name': info['name'], 'rmse': info['metrics']['rmse']} 
                   for state, info in self.best_models.items()}
        with open(os.path.join(self.models_dir, 'summary.pkl'), 'wb') as f:
            pickle.dump(summary, f)
        
        return summary

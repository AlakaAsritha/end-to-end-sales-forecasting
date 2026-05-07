import argparse
import os
from src.data_manager import DataManager
from src.trainer import ModelTrainer

def main(state_to_train=None):
    data_file = r'c:\Users\AGhantasala\Downloads\Project\Forecasting Case- Study.xlsx'
    
    print("--- 1. Loading and Preprocessing Data ---")
    dm = DataManager(data_file)
    dm.preprocess_data()
    
    print("--- 2. Starting Model Training and Selection ---")
    trainer = ModelTrainer(models_dir='models_storage')
    
    if state_to_train:
        print(f"Training specifically for state: {state_to_train}")
        series = dm.get_state_series(state_to_train)
        if series is not None:
            trainer.select_best_model(state_to_train, series)
        else:
            print(f"State '{state_to_train}' not found in data.")
    else:
        print("Training for all states (this may take a while)...")
        summary = trainer.train_all(dm)
        print("\n--- Training Summary ---")
        for state, info in summary.items():
            print(f"{state}: {info['name']} (RMSE: {info['rmse']:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting System")
    parser.add_argument("--state", type=str, help="Specific state to train for (optional)")
    args = parser.parse_args()
    
    main(state_to_train=args.state)

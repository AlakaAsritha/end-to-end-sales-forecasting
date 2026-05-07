from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import os
from .trainer import ModelTrainer
from .data_manager import DataManager

app = FastAPI(title="Sales Forecasting API")

# Initialize components
MODELS_DIR = 'models_storage'
DATA_FILE = r'c:\Users\AGhantasala\Downloads\Project\Forecasting Case- Study.xlsx'

trainer = ModelTrainer(models_dir=MODELS_DIR)
dm = DataManager(DATA_FILE)

@app.on_event("startup")
async def startup_event():
    # Load or preprocess data
    dm.preprocess_data()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Forecasting API. Use /forecast/{state} to get predictions."}

@app.get("/forecast/{state}")
def get_forecast(state: str, weeks: int = 8):
    """
    Returns next N weeks of sales forecast for a given state.
    """
    model_obj = trainer.load_model(state)
    if model_obj is None:
        # Try to match with underscores if not found
        model_obj = trainer.load_model(state.replace('_', ' '))
        
    if model_obj is None:
        raise HTTPException(status_code=404, detail=f"Model for state '{state}' not found. Ensure it has been trained.")
    
    try:
        predictions = model_obj.predict(weeks)
        
        # Create dates for predictions
        # Need to know the last date in history for this state
        series = dm.get_state_series(state)
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=weeks, freq='W-SAT')
        
        forecast_results = []
        for i, (d, p) in enumerate(zip(future_dates, predictions)):
            forecast_results.append({
                "week": f"Week {i+1}",
                "date": d.strftime('%Y-%m-%d'),
                "forecast": float(p)
            })
            
        return {
            "state": state,
            "model_used": model_obj.name,
            "forecast": forecast_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

@app.get("/states")
def list_states():
    if dm.processed_data is None:
        dm.preprocess_data()
    return {"states": dm.processed_data['State'].unique().tolist()}

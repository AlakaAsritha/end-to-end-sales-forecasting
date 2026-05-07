# End-to-End Time Series Forecasting System with REST API
Production-ready time series forecasting system for predicting next 8 weeks of state-wise sales using ARIMA/SARIMA, Facebook Prophet, XGBoost, and LSTM models with FastAPI integration.

## Features
State-wise 8-week sales forecasting
Multiple forecasting models:
ARIMA / SARIMA
Facebook Prophet
XGBoost
LSTM
Automatic best model selection
Feature engineering with lag & rolling features
Missing value and missing date handling
FastAPI-based REST API
Forecast visualization plots

## Tech Stack
Python
Pandas, NumPy
Statsmodels
Prophet
XGBoost
TensorFlow / Keras
FastAPI
Matplotlib

## Run the Project
Install dependencies
pip install -r requirements.txt
Train forecasting models
python main.py

Train for a specific state:
python main.py --state Alabama

Generate reports and plots
python -m src.export_metrics
python -m src.visualizer
Run FastAPI server
uvicorn src.api:app --reload
Open Swagger UI
http://127.0.0.1:8000/docs
Test API
http://localhost:8000/forecast/California

## Outputs
Future 8-week forecasts
Actual vs Predicted graphs
Forecast visualization plots
JSON API responses

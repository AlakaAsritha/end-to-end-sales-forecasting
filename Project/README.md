# Time Series Forecasting System

A production-ready system for state-wise sales forecasting using SARIMA, Facebook Prophet, XGBoost, and LSTM models.

## Features
- **Multi-Model Pipeline**: Automatically trains and compares ARIMA/SARIMA, Prophet, XGBoost, and LSTM.
- **Automated Selection**: Picks the best model for each state based on RMSE.
- **Feature Engineering**: Automated generation of lags, rolling windows, and holiday flags.
- **REST API**: Serves 8-week forecasts via FastAPI.
- **Visualization**: Generates forecast plots for evaluation.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Install `tabulate` for readable metric exports:
   ```bash
   pip install tabulate
   ```

## Usage

### 1. Training
To train the system for all states:
```bash
python main.py
```

To train for a specific state (e.g., Alabama):
```bash
python main.py --state Alabama
```

### 2. Export Metrics
Generate a performance summary:
```bash
python -m src.export_metrics
```

### 3. Generate Visualizations
Create forecast plots:
```bash
python -m src.visualizer
```
Plots will be saved in the `plots/` directory.

### 4. Run API
Start the FastAPI server:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## API Documentation

### Get Forecast
**Endpoint**: `GET /forecast/{state}`  
**Query Parameters**: `weeks` (default: 8)

**Example Request (PowerShell)**:
```powershell
Invoke-RestMethod http://localhost:8000/forecast/Alabama
```

**Example Response**:
```json
{
  "state": "Alabama",
  "model_used": "LSTM",
  "forecast": [
    {
      "date": "2023-12-16",
      "forecast": 197428976.0
    },
    ...
  ]
}
```

### List Available States
**Endpoint**: `GET /states`

## Project Structure
- `src/`: Core logic and model implementations.
- `models_storage/`: Trained model binaries and performance summaries.
- `plots/`: Generated forecast visualizations.
- `main.py`: Entry point for training.
- `requirements.txt`: Project dependencies.
```

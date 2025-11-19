# src/serving/api.py
import sys
import os
import pickle
import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.lstm import DemandLSTM
from src.database.db import load_data

# CONFIG
MODEL_PATH = "src/models/production_model.pt"
SCALER_PATH = "src/models/scaler.pkl"
LOOKBACK_WINDOW = 30

app = FastAPI(title="Drift-Aware Demand Forecaster")
Instrumentator().instrument(app).expose(app)

# Global variables
model = None
scaler = None

class WeatherRequest(BaseModel):
    temperature: float
    humidity: float

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    print("Loading model artifacts...")
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        model = DemandLSTM(input_size=2, hidden_size=50)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("System Ready!")
    except Exception as e:
        print(f"Start-up failed: {e}")

@app.post("/predict")
def predict(request: WeatherRequest):
    global model, scaler
    try:
        # 1. Fetch History
        # Explicitly requesting date, temperature, humidity
        query = f"""
            SELECT date, temperature, humidity 
            FROM features 
            ORDER BY date DESC 
            LIMIT {LOOKBACK_WINDOW - 1}
        """
        history_df = load_data(query)
        
        # Check if we have enough data
        if len(history_df) < LOOKBACK_WINDOW - 1:
            print(f"Not enough data. Found {len(history_df)} rows.")
            raise HTTPException(status_code=500, detail="Not enough history in Feature Store")
        
        # 2. Sort by date (Oldest first)
        history_df = history_df.sort_values(by="date", ascending=True)
        
        # 3. Drop date column (We only need temp/humidity for the model)
        history_data = history_df[['temperature', 'humidity']].values
        
        # 4. Combine with Current Request
        current_data = np.array([[request.temperature, request.humidity]])
        full_window = np.concatenate((history_data, current_data), axis=0)
        
        # 5. Scale
        # Pad with dummy demand (0) because scaler expects 3 columns
        dummy_demand = np.zeros((LOOKBACK_WINDOW, 1))
        window_with_dummy = np.hstack((full_window, dummy_demand))
        scaled_window = scaler.transform(window_with_dummy)
        
        # 6. Predict
        final_input = scaled_window[:, :2] # Take only temp/hum
        input_tensor = torch.tensor(final_input, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            prediction_scaled = model(input_tensor).item()
            
        # 7. Inverse Scale
        dummy_row_scaled = np.array([[0, 0, prediction_scaled]])
        dummy_row_original = scaler.inverse_transform(dummy_row_scaled)
        final_prediction = dummy_row_original[0, 2]
        
        return {
            "model_version": "v1",
            "predicted_demand": round(final_prediction, 2)
        }

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
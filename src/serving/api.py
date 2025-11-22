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
MIN_DATA_REQUIRED = LOOKBACK_WINDOW - 1
TEMP_MIN, TEMP_MAX = 15, 35
HUMIDITY_MIN, HUMIDITY_MAX = 30, 90

app = FastAPI(title="Drift-Aware Demand Forecaster")
Instrumentator().instrument(app).expose(app)

# Global variables
model = None
scaler = None
model_loaded = False

class WeatherRequest(BaseModel):
    temperature: float
    humidity: float

@app.on_event("startup")
def load_artifacts():
    global model, scaler, model_loaded
    print("Loading model artifacts...")
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Run training first.")
        
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        model = DemandLSTM(input_size=2, hidden_size=50)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()
        model_loaded = True
        print("✓ System Ready! Model artifacts loaded successfully.")
    except FileNotFoundError as fe:
        print(f"✗ Start-up failed: {fe}")
        model_loaded = False
    except Exception as e:
        print(f"✗ Start-up failed: {e}")
        model_loaded = False

@app.post("/predict")
def predict(request: WeatherRequest):
    global model, scaler, model_loaded
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Check logs.")
    
    try:
        # Validate input ranges
        if not (TEMP_MIN <= request.temperature <= TEMP_MAX):
            print(f"⚠ Temperature {request.temperature}°C outside typical range [{TEMP_MIN}-{TEMP_MAX}]")
        if not (HUMIDITY_MIN <= request.humidity <= HUMIDITY_MAX):
            print(f"⚠ Humidity {request.humidity}% outside typical range [{HUMIDITY_MIN}-{HUMIDITY_MAX}]")
        
        # 1. Fetch History
        query = f"""
            SELECT date, temperature, humidity 
            FROM features 
            ORDER BY date DESC 
            LIMIT {MIN_DATA_REQUIRED}
        """
        history_df = load_data(query)
        
        # Check if we have enough data
        if len(history_df) < MIN_DATA_REQUIRED:
            raise HTTPException(
                status_code=500, 
                detail=f"Insufficient data: {len(history_df)}/{MIN_DATA_REQUIRED} records"
            )
        
        # 2. Sort by date (Oldest first)
        history_df = history_df.sort_values(by="date", ascending=True)
        history_data = history_df[['temperature', 'humidity']].values
        
        # 3. Combine with current request
        current_data = np.array([[request.temperature, request.humidity]])
        full_window = np.vstack((history_data, current_data))
        
        # 4. Scale features only (model was trained on 2 features)
        scaled_window = scaler.transform(np.hstack([
            full_window,
            np.zeros((LOOKBACK_WINDOW, 1))  # Placeholder for demand column
        ]))[:, :2]  # Take only scaled temp/humidity
        
        # 5. Predict
        input_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction_scaled = model(input_tensor).item()
        
        # 6. Inverse scale the prediction
        # Create a dummy row with scaled values to inverse transform
        dummy_scaled = np.array([[0.0, 0.0, prediction_scaled]])
        dummy_original = scaler.inverse_transform(dummy_scaled)
        final_prediction = max(0, dummy_original[0, 2])  # Demand should be non-negative
        
        return {
            "model_version": "v1",
            "predicted_demand": round(final_prediction, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
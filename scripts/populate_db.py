# scripts/populate_db.py
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to Python path so we can import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.database.db import save_data

def generate_weather_data(start_date="2022-01-01", end_date="2024-01-01"):
    """
    Generates synthetic weather and demand data.
    Logic:
    - Temperature: Sine wave to simulate seasonality (Summer/Winter).
    - Humidity: Inverse of temp + monsoon spikes.
    - Demand: Correlated with Temp (Higher temp = Higher demand).
    """
    print("Generating synthetic data...")
    
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)
    
    # 1. Synthetic Temperature (Seasonality: Sin wave)
    # Base of 25 degrees, amplitude of 10 (ranges 15-35), plus noise
    t = np.linspace(0, 4 * np.pi, n) # 2 years = 2 cycles
    temperature = 25 + 10 * np.sin(t) + np.random.normal(0, 2, n)
    
    # 2. Synthetic Humidity
    # Roughly inverse to temp, but random
    humidity = 60 - 10 * np.cos(t) + np.random.normal(0, 5, n)
    
    # 3. Synthetic Demand (The Target Variable)
    # Formula: Demand = 100 + (Temp * 3) + (Humidity * 0.5) + Noise
    demand = 100 + (temperature * 3) + (humidity * 0.5) + np.random.normal(0, 5, n)
    
    df = pd.DataFrame({
        "date": dates,
        "temperature": temperature,
        "humidity": humidity,
        "demand": demand,
        "is_drifted": 0  # Flag to mark normal data
    })
    
    return df

if __name__ == "__main__":
    # Generate 2 years of data
    data = generate_weather_data()
    
    # Save to Postgres
    print("Pushing data to Postgres...")
    try:
        save_data(data, "features", if_exists='replace')
        print(f"✓ Database successfully populated with {len(data)} historical records!")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
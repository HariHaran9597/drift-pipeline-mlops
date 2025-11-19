# src/training/train.py
import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Fix paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.database.db import load_data
from src.models.lstm import DemandLSTM

# CONFIG
LOOKBACK_WINDOW = 30
EPOCHS = 20
MODEL_PATH = "src/models/production_model.pt"
SCALER_PATH = "src/models/scaler.pkl"

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x_window = data[i:(i + seq_length)]
        y_target = data[i + seq_length, -1]
        xs.append(x_window)
        ys.append(y_target)
    return np.array(xs), np.array(ys)

def train_model():
    print("Starting Retraining...")
    
    # 1. Load ALL data (including the new drifted data)
    df = load_data("SELECT temperature, humidity, demand FROM features ORDER BY date ASC")
    
    # 2. Scale
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df.values)
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    # 3. Create Sequences
    X_all, y_all = create_sequences(data_scaled, LOOKBACK_WINDOW)
    X_train = X_all[:, :, :2]
    y_train = y_all
    
    # 4. Train
    X_tensor = torch.from_numpy(X_train).float()
    y_tensor = torch.from_numpy(y_train).float().view(-1, 1)
    
    model = DemandLSTM(input_size=2, hidden_size=50)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Increased LR for faster convergence
    
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        
    # 5. Calculate Final RMSE (Error Score)
    # We want to know how much we missed by on average
    final_loss = loss.item()
    rmse = np.sqrt(final_loss)
    
    # 6. Save
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Retraining Complete. New RMSE: {rmse:.4f}")
    
    return rmse

if __name__ == "__main__":
    train_model()
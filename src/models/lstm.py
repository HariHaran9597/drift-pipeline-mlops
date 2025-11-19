# src/models/lstm.py
import torch
import torch.nn as nn

class DemandLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, output_size=1):
        super(DemandLSTM, self).__init__()
        # 1. LSTM Layer: Captures time patterns
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # 2. Fully Connected Layer: Maps features to demand
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        # Run through LSTM
        out, _ = self.lstm(x)
        
        # Take only the output of the last time step
        out = out[:, -1, :]
        
        # Pass through Linear layer to get prediction
        out = self.fc(out)
        return out
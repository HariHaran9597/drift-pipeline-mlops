# src/orchestration/flow.py
import sys
import os
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

# Add path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.drift.monitor import detect_drift
from src.training.train import train_model

@task
def check_drift_task():
    """Run the drift detection script."""
    return detect_drift()

@task
def retrain_model_task():
    """Run the training script."""
    return train_model()

@flow(task_runner=SequentialTaskRunner())
def drift_correction_flow():
    print("Starting Drift Correction Flow...")
    
    # Step 1: Check for Drift
    is_drifted = check_drift_task()
    
    if is_drifted:
        print("Drift Detected! Initiating Retraining Protocol...")
        
        # Step 2: Retrain Model
        # In a real system, we would verify if the new model is better.
        # Here, we assume retraining on fresh data is always better.
        new_rmse = retrain_model_task()
        
        print(f"Model Updated! New Accuracy (RMSE): {new_rmse:.4f}")
        print("System Self-Healed.")
    else:
        print("No Drift Detected. System Healthy.")

if __name__ == "__main__":
    drift_correction_flow()
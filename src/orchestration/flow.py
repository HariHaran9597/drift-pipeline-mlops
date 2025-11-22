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
    print("=" * 60)
    print("Starting Drift Correction Flow...")
    print("=" * 60)
    
    try:
        # Step 1: Check for Drift
        is_drifted = check_drift_task()
        
        if is_drifted:
            print("\n" + "🚨 " * 20)
            print("Drift Detected! Initiating Retraining Protocol...")
            print("🚨 " * 20)
            
            # Step 2: Retrain Model
            new_rmse = retrain_model_task()
            
            print("\n" + "=" * 60)
            print(f"✓ Model Updated! New Accuracy (RMSE): {new_rmse:.4f}")
            print("✓ System Self-Healed Successfully.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("✓ No Drift Detected. System Healthy.")
            print("=" * 60)
            
    except Exception as e:
        print(f"\n✗ Flow failed with error: {e}")
        raise

if __name__ == "__main__":
    drift_correction_flow()
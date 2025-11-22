# src/drift/monitor.py
import sys
import os
import json
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.database.db import load_data

# CONFIG
DRIFT_REPORT_PATH = "data/drift_report.json"
REFERENCE_WINDOW_SIZE = 500  # First N records as baseline
CURRENT_WINDOW_SIZE = 30     # Last N records for comparison

def detect_drift():
    print("Starting Drift Check...")
    
    try:
        # 1. Load Reference Data (The "Good" History)
        print("Fetching Reference Data...")
        reference_df = load_data(
            f"SELECT temperature, humidity, demand FROM features ORDER BY date ASC LIMIT {REFERENCE_WINDOW_SIZE}"
        )
        
        if len(reference_df) < REFERENCE_WINDOW_SIZE:
            print(f"⚠ Warning: Reference data has only {len(reference_df)} records (need {REFERENCE_WINDOW_SIZE})")
        
        # 2. Load Current Data (The "New" Stuff)
        print("Fetching Current Data...")
        current_df = load_data(
            f"SELECT temperature, humidity, demand FROM features ORDER BY date DESC LIMIT {CURRENT_WINDOW_SIZE}"
        )
        
        if len(current_df) < CURRENT_WINDOW_SIZE:
            print(f"⚠ Warning: Current data has only {len(current_df)} records (need {CURRENT_WINDOW_SIZE})")
        
        # 3. Run Evidently Report
        print("Running Statistical Tests (Evidently AI)...")
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        
        # 4. Parse Result
        report_dict = report.as_dict()
        
        # The result structure is deeply nested, we extract the boolean flag
        drift_detected = report_dict['metrics'][0]['result']['dataset_drift']
        
        # 5. Save Report (For debugging/Grafana later)
        with open(DRIFT_REPORT_PATH, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        if drift_detected:
            print("🚨 DRIFT DETECTED! Data distribution has significantly changed.")
            return True
        else:
            print("✓ Data is stable. No retraining needed.")
            return False
            
    except Exception as e:
        print(f"✗ Drift detection failed: {e}")
        raise

if __name__ == "__main__":
    try:
        is_drifted = detect_drift()
        # Exit with code 1 if drift detected (useful for CI/CD/Prefect)
        sys.exit(1 if is_drifted else 0)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
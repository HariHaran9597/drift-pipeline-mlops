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

def detect_drift():
    print("Starting Drift Check...")
    
    # 1. Load Reference Data (The "Good" History)
    # We take the first 500 rows as our baseline of "normal" behavior
    print("Fetching Reference Data...")
    reference_df = load_data("SELECT temperature, humidity, demand FROM features ORDER BY date ASC LIMIT 500")
    
    # 2. Load Current Data (The "New" Stuff)
    # We take the last 30 days. If this looks different from Reference, we have drift.
    print("Fetching Current Data...")
    current_df = load_data("SELECT temperature, humidity, demand FROM features ORDER BY date DESC LIMIT 30")
    
    # 3. Run Evidently Report
    # This runs statistical tests on every column
    print("Running Statistical Tests (Evidently AI)...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    
    # 4. Parse Result
    report_dict = report.as_dict()
    
    # The result structure is deeply nested, we extract the boolean flag
    drift_detected = report_dict['metrics'][0]['result']['dataset_drift']
    
    # 5. Save Report (For debugging/Grafana later)
    with open(DRIFT_REPORT_PATH, 'w') as f:
        json.dump(report_dict, f)
        
    if drift_detected:
        print("DRIFT DETECTED! Data distribution has significantly changed.")
        return True
    else:
        print("Data is stable. No retraining needed.")
        return False

if __name__ == "__main__":
    try:
        is_drifted = detect_drift()
        # Exit with code 1 if drift detected (useful for CI/CD/Prefect)
        if is_drifted:
            sys.exit(1) 
        else:
            sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
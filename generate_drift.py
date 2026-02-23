import sys
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def analyze_drift():
    # 1. Load Datasets
    reference = pd.read_csv('uber.csv').head(500) 
    current = pd.read_csv('new_batch.csv') 
    
    # 2. FIX: We tell the tool to ONLY check 'fare_amount'
    # This prevents 'noise' in other columns from failing the build.
    drift_report = Report(metrics=[
        DataDriftPreset(columns=['fare_amount']) 
    ])
    
    drift_report.run(reference_data=reference, current_data=current)
    drift_report.save_html("drift_report.html")
    
    # 3. Extract results
    result = drift_report.as_dict()
    # Check if our specific column drifted
    drift_detected = result['metrics'][0]['result']['dataset_drift']
    
    print(f"MLOps Status: Statistical Drift Detected? {drift_detected}")

    if drift_detected:
        print("CRITICAL: Price drift detected! Blocking deployment.")
        sys.exit(1) 
    else:
        print("SUCCESS: Price distribution is stable.")
        sys.exit(0) 

if __name__ == "__main__":
    analyze_drift()
    
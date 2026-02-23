import sys
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def analyze_drift():
    # 1. Load Real Datasets (No more hardcoded averages!)
    reference = pd.read_csv('uber.csv').head(500)  # Baseline training data
    current = pd.read_csv('new_batch.csv')         # New production batch
    
    # 2. Run Statistical Drift Analysis (KS Test/PSI)
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference, current_data=current)
    
    # 3. Save the HTML Dashboard
    drift_report.save_html("drift_report.html")
    
    # 4. Extract results to control the Pipeline
    result = drift_report.as_dict()
    drift_detected = result['metrics'][0]['result']['dataset_drift']
    
    print(f"MLOps Status: Statistical Drift Detected? {drift_detected}")

    if drift_detected:
        print("ALERT: Statistical shift detected in features. Stopping Pipeline!")
        sys.exit(1) # Fails the GitHub Action
    else:
        print("SUCCESS: Data distribution is stable.")
        sys.exit(0) # Passes the GitHub Action

if __name__ == "__main__":
    analyze_drift()
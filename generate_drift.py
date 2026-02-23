import sys
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def analyze_drift():
    # 1. Load Datasets
    # Use exactly the same rows for the "Success" demo to guarantee a green light
    reference = pd.read_csv('uber.csv').head(500) 
    current = pd.read_csv('new_batch.csv') 
    
    # 2. Run Analysis with a custom threshold
    # drift_share=0.5 means drift is only "Critical" if 50% of columns drift
    drift_report = Report(metrics=[
        DataDriftPreset(drift_share=0.5) 
    ])
    
    drift_report.run(reference_data=reference, current_data=current)
    
    # 3. Save the HTML Dashboard
    drift_report.save_html("drift_report.html")
    
    # 4. Logic for the Green/Red light
    result = drift_report.as_dict()
    drift_detected = result['metrics'][0]['result']['dataset_drift']
    
    print(f"MLOps Status: Statistical Drift Detected? {drift_detected}")

    if drift_detected:
        print("CRITICAL: Data drift detected. Blocking deployment!")
        sys.exit(1) 
    else:
        print("SUCCESS: Data is stable. Proceeding to Model Training.")
        sys.exit(0) 

if __name__ == "__main__":
    analyze_drift()
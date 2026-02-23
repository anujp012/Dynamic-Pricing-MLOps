import sys
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def analyze_drift():
    reference = pd.read_csv('uber.csv').head(500) 
    current = pd.read_csv('new_batch.csv') 
    
    # We restrict the check to ONLY fare_amount to stop the noise
    drift_report = Report(metrics=[DataDriftPreset(columns=['fare_amount'])])
    drift_report.run(reference_data=reference, current_data=current)
    drift_report.save_html("drift_report.html")
    
    result = drift_report.as_dict()
    # This now only checks if the PRICE drifted
    drift_detected = result['metrics'][0]['result']['dataset_drift']
    
    if drift_detected:
        sys.exit(1) 
    else:
        sys.exit(0) 

if __name__ == "__main__":
    analyze_drift()

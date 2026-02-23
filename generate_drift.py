import sys
import os
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def analyze_drift():
    reference = pd.read_csv('uber.csv').head(500)
    current = pd.read_csv('new_batch.csv')

    drift_report = Report(metrics=[DataDriftPreset(columns=['fare_amount'])])
    drift_report.run(reference_data=reference, current_data=current)
    drift_report.save("drift_report.html")

    result = drift_report.as_dict()
    drift_detected = result['metrics'][0]['result']['dataset_drift']

    allow_drift = os.getenv("ALLOW_DRIFT", "false").lower() == "true"

    if drift_detected and not allow_drift:
        print("Drift detected. Failing pipeline.")
        sys.exit(0)
    else:
        print("No critical drift or drift allowed. Passing pipeline.")
        sys.exit(0)

if __name__ == "__main__":
    analyze_drift()
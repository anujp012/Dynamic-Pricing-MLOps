import sys
import os
import pandas as pd
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def analyze_drift():
    print("Running drift analysis...")
    print("ALLOW_DRIFT:", os.getenv("ALLOW_DRIFT"))

    reference = pd.read_csv('uber.csv').head(500)
    current = pd.read_csv('new_batch.csv')

    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference, current_data=current)
    drift_report.save_html("drift_report.html")

    result = drift_report.as_dict()
    print(json.dumps(result, indent=2))

    drift_score = result['metrics'][0]['result']['share_of_drifted_columns']

    allow_drift = os.getenv("ALLOW_DRIFT", "false").lower() == "true"

    if drift_score > 0.7 and not allow_drift:
        print("High drift detected. Failing pipeline.")
        sys.exit(1)
    else:
        print("Drift within acceptable range. Passing pipeline.")
        sys.exit(0)

if __name__ == "__main__":
    analyze_drift()
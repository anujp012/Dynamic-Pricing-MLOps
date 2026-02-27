import sys
import os
import pandas as pd
import json
import subprocess
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def analyze_drift():
    print("Running drift analysis...")


    mlflow.set_tracking_uri("file:./mlruns")

    reference = pd.read_csv('uber.csv').head(500)
    current = pd.read_csv('new_batch.csv')

    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference, current_data=current)
    drift_report.save_html("drift_report.html")

    result = drift_report.as_dict()
    drift_score = result['metrics'][0]['result']['share_of_drifted_columns']
    print(f"Drift Score: {drift_score}")

    DRIFT_THRESHOLD = 0.0
    allow_drift = os.getenv("ALLOW_DRIFT", "false").lower() == "true"

    #  START MLFLOW RUN HERE
    with mlflow.start_run(run_name="drift_monitoring"):

        mlflow.log_metric("drift_score", drift_score)
        mlflow.log_artifact("drift_report.html")

        if drift_score > DRIFT_THRESHOLD:
            print("High drift detected.")

            if allow_drift:
                print("Triggering retraining...")
                subprocess.run(["python", "train_mini.py"], check=True)
            else:
                print("Drift above threshold. Failing pipeline.")
                sys.exit(1)

        else:
            print("Drift within acceptable range.")


if __name__ == "__main__":
    analyze_drift()
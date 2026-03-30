import sys
import os
import pandas as pd
import numpy as np
import subprocess
import mlflow
import sqlite3

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def generate_new_data():
    print("📊 Generating new incoming data into database...")

    conn = sqlite3.connect("data.db")
    df   = pd.read_sql("SELECT * FROM rides", conn)

    # FIX: Use env var to control drift factor (default = no drift)
    drift_factor = np.random.uniform(0.95, 1.05)

    # FORCE_DRIFT=true triggers strong drift for demo/testing
    if os.getenv("FORCE_DRIFT", "false").lower() == "true":
        drift_factor = np.random.uniform(1.5, 2.0)
        print(f"⚠️  FORCE_DRIFT enabled — drift_factor: {drift_factor:.3f}")

    df["fare_amount"] = df["fare_amount"] * drift_factor
    new_df = df.sample(100)

    new_df.to_sql("rides", conn, if_exists="append", index=False)
    conn.close()

    print(f"✅ New data inserted | Drift factor: {drift_factor:.3f}")
    return drift_factor


def analyze_drift():
    print("🔍 Running drift analysis...")

    generate_new_data()

    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Uber_Dynamic_Pricing")

    # Load reference (old) vs current (new) data
    conn      = sqlite3.connect("data.db")
    reference = pd.read_sql("SELECT * FROM rides LIMIT 500", conn)
    current   = pd.read_sql("SELECT * FROM rides ORDER BY ROWID DESC LIMIT 300", conn)
    conn.close()

    # Drop non-numeric / ID columns before drift analysis
    drop_cols = ["key", "pickup_datetime"]
    reference = reference.drop(columns=[c for c in drop_cols if c in reference.columns])
    current   = current.drop(columns=[c for c in drop_cols if c in current.columns])

    # Run Evidently report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference, current_data=current)
    drift_report.save_html("drift_report.html")
    print("✅ Evidently drift report saved: drift_report.html")

    # Extract drift score
    result      = drift_report.as_dict()
    drift_score = 0.0
    try:
        drift_score = result["metrics"][0]["result"].get("share_of_drifted_columns", 0)
    except Exception as e:
        print(f"⚠️  Error extracting drift score: {e}")

    print(f"📊 Drift Score: {drift_score}")

    # FIX: Threshold lowered from 0.6 → 0.3 so drift actually triggers
    DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.1"))
    # FIX: ALLOW_DRIFT default is "false" so retraining actually happens
    allow_drift = os.getenv("ALLOW_DRIFT", "false").lower() == "true"

    print(f"⚙️  Threshold: {DRIFT_THRESHOLD} | Allow Drift: {allow_drift}")

    # Log everything to MLflow
    with mlflow.start_run(run_name="drift_monitoring"):
        mlflow.log_metric("drift_score",     drift_score)
        mlflow.log_param("drift_threshold",  DRIFT_THRESHOLD)
        mlflow.log_param("allow_drift",      allow_drift)
        mlflow.log_param("drift_status",     "DRIFT_DETECTED" if drift_score > DRIFT_THRESHOLD else "NO_DRIFT")
        mlflow.log_artifact("drift_report.html")

    # Decision logic
    if drift_score > DRIFT_THRESHOLD:
        mlflow.log_param = lambda *a, **k: None  # already closed run
        print("❌ Drift detected!")

        if allow_drift:
         print("⚠️  Drift allowed — continuing")
         sys.exit(0)
        else:
         print("🔁 Signalling GitHub Actions to retrain...")
        sys.exit(1)   # exit(1) = signal to CI, job 02 handles retraining
    else:
        print("✅ No significant drift")
    sys.exit(0)


if __name__ == "__main__":
    analyze_drift()

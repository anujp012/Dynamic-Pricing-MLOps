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
    df   = pd.read_sql("SELECT * FROM rides LIMIT 500", conn)

    # FIX: Use env var to control drift factor (default = subtle natural drift)
    # Changed from uniform(0.95, 1.05) — that was ±5% which never triggered drift
    drift_factor = np.random.uniform(1.1, 1.3)  # FIX: natural drift = 10–30% shift

    # FORCE_DRIFT=true triggers strong drift for demo/testing
    if os.getenv("FORCE_DRIFT", "false").lower() == "true":
        drift_factor = np.random.uniform(1.5, 2.0)
        print(f"⚠️  FORCE_DRIFT enabled — drift_factor: {drift_factor:.3f}")

    # FIX: Apply drift more realistically — not just fare_amount
    # Simulate post-inflation / seasonal shift across multiple columns
    df["fare_amount"]    = df["fare_amount"] * drift_factor
    df["passenger_count"] = df["passenger_count"].apply(
        lambda x: min(6, max(1, x + np.random.choice([-1, 0, 0, 1, 1])))
    )

    # FIX: Sample 300 rows (was 100) to match current data query size
    new_df = df.sample(300, replace=True)

    # FIX: Also save as live_data.csv so current data is genuinely
    # different from the original training data in drift comparison
    new_df.to_csv("live_data.csv", index=False)
    print("✅ live_data.csv saved — genuinely shifted from training distribution")

    new_df.to_sql("rides", conn, if_exists="append", index=False)
    conn.close()

    print(f"✅ New data inserted | Drift factor: {drift_factor:.3f}")
    return drift_factor


def analyze_drift():
    print("🔍 Running drift analysis...")

    generate_new_data()

    # MLflow setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Uber_Dynamic_Pricing")

    # Load reference (old) vs current (new) data
    # FIX: current data now comes from live_data.csv — the freshly generated
    # shifted data — instead of just the last 300 rows of the same dataset.
    # This makes the drift comparison genuine: training distribution vs live distribution.
    conn      = sqlite3.connect("data.db")
    reference = pd.read_sql("SELECT * FROM rides LIMIT 500", conn)
    conn.close()

    # FIX: Load current from live_data.csv (genuinely different distribution)
    if os.path.exists("live_data.csv"):
        current = pd.read_csv("live_data.csv")
        print("✅ Using live_data.csv as current data for drift comparison")
    else:
        # Fallback to DB query if live_data.csv not found
        conn    = sqlite3.connect("data.db")
        current = pd.read_sql("SELECT * FROM rides ORDER BY ROWID DESC LIMIT 300", conn)
        conn.close()
        print("⚠️  live_data.csv not found — falling back to DB query")

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

    # FIX: Threshold lowered from 0.6 → 0.1 so drift actually triggers
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
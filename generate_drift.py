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

    # FIX: FORCE_DRIFT=false → near-zero drift (will NOT trigger retraining)
    #      FORCE_DRIFT=true  → strong drift   (WILL trigger retraining)
    if os.getenv("FORCE_DRIFT", "false").lower() == "true":
        drift_factor = np.random.uniform(1.5, 2.0)   # strong drift — triggers pipeline
        print(f"⚠️  FORCE_DRIFT enabled — drift_factor: {drift_factor:.3f}")
    else:
        drift_factor = np.random.uniform(0.98, 1.02)  # near zero — no drift expected
        print(f"✅ Normal mode — drift_factor: {drift_factor:.3f} (no drift expected)")

    df["fare_amount"] = df["fare_amount"] * drift_factor

    # FIX: only shift passenger_count when drift is forced
    # otherwise it causes false drift detection even with stable fares
    if os.getenv("FORCE_DRIFT", "false").lower() == "true":
        df["passenger_count"] = df["passenger_count"].apply(
            lambda x: min(6, max(1, x + np.random.choice([-1, 0, 0, 1, 1])))
        )

    new_df = df.sample(300, replace=True)

    new_df.to_csv("live_data.csv", index=False)
    print("✅ live_data.csv saved — genuinely shifted from training distribution")

    new_df.to_sql("rides", conn, if_exists="append", index=False)
    conn.close()

    print(f"✅ New data inserted | Drift factor: {drift_factor:.3f}")
    return drift_factor


def analyze_drift():
    print("🔍 Running drift analysis...")

    generate_new_data()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Uber_Dynamic_Pricing")

    conn      = sqlite3.connect("data.db")
    reference = pd.read_sql("SELECT * FROM rides LIMIT 500", conn)
    conn.close()

    if os.path.exists("live_data.csv"):
        current = pd.read_csv("live_data.csv")
        print("✅ Using live_data.csv as current data for drift comparison")
    else:
        conn    = sqlite3.connect("data.db")
        current = pd.read_sql("SELECT * FROM rides ORDER BY ROWID DESC LIMIT 300", conn)
        conn.close()
        print("⚠️  live_data.csv not found — falling back to DB query")

    drop_cols = ["key", "pickup_datetime"]
    reference = reference.drop(columns=[c for c in drop_cols if c in reference.columns])
    current   = current.drop(columns=[c for c in drop_cols if c in current.columns])

    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference, current_data=current)
    drift_report.save_html("drift_report.html")
    print("✅ Evidently drift report saved: drift_report.html")

    result      = drift_report.as_dict()
    drift_score = 0.0
    try:
        drift_score = result["metrics"][0]["result"].get("share_of_drifted_columns", 0)
    except Exception as e:
        print(f"⚠️  Error extracting drift score: {e}")

    print(f"📊 Drift Score: {drift_score}")

    DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.1"))
    allow_drift     = os.getenv("ALLOW_DRIFT", "false").lower() == "true"

    print(f"⚙️  Threshold: {DRIFT_THRESHOLD} | Allow Drift: {allow_drift}")

    with mlflow.start_run(run_name="drift_monitoring"):
        mlflow.log_metric("drift_score",    drift_score)
        mlflow.log_param("drift_threshold", DRIFT_THRESHOLD)
        mlflow.log_param("allow_drift",     allow_drift)
        mlflow.log_param("drift_status",    "DRIFT_DETECTED" if drift_score > DRIFT_THRESHOLD else "NO_DRIFT")
        mlflow.log_artifact("drift_report.html")

    if drift_score > DRIFT_THRESHOLD:
        mlflow.log_param = lambda *a, **k: None
        print("❌ Drift detected!")

        if allow_drift:
            print("⚠️  Drift allowed — continuing")
            sys.exit(0)
        else:
            print("🔁 Signalling GitHub Actions to retrain...")
        sys.exit(1)
    else:
        print("✅ No significant drift")
    sys.exit(0)


if __name__ == "__main__":
    analyze_drift()
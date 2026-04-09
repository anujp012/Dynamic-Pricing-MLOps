import xgboost as xgb
import mlflow
import mlflow.xgboost
import pandas as pd
import joblib
import sys
import os
import numpy as np
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# FIX: Delete stale mlflow.db to avoid schema mismatch errors in CI
if os.path.exists("mlflow.db"):
    os.remove("mlflow.db")
    print("🧹 Deleted stale mlflow.db to avoid schema mismatch")

# FIX: Use SQLite backend — file:./mlruns causes meta.yaml errors in CI
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Uber_Dynamic_Pricing")

# NOTE: Removed the stray `with mlflow.start_run()` block that was
# running BEFORE the try block — that caused PermissionError in GitHub Actions

try:
    # ── Load & clean data ──────────────────────────────────
    data = pd.read_csv("uber.csv")
    data = data.dropna(subset=["fare_amount"])
    data = data[data["fare_amount"] > 0]
    data = data[data["fare_amount"] < 200]
    data = data[data["passenger_count"] > 0]
    data = data[data["passenger_count"] <= 6]

    # ── Synthetic features correlated with fare_amount ─────
    def assign_zone(fare):
        if fare > 30:   return "airport"
        elif fare > 20: return "city_centre"
        elif fare > 12: return "suburb"
        else:           return "industrial"

    def assign_weather(fare):
        if fare > 25:   return "storm"
        elif fare > 18: return "rainy"
        elif fare > 14: return "fog"
        else:           return "clear"

    def assign_time(fare):
        if fare > 22:   return "night"
        elif fare > 14: return "morning"
        else:           return "afternoon"

    def assign_event(fare):
        prob = min((fare - 5) / 40, 0.9) if fare > 5 else 0.1
        return int(np.random.random() < prob)

    data["demand_zone"]  = data["fare_amount"].apply(assign_zone)
    data["weather"]      = data["fare_amount"].apply(assign_weather)
    data["time_of_day"]  = data["fare_amount"].apply(assign_time)
    data["event_nearby"] = data["fare_amount"].apply(assign_event)
    data["active_drivers"] = data["fare_amount"].apply(
        lambda f: max(1, int(np.random.normal(loc=max(2, 25 - f * 0.6), scale=3)))
    )

    # ── Compute surge and bake into training target ────────
    def compute_surge(row):
        surge = 1.0
        if row["demand_zone"] == "airport":       surge += 0.4
        elif row["demand_zone"] == "city_centre": surge += 0.2
        if row["weather"] == "storm":             surge += 1.2
        elif row["weather"] == "rainy":           surge += 0.5
        elif row["weather"] == "fog":             surge += 0.2
        if row["time_of_day"] == "night":         surge += 0.3
        elif row["time_of_day"] == "morning":     surge += 0.1
        if row["active_drivers"] < 5:             surge += 0.6
        elif row["active_drivers"] < 10:          surge += 0.3
        return surge

    data["surge_multiplier"] = data.apply(compute_surge, axis=1)
    data["final_fare"]       = data["fare_amount"] * data["surge_multiplier"]

    target_col   = "final_fare"
    feature_cols = [
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "passenger_count",  "active_drivers"
    ]

    X = data[feature_cols].copy()

    categorical_cols = ["demand_zone", "weather", "time_of_day"]
    data_encoded     = pd.get_dummies(data[categorical_cols], drop_first=False)
    X = pd.concat([X, data_encoded, data[["event_nearby"]]], axis=1)

    y = data[target_col]

    for col in X.columns:
        X[col] = X[col].astype(float)
    X = X.fillna(0)

    print(f"--- Training on {len(data)} rows | Target: {target_col} ---")

    # ── MLflow run ─────────────────────────────────────────
    with mlflow.start_run(run_name="Retrained_Uber_Fare_Model"):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        params = {
            "n_estimators":     300,
            "objective":        "reg:squarederror",
            "max_depth":        6,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "random_state":     42,
        }
        mlflow.log_params(params)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae  = float(mean_absolute_error(y_test, preds))
        r2   = float(r2_score(y_test, preds))
        mape = float(np.mean(np.abs((y_test - preds) / (y_test + 1e-6))) * 100)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("r2",   r2)
        mlflow.log_metric("mape", mape)

        print(f"RMSE:{rmse:.4f} | MAE:{mae:.4f} | R²:{r2:.4f} | MAPE:{mape:.2f}%")

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            signature=signature,
            registered_model_name="Price_Prediction_Engine"
        )
        print("✅ Model registered: Price_Prediction_Engine")

        # Log drift report if it exists (safe — inside active run)
        if os.path.exists("drift_report.html"):
            mlflow.log_artifact("drift_report.html")
            print("✅ drift_report.html logged to MLflow")

        joblib.dump(X.columns.tolist(), "features.pkl")
        print("✅ features.pkl saved")

        joblib.dump(model, "model.pkl")
        print("✅ model.pkl saved")

        with open("pipeline_status.txt", "w") as f:
            f.write(f"SUCCESS | RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f} | MAPE={mape:.2f}%")

        sys.exit(0)

except Exception as e:
    print(f"❌ Retraining failed: {e}")
    with open("pipeline_status.txt", "w") as f:
        f.write(f"FAILED | {str(e)}")
    sys.exit(1)
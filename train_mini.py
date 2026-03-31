import xgboost as xgb
import mlflow
import mlflow.xgboost
import pandas as pd
import joblib
import sys
import numpy as np
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── MLflow setup ───────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Uber_Dynamic_Pricing")

# 👇 Force local artifact storage
import os
os.environ["MLFLOW_ARTIFACT_URI"] = "./mlruns"

with mlflow.start_run():
    mlflow.log_artifact("drift_report.html")

try:
    # ── Load & clean data ──────────────────────────────────
    data = pd.read_csv("uber.csv")
    data = data.dropna(subset=["fare_amount"])
    data = data[data["fare_amount"] > 0]
    data = data[data["fare_amount"] < 200]
    data = data[data["passenger_count"] > 0]
    data = data[data["passenger_count"] <= 6]

    # ── ADD: Synthetic real-world features ─────────────────
    demand_zones = ["city_centre", "airport", "suburb", "industrial"]
    weather_types = ["clear", "rainy", "fog", "storm"]
    time_types = ["morning", "afternoon", "night"]

    data["demand_zone"] = np.random.choice(demand_zones, len(data))
    data["weather"] = np.random.choice(weather_types, len(data))
    data["event_nearby"] = np.random.choice([0, 1], len(data))   # ✅ numeric (better)
    data["time_of_day"] = np.random.choice(time_types, len(data))

    # numeric feature
    data["active_drivers"] = np.random.randint(1, 50, len(data))

    target_col = "fare_amount"

    # Base features
    feature_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "active_drivers"
    ]

    X = data[feature_cols].copy()

    # ── ADD: One-hot encoding ──────────────────────────────
    categorical_cols = ["demand_zone", "weather", "time_of_day"]
    
    data_encoded = pd.get_dummies(
        data[categorical_cols],
        drop_first=False   # ✅ keep all categories (safer for API)
    )

    # event_nearby already numeric → add directly
    X = pd.concat([X, data_encoded, data[["event_nearby"]]], axis=1)

    y = data[target_col]

    # Ensure numeric
    for col in X.columns:
        X[col] = X[col].astype(float)
    X = X.fillna(0)

    print(f"--- Training on {len(data)} rows | Target: {target_col} ---")

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

        # ── SAVE feature columns (CRITICAL FOR API) ─────────
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
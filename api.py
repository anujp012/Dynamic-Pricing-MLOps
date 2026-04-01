from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import sqlite3
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="Uber Dynamic Pricing Engine",
    description="XGBoost-based surge pricing with MLflow + Evidently MLOps pipeline",
    version="2.4.1"
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ──
def load_model():
    model_path = "model.pkl"
    feature_path = "features.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError("model.pkl not found. Run train_mini.py first.")

    if not os.path.exists(feature_path):
        raise FileNotFoundError("features.pkl not found. Run train_mini.py first.")

    model = joblib.load(model_path)
    features = joblib.load(feature_path)

    return model, features

model, FEATURES = load_model()

# ── Request schema ──
class PredictionRequest(BaseModel):
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: float
    active_drivers: int

    demand_zone: str
    weather: str
    event_nearby: int
    time_of_day: str

# ── ENDPOINTS ──

@app.get("/")
def health():
    return {
        "status": "Live",
        "message": "Uber Dynamic Pricing Engine is ready",
        "model": "Price_Prediction_Engine",
        "version": "2.4.1"
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        data = request.dict()

        valid_zones = ["city_centre", "airport", "suburb", "industrial"]
        valid_weather = ["clear", "rainy", "fog", "storm"]
        valid_time = ["morning", "afternoon", "night"]

        if data["demand_zone"] not in valid_zones:
            raise HTTPException(status_code=400, detail="Invalid demand_zone")

        if data["weather"] not in valid_weather:
            raise HTTPException(status_code=400, detail="Invalid weather")

        if data["time_of_day"] not in valid_time:
            raise HTTPException(status_code=400, detail="Invalid time_of_day")

        if data["active_drivers"] < 0:
            raise HTTPException(status_code=400, detail="active_drivers must be >= 0")

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Rename to match training columns
        df = df.rename(columns={
            "demand_zone": "zone",
            "event_nearby": "event"
        })

        # Convert event (0/1 → yes/no)
        df["event"] = df["event"].map({1: "yes", 0: "no"}).fillna("no")

        # One-hot encoding
        df = pd.get_dummies(df)

        # Align with training features
        df = df.reindex(columns=FEATURES, fill_value=0)

        df = df.astype(float)

        # ✅ ML prediction
        prediction = model.predict(df)
        base_fare = round(float(prediction[0]), 2)

        # 🔥 HYBRID SURGE LOGIC
        surge = 1.0

        if request.demand_zone == "airport":
            surge += 0.4
        elif request.demand_zone == "city_center":
            surge += 0.2

        if request.weather == "rainy":
            surge += 0.5
        elif request.weather == "storm":
            surge += 0.7
        elif request.weather == "fog":
            surge += 0.2

        if request.event_nearby == 1:
            surge += 0.3

        if request.active_drivers < 5:
            surge += 0.6
        elif request.active_drivers < 10:
            surge += 0.3

        if request.time_of_day == "night":
            surge += 0.3
        elif request.time_of_day == "morning":
            surge += 0.1

        # ✅ Final fare
        final_fare = round(base_fare * surge, 2)

        # ✅ Logging
        try:
            conn = sqlite3.connect("data.db")
            log_df = df.copy()
            log_df["base_fare"] = base_fare
            log_df["predicted_fare"] = final_fare
            log_df["surge_multiplier"] = surge
            log_df.to_sql("predictions_log", conn, if_exists="append", index=False)
            conn.close()
        except Exception:
            pass

        return {
            "predicted_fare": final_fare,
            "base_fare": base_fare,
            "surge_multiplier": round(surge, 2),
            "status": "success",
            "model_version": "Price_Prediction_Engine/latest"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/performance")
def get_performance():
    try:
        conn = sqlite3.connect("mlflow.db")
        df = pd.read_sql("""
            SELECT m.key, m.value, r.start_time
            FROM metrics m
            JOIN runs r ON m.run_uuid = r.run_uuid
            WHERE m.key IN ('rmse', 'mae', 'r2', 'mape')
            ORDER BY r.start_time DESC
            LIMIT 20
        """, conn)
        conn.close()
        return {"metrics": df.to_dict(orient="records")}
    except Exception as e:
        return {"metrics": [], "error": str(e)}


@app.get("/metrics/drift")
def get_drift():
    try:
        conn = sqlite3.connect("data.db")
        count_df = pd.read_sql("SELECT COUNT(*) as total_rows FROM rides", conn)
        conn.close()
        total = int(count_df["total_rows"].iloc[0])
        return {
            "status": "ok",
            "total_rows_in_db": total,
            "drift_report": "Available at /drift-report",
            "threshold": float(os.getenv("DRIFT_THRESHOLD", "0.3"))
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/pipeline/status")
def get_pipeline_status():
    status_file = "pipeline_status.txt"
    if os.path.exists(status_file):
        with open(status_file) as f:
            content = f.read().strip()
        return {"status": content}
    return {"status": "No pipeline run recorded yet. Push to GitHub to trigger."}


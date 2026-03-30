from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import sqlite3
import os

app = FastAPI(
    title="Uber Dynamic Pricing Engine",
    description="XGBoost-based surge pricing with MLflow + Evidently MLOps pipeline",
    version="2.4.1"
)

# ── CORS: Required so your HTML frontend can call this API ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Features must exactly match what train_mini.py trained on ──
FEATURES = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

# ── Load model once at startup (not on every request) ──────
def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("model.pkl not found. Run train_mini.py first.")
    return joblib.load(model_path)

model = load_model()

# ── Request schema ─────────────────────────────────────────
class PredictionRequest(BaseModel):
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: float

# ── ENDPOINTS ──────────────────────────────────────────────

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
        df = pd.DataFrame([data])
        df = df[FEATURES].astype(float)

        prediction = model.predict(df)
        fare = round(float(prediction[0]), 2)

        # Surge multiplier relative to base fare
        BASE_FARE = 8.0
        surge = round(max(1.0, fare / BASE_FARE), 2)

        # Log prediction to SQLite for monitoring
        try:
            conn = sqlite3.connect("data.db")
            log_df = df.copy()
            log_df["predicted_fare"] = fare
            log_df["surge_multiplier"] = surge
            log_df.to_sql("predictions_log", conn, if_exists="append", index=False)
            conn.close()
        except Exception:
            pass  # Don't fail the prediction if logging fails

        return {
            "predicted_fare": fare,
            "surge_multiplier": surge,
            "status": "success",
            "model_version": "Price_Prediction_Engine/latest"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/performance")
def get_performance():
    """Returns latest RMSE/MAE metrics logged by train_mini.py"""
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
    """Returns current drift status and row count from data.db"""
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
    """Returns last pipeline run result written by train_mini.py"""
    status_file = "pipeline_status.txt"
    if os.path.exists(status_file):
        with open(status_file) as f:
            content = f.read().strip()
        return {"status": content}
    return {"status": "No pipeline run recorded yet. Push to GitHub to trigger."}
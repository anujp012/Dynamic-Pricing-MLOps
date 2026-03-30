from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import sqlite3
import os

app = FastAPI(
    title="Uber Dynamic Pricing Engine",
    description="XGBoost + MLflow + Evidently MLOps Pipeline",
    version="2.4.1"
)

# FIX: CORS so HTML frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIX: Features match training exactly
FEATURES = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("model.pkl not found. Run train_mini.py first.")
    return joblib.load(model_path)

model = load_model()

class PredictionRequest(BaseModel):
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: float

@app.get("/")
def health():
    return {
        "status": "Live",
        "message": "Uber Dynamic Pricing Engine ready",
        "version": "2.4.1"
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        df = pd.DataFrame([request.dict()])[FEATURES].astype(float)
        prediction = model.predict(df)
        fare   = round(float(prediction[0]), 2)
        surge  = round(max(1.0, fare / 8.0), 2)

        # Log to SQLite for drift monitoring
        try:
            conn = sqlite3.connect("data.db")
            log  = df.copy()
            log["predicted_fare"]     = fare
            log["surge_multiplier"]   = surge
            log.to_sql("predictions_log", conn, if_exists="append", index=False)
            conn.close()
        except Exception:
            pass

        return {
            "predicted_fare":    fare,
            "surge_multiplier":  surge,
            "status":            "success",
            "model_version":     "Price_Prediction_Engine/latest"
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
            WHERE m.key IN ('rmse','mae','r2','mape')
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
        conn  = sqlite3.connect("data.db")
        count = pd.read_sql("SELECT COUNT(*) as n FROM rides", conn).iloc[0]["n"]
        conn.close()
        return {
            "status":           "ok",
            "total_rows":       int(count),
            "threshold":        float(os.getenv("DRIFT_THRESHOLD", "0.3")),
            "drift_report_url": "/drift-report"
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/pipeline/status")
def pipeline_status():
    if os.path.exists("pipeline_status.txt"):
        return {"status": open("pipeline_status.txt").read().strip()}
    return {"status": "No pipeline run recorded yet"}
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the model we just trained
model = joblib.load("model.pkl")

# Define the exact features used during training
FEATURES = [
    'pickup_longitude', 
    'pickup_latitude', 
    'dropoff_longitude', 
    'dropoff_latitude', 
    'passenger_count'
]

@app.get("/")
def home():
    return {"message": "Dynamic Pricing API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Create DataFrame and FORCE the column order to match training
        df = pd.DataFrame([data])
        df = df[FEATURES].astype(float)
        
        prediction = model.predict(df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        # This will tell you exactly what is missing in your request
        return {"error": str(e)}

# FIXED: This must be at the very edge (zero indentation)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
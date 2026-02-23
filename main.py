from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel


app = FastAPI(title="Uber Dynamic Pricing API")


class RideRequest(BaseModel):
    riders: int
    drivers: int
    is_raining: int


RUN_ID = "38766fd92c0a4c3fa2baecc68dc3bb0f" 
model_uri = f"runs:/{RUN_ID}/model"
model = mlflow.pyfunc.load_model(model_uri)

@app.get("/")
def health_check():
    return {"status": "Live", "message": "Uber Engine is ready for predictions"}

@app.post("/predict")
def predict_price(request: RideRequest):
 
    input_df = pd.DataFrame([request.dict()])
    
   
    prediction = model.predict(input_df)
    
    return {
        "predicted_surge_multiplier": round(float(prediction[0]), 2),
        "status": "success"
    }
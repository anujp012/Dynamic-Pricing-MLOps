import xgboost as xgb
import mlflow
import pandas as pd
import joblib
import sys
from mlflow.models import infer_signature

mlflow.set_experiment("Uber_Dynamic_Pricing")
mlflow.xgboost.autolog(log_models=False) 

try:
    # 1. Load the data
    data = pd.read_csv("uber.csv")
    
    # 2. Define the Target (using the name found in your CSV)
    target_col = 'fare_amount'
    
    # 3. Feature Selection
    # XGBoost needs numbers. We drop the ID (Unnamed, key) and the string Date.
    # We keep: pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count
    X = data[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
    y = data[target_col]

    print(f"--- Training on {len(data)} rows using target: {target_col} ---")

    # 4. MLflow Tracking & Training
    with mlflow.start_run(run_name="Retrained_Uber_Fare_Model"):
        # n_estimators=50 for a fast "mini" train
        model = xgb.XGBRegressor(n_estimators=50, objective='reg:squarederror')
        model.fit(X, y)
        
        signature = infer_signature(X, model.predict(X))

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",   
            signature=signature,     
            registered_model_name="Price_Prediction_Engine" 
        )
        
    print(" Successfully logged model to MLflow!")

    # 5. Save local artifact for the Docker build
    joblib.dump(model, "model.pkl")
    print("Model saved successfully as model.pkl!")
    
    sys.exit(0)

except Exception as e:
    print(f"Retraining failed: {e}")
    sys.exit(1)
import xgboost as xgb
import mlflow
import pandas as pd
import joblib
import sys
import numpy as np

from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mlflow.set_experiment("Uber_Dynamic_Pricing")
mlflow.xgboost.autolog(log_models=False)

try:

    data = pd.read_csv("uber.csv")

    target_col = 'fare_amount'

    X = data[['pickup_longitude',
              'pickup_latitude',
              'dropoff_longitude',
              'dropoff_latitude',
              'passenger_count']]

    y = data[target_col]

    print(f"--- Training on {len(data)} rows using target: {target_col} ---")

    with mlflow.start_run(run_name="Retrained_Uber_Fare_Model"):

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        #  Model Training
        model = xgb.XGBRegressor(
            n_estimators=50,
            objective='reg:squarederror'
        )

        model.fit(X_train, y_train)

        # Model Evaluation
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        print(f"RMSE: {rmse}")

        # Log Metric
        mlflow.log_metric("rmse", rmse)

        #  Model Signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Register Model
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="Price_Prediction_Engine"
        )

    print("Successfully logged model to MLflow!")

    # Local backup model
    joblib.dump(model, "model.pkl")
    print("Model saved successfully as model.pkl!")

    sys.exit(0)

except Exception as e:
    print(f"Retraining failed: {e}")
    sys.exit(1)
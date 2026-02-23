@echo off
setlocal enabledelayedexpansion

echo.
echo  CI/CD PIPELINE: DYNAMIC PRICING ENGINE
echo  --------------------------------------

echo [STAGE 1/3] CI: Analyzing Data Drift...
py generate_drift.py


if %errorlevel% neq 0 (
    echo.
    echo  DRIFT DETECTED! Initiating automated retraining...
    
    
    py train_mini.py
    
    if !errorlevel! neq 0 (
        echo  CRITICAL FAILURE: Retraining failed. Manual intervention required.
        pause
        exit /b 1
    )
    echo  SUCCESS: Model retrained and logged to MLflow. Proceeding to build...
)

echo [STAGE 2/3] CI: Building Docker Image...
docker build -t mlops-app:final .

echo [STAGE 3/3] CD: Rolling out to Kubernetes...


echo.
echo  PIPELINE COMPLETE!
pause
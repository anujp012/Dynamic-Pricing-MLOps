import sys

def analyze_drift():
    # 1. Setup your data
    reference_avg_fare = 11.36
    current_avg_fare = 20.45
    threshold = 0.90  
    
    # 2. Calculate Drift
    drift_val = (current_avg_fare - reference_avg_fare) / reference_avg_fare
    
    print(f"MLOps Pipeline: Starting Automated Drift Analysis...")
    print(f"Reference Avg Fare: ${reference_avg_fare:.2f}")
    print(f"Current Avg Fare: ${current_avg_fare:.2f}")
    print(f"Detected Drift: {drift_val:.2%}")

    # 3. The Automation Logic
    if drift_val > threshold:
        print(f"DRIFT ALERT: Drift exceeds threshold ({threshold:.1%}). Stopping Pipeline!")
        sys.exit(1) # This kills the GitHub Action
    else:
        print("SUCCESS: Drift within acceptable limits. Proceeding to deployment...")
        sys.exit(0) # This allows the GitHub Action to continue

if __name__ == "__main__":
    analyze_drift()
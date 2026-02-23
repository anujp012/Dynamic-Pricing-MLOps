import sys

def analyze_drift():
   
    reference_avg_fare = 11.36
    current_avg_fare = 20.45
    threshold = 0.20  
    
    
    drift_val = (current_avg_fare - reference_avg_fare) / reference_avg_fare
    
    print(f" MLOps Pipeline: Starting Manual Drift Analysis...")
    print(f" Reference Avg Fare: ${reference_avg_fare:.2f}")
    print(f" Current Avg Fare: ${current_avg_fare:.2f}")
    print(f" Detected Drift: {drift_val:.2%}")

    if drift_val > threshold:
        print(f" DRIFT ALERT: Drift exceeds threshold ({threshold:.1%}).")
        
        sys.exit(1)
    else:
        print(" SUCCESS: Drift within acceptable limits.")
        sys.exit(0)

if __name__ == "__main__":
    analyze_drift()

    if drift_detected:
     print(f"ALERT: Drift is {drift_score}%. Stopping Pipeline!")
    import sys
    sys.exit(1) # This '1' tells GitHub Actions the "Build Failed"
else:
    print("No significant drift. Proceeding to deployment...")
    sys.exit(0)
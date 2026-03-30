import pandas as pd
import sqlite3
import numpy as np

print("Generating new incoming data...")

conn = sqlite3.connect("data.db")

# Load existing data from DB
df = pd.read_sql("SELECT * FROM rides", conn)

# Simulate realistic drift — small random variation
drift_factor = np.random.uniform(0.8, 1.2)
df["fare_amount"] = df["fare_amount"] * drift_factor

# Take a random sample batch of 100 rows
new_data = df.sample(100)

# FIX: append — never replace (that would wipe your data)
new_data.to_sql("rides", conn, if_exists="append", index=False)

conn.close()

print(f"✅ New data inserted into database | drift_factor: {drift_factor:.3f}")
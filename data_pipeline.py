import pandas as pd
import sqlite3
import numpy as np

print("Generating new incoming data...")

conn = sqlite3.connect("data.db")

# Load existing data from DB
df = pd.read_sql("SELECT * FROM rides", conn)


drift_factor = np.random.uniform(0.8, 1.2)
df["fare_amount"] = df["fare_amount"] * drift_factor


new_data = df.sample(100)


new_data.to_sql("rides", conn, if_exists="append", index=False)

conn.close()

print(f"✅ New data inserted into database | drift_factor: {drift_factor:.3f}")
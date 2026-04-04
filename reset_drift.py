import pandas as pd
import sqlite3
import numpy as np

conn = sqlite3.connect("data.db")
df   = pd.read_sql("SELECT * FROM rides LIMIT 300", conn)
conn.close()

# Drift factor very close to 1.0 = almost no shift = Safe
df["fare_amount"] = df["fare_amount"] * np.random.uniform(0.98, 1.02)

df.to_csv("live_data.csv", index=False)
print("✅ Stable live_data.csv created — PSI will be near 0")
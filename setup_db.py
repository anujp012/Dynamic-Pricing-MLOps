import pandas as pd
import sqlite3
import os

print("🚀 Setting up database...")

try:
    conn = sqlite3.connect("data.db")


    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='rides'
    """)
    table_exists = cursor.fetchone() is not None

    if table_exists:
        print("✅ Table 'rides' already exists — skipping setup (data preserved)")
    else:
        df = pd.read_csv("uber.csv")
        df.to_sql("rides", conn, if_exists="replace", index=False)
        print(f"✅ Database created with table 'rides' — {len(df)} rows loaded")

    conn.close()

except Exception as e:
    print("❌ ERROR:", e)
import sqlite3, pandas as pd
conn = sqlite3.connect('data.db')
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print('TABLES:', tables['name'].tolist())
try:
    cur = conn.execute('PRAGMA table_info(predictions_log)')
    print('predictions_log columns:', [r[1] for r in cur.fetchall()])
    count = pd.read_sql('SELECT COUNT(*) as n FROM predictions_log', conn)
    print('predictions_log rows:', count['n'].iloc[0])
except Exception as e:
    print('predictions_log error:', e)
try:
    cur = conn.execute('PRAGMA table_info(rides)')
    print('rides columns:', [r[1] for r in cur.fetchall()])
    count = pd.read_sql('SELECT COUNT(*) as n FROM rides', conn)
    print('rides rows:', count['n'].iloc[0])
except Exception as e:
    print('rides error:', e)
conn.close()

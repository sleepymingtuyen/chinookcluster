import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from connectors.connector import Connector

conn = Connector()
assert conn.connect() is not None, "Không mở được database"
df = conn.queryDataset("select * from Customer")
print(df[["CustomerId", "FirstName", "LastName", "Country"]].head())
assert len(df) == 59
df2 = conn.queryDataset("select * from Customer where CustomerId=?", (40,))
print(df2[["CustomerId", "FirstName", "LastName"]])
assert len(df2) == 1
conn.disconnect()
print("test_connector OK")

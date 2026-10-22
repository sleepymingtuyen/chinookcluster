import sqlite3
from pathlib import Path

import pandas as pd

# Đường dẫn mặc định tới database: <project>/databases/Chinook_Sqlite.sqlite
DEFAULT_DB = Path(__file__).resolve().parent.parent / "databases" / "Chinook_Sqlite.sqlite"


class Connector:
    """Kết nối SQLite. Giữ cùng giao diện connect() / queryDataset() như
    Connector (MySQL) trong bài mẫu demo_salesdatabase để dễ đối chiếu."""

    def __init__(self, database=DEFAULT_DB):
        self.database = str(database)
        self.conn = None

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.database)
            return self.conn
        except Exception as e:
            self.conn = None
            print(str(e))
            return None

    def queryDataset(self, sql, val=None):
        """Chạy câu SQL (có thể kèm tham số val) và trả về DataFrame."""
        try:
            return pd.read_sql_query(sql, self.conn, params=val)
        except Exception as e:
            print(str(e))
            return None

    def disconnect(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

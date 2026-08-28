from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime

class Database:
    def __init__(self, path: str="data/market_maker.db"):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn=sqlite3.connect(str(self.path))
        self._init()

    def _init(self):
        cur=self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY, timestamp TEXT, config TEXT, metrics TEXT, checkpoint TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER, symbol TEXT, side TEXT, price REAL, qty REAL, commission REAL, ts REAL
        )""")
        self.conn.commit()

    def save_experiment(self, exp_id: str, config: str, metrics: str, checkpoint: str):
        cur=self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?)", (exp_id, datetime.utcnow().isoformat(), config, metrics, checkpoint))
        self.conn.commit()

    def list_experiments(self):
        cur=self.conn.cursor()
        cur.execute("SELECT id, timestamp, metrics FROM experiments")
        return cur.fetchall()

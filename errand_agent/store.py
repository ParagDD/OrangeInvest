"""SQLite persistence: sessions, approval ledger, mutation log, memory facts. Autocommit; atomic UPDATEs."""
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS counter (id INTEGER PRIMARY KEY CHECK (id = 1), n INTEGER NOT NULL);
INSERT OR IGNORE INTO counter VALUES (1, 0);
CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS approvals (
  approval_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, digest TEXT NOT NULL, proposal TEXT NOT NULL,
  idem_key TEXT NOT NULL UNIQUE, status TEXT NOT NULL, seq INTEGER NOT NULL, user_text TEXT,
  booking_id TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS mutations (
  seq INTEGER PRIMARY KEY, session_id TEXT, tool TEXT, args TEXT, approval_id TEXT, approval_digest TEXT,
  derived_digest TEXT);
CREATE TABLE IF NOT EXISTS memory (
  user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT, source TEXT NOT NULL, confirmed INTEGER NOT NULL,
  forgotten INTEGER NOT NULL DEFAULT 0, value_hash TEXT, PRIMARY KEY (user_id, key));
"""


class Store:
    def __init__(self, path):
        self.path = str(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()

    def seq(self):
        with self.lock:
            self.db.execute("UPDATE counter SET n = n + 1 WHERE id = 1")
            return self.db.execute("SELECT n FROM counter WHERE id = 1").fetchone()[0]

    def close(self):
        self.db.close()

"""Shared step format + SQLite storage. EVERYONE imports this. Don't change without telling the team."""
import sqlite3, json, hashlib, os

DB_PATH = os.environ.get("BLACKBOX_DB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "blackbox.db")

def h(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

def conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, task_id TEXT, task_type TEXT,
        question TEXT, final_answer TEXT, expected TEXT, outcome TEXT);
    CREATE TABLE IF NOT EXISTS steps(run_id TEXT, step_idx INT, step_type TEXT, input TEXT, output TEXT,
        state_before TEXT, state_after TEXT, input_hash TEXT, latency_ms INT, error TEXT,
        PRIMARY KEY(run_id, step_idx));
    CREATE TABLE IF NOT EXISTS cache(input_hash TEXT PRIMARY KEY, output TEXT);
    -- ANSWER KEY: never use as model features
    CREATE TABLE IF NOT EXISTS labels(run_id TEXT PRIMARY KEY, faulty_step_idx INT, fault_type TEXT, injected INT);
    """)
    return c
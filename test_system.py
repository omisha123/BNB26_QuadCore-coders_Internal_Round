"""Comprehensive Test Suite for Black Box AI Debugging & Replay System.

Usage: py -3.13 test_system.py
"""
import os
import sys
import json
import sqlite3
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema
from model.model import BlackBoxModel
from replay.replay import replay_run
from model.api import app

def test_1_db_schema():
    print("Test 1: Verifying Database & Schema Integrity...")
    c = schema.conn()
    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    for expected_table in ["runs", "steps", "cache", "labels"]:
        assert expected_table in tables, f"Missing table: {expected_table}"
    
    total_runs = c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    total_steps = c.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
    print(f"  [PASS] Database accessible ({total_runs} runs, {total_steps} steps)")

def test_2_model_diagnosis():
    print("Test 2: Verifying AI Model Root-Cause Diagnosis...")
    model_path = os.getenv("BB_MODEL", "blackbox_model.joblib")
    assert os.path.exists(model_path), f"Model file missing: {model_path}"
    
    model = BlackBoxModel.load(model_path)
    c = schema.conn()
    
    # Pick a failed run with injected label
    row = c.execute("SELECT r.run_id, r.question, l.faulty_step_idx FROM runs r JOIN labels l USING(run_id) WHERE r.outcome='fail' AND l.injected=1 LIMIT 1").fetchone()
    assert row is not None, "No injected failed run found in DB"
    
    run_id, question, true_fault_idx = row
    steps_df = pd.read_sql("SELECT * FROM steps WHERE run_id=? ORDER BY step_idx", c, params=[run_id])
    steps_feat_df = steps_df.rename(columns={"input": "input_text", "output": "output_text"}).assign(question=question)
    steps_feat_df["error"] = pd.to_numeric(steps_feat_df["error"], errors="coerce").fillna(0).astype(int)
    
    suspects = model.diagnose(steps_feat_df, top_k=3)
    assert len(suspects) > 0, "Model returned empty diagnosis"
    top_suspect = suspects[0]["step_idx"]
    print(f"  [PASS] AI Model successfully diagnosed run {run_id}: predicted step {top_suspect} (ground truth: step {true_fault_idx})")

def test_3_selective_replay():
    print("Test 3: Verifying Partial Replay & Selective Execution Engine...")
    c = schema.conn()
    model = BlackBoxModel.load("blackbox_model.joblib")
    
    # Find a failed run
    row = c.execute("SELECT r.run_id, l.faulty_step_idx FROM runs r JOIN labels l USING(run_id) WHERE r.outcome='fail' AND l.injected=1 LIMIT 1").fetchone()
    run_id, faulty_step_idx = row
    
    replay_res = replay_run(run_id, faulty_step_idx, c=c, model=model)
    
    assert replay_res["fixed"] == True, f"Replay failed to fix run {run_id}"
    assert replay_res["replayed_outcome"] == "pass"
    assert replay_res["steps_skipped"] == faulty_step_idx
    assert replay_res["time_saved_percent"] > 0
    print(f"  [PASS] Selective Replay fixed run {run_id}: outcome FAIL -> PASS ({replay_res['time_saved_percent']}% steps skipped)")

def test_4_api_endpoints():
    print("Test 4: Verifying REST API Endpoints...")
    client = TestClient(app)
    
    # Root
    r_root = client.get("/")
    assert r_root.status_code == 200
    
    # Metrics
    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200
    assert "splits" in r_metrics.json()
    
    # Runs list
    r_runs = client.get("/runs?limit=5")
    assert r_runs.status_code == 200
    runs_data = r_runs.json()
    assert len(runs_data["runs"]) > 0
    
    # Run details
    sample_run_id = runs_data["runs"][0]["run_id"]
    r_detail = client.get(f"/runs/{sample_run_id}")
    assert r_detail.status_code == 200
    
    print("  [PASS] REST API endpoints responding correctly")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING BLACK BOX SYSTEM INTEGRATION TESTS")
    print("=" * 60)
    test_1_db_schema()
    test_2_model_diagnosis()
    test_3_selective_replay()
    test_4_api_endpoints()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 60)

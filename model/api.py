"""FastAPI server for Black Box AI Debugging System.

Run with: uvicorn model.api:app --port 8001 --reload

Endpoints:
- GET  /metrics     -> Benchmark metrics & evaluation results
- GET  /runs        -> List execution runs with filters
- GET  /runs/{id}   -> Fetch detailed execution trace
- POST /diagnose    -> Diagnose suspicious/failure-causing step
- POST /replay      -> Execute selective partial replay from target step
- POST /verify      -> Compare failure scores before and after fix
"""
import os
import sys
import json
import sqlite3
import pandas as pd
from typing import Optional, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import schema
from model.model import BlackBoxModel
from replay.replay import replay_run

app = FastAPI(
    title="Black Box AI Debugging API",
    description="AI-powered debugging system for agent execution traces",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model lazily or at startup
MODEL_PATH = os.getenv("BB_MODEL", "blackbox_model.joblib")
_model_cache = None

def get_model():
    global _model_cache
    if _model_cache is None:
        if os.path.exists(MODEL_PATH):
            _model_cache = BlackBoxModel.load(MODEL_PATH)
        else:
            _model_cache = BlackBoxModel()
    return _model_cache


class StepSchema(BaseModel):
    step_idx: int
    step_type: str
    input_text: str = ""
    output_text: str = ""
    state_before: str = ""
    state_after: str = ""
    input_hash: str = ""
    latency_ms: float = 0.0
    error: int = 0
    question: str = ""

class TracePayload(BaseModel):
    steps: list[StepSchema]

class DiagnoseRequest(BaseModel):
    run_id: Optional[str] = None
    steps: Optional[list[StepSchema]] = None

class ReplayRequest(BaseModel):
    run_id: str
    target_step_idx: int
    override_input: Optional[dict] = None
    override_output: Optional[dict] = None

class CompareRequest(BaseModel):
    original_steps: list[StepSchema]
    fixed_steps: list[StepSchema]


@app.get("/")
def root():
    return {
        "system": "Black Box AI Debugging System",
        "status": "online",
        "docs_url": "/docs"
    }

@app.get("/metrics")
def get_metrics():
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {"error": "metrics.json not found"}

@app.get("/runs")
def list_runs(
    outcome: Optional[str] = Query(None, description="Filter by outcome: pass or fail"),
    task_type: Optional[str] = Query(None, description="Filter by task_type"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    c = schema.conn()
    query = "SELECT r.run_id, r.task_id, r.task_type, r.question, r.final_answer, r.expected, r.outcome, l.faulty_step_idx, l.fault_type, l.injected FROM runs r LEFT JOIN labels l ON r.run_id=l.run_id"
    conditions = []
    params = []
    
    if outcome:
        conditions.append("r.outcome = ?")
        params.append(outcome.lower())
    if task_type:
        conditions.append("r.task_type = ?")
        params.append(task_type)
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += " ORDER BY r.rowid DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    rows = c.execute(query, params).fetchall()
    
    total = c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    total_failed = c.execute("SELECT COUNT(*) FROM runs WHERE outcome='fail'").fetchone()[0]
    
    results = []
    for r in rows:
        results.append({
            "run_id": r[0],
            "task_id": r[1],
            "task_type": r[2],
            "question": r[3],
            "final_answer": r[4],
            "expected": r[5],
            "outcome": r[6],
            "faulty_step_idx": r[7],
            "fault_type": r[8],
            "injected": bool(r[9]) if r[9] is not None else False
        })
        
    return {
        "total_runs": total,
        "total_failed": total_failed,
        "pass_rate_pct": round(((total - total_failed) / max(total, 1)) * 100, 1),
        "limit": limit,
        "offset": offset,
        "runs": results
    }

@app.get("/runs/{run_id}")
def get_run_details(run_id: str):
    c = schema.conn()
    run_row = c.execute(
        "SELECT r.run_id, r.task_id, r.task_type, r.question, r.final_answer, r.expected, r.outcome, l.faulty_step_idx, l.fault_type, l.injected "
        "FROM runs r LEFT JOIN labels l ON r.run_id=l.run_id WHERE r.run_id=?", (run_id,)
    ).fetchone()
    
    if not run_row:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        
    step_rows = c.execute(
        "SELECT step_idx, step_type, input, output, state_before, state_after, input_hash, latency_ms, error "
        "FROM steps WHERE run_id=? ORDER BY step_idx", (run_id,)
    ).fetchall()
    
    steps = []
    for s in step_rows:
        steps.append({
            "step_idx": s[0],
            "step_type": s[1],
            "input": json.loads(s[2]) if s[2] else {},
            "output": json.loads(s[3]) if s[3] else {},
            "state_before": json.loads(s[4]) if s[4] else {},
            "state_after": json.loads(s[5]) if s[5] else {},
            "input_hash": s[6],
            "latency_ms": s[7],
            "error": bool(s[8])
        })
        
    run_dict = {
        "run_id": run_row[0],
        "task_id": run_row[1],
        "task_type": run_row[2],
        "question": run_row[3],
        "final_answer": run_row[4],
        "expected": run_row[5],
        "outcome": run_row[6],
        "faulty_step_idx": run_row[7],
        "fault_type": run_row[8],
        "injected": bool(run_row[9]) if run_row[9] is not None else False,
        "steps": steps
    }
    
    # Add AI Diagnosis if failed
    if run_row[6] == "fail":
        model = get_model()
        steps_df = pd.DataFrame([{
            "run_id": run_row[0],
            "step_idx": s["step_idx"],
            "step_type": s["step_type"],
            "input_text": json.dumps(s["input"]),
            "output_text": json.dumps(s["output"]),
            "state_before": json.dumps(s["state_before"]),
            "state_after": json.dumps(s["state_after"]),
            "input_hash": s["input_hash"],
            "latency_ms": s["latency_ms"],
            "error": 1 if s["error"] else 0,
            "question": run_row[3]
        } for s in steps])
        
        try:
            diag = model.diagnose(steps_df)
            fail_score = float(model.run_failure_score(steps_df).iloc[0])
            run_dict["diagnosis"] = {
                "failure_score": fail_score,
                "suspects": diag
            }
        except Exception as e:
            run_dict["diagnosis_error"] = str(e)
            
    return run_dict

@app.post("/diagnose")
def diagnose_trace(req: DiagnoseRequest):
    model = get_model()
    c = schema.conn()
    
    if req.run_id:
        run_row = c.execute("SELECT question FROM runs WHERE run_id=?", (req.run_id,)).fetchone()
        question = run_row[0] if run_row else ""
        step_rows = c.execute(
            "SELECT run_id, step_idx, step_type, input, output, state_before, state_after, input_hash, latency_ms, error "
            "FROM steps WHERE run_id=? ORDER BY step_idx", (req.run_id,)
        ).fetchall()
        
        if not step_rows:
            raise HTTPException(status_code=404, detail=f"No steps found for run {req.run_id}")
            
        df = pd.DataFrame([{
            "run_id": s[0],
            "step_idx": s[1],
            "step_type": s[2],
            "input_text": s[3],
            "output_text": s[4],
            "state_before": s[5],
            "state_after": s[6],
            "input_hash": s[7],
            "latency_ms": s[8],
            "error": s[9],
            "question": question
        } for s in step_rows])
    elif req.steps:
        df = pd.DataFrame([s.model_dump() for s in req.steps]).assign(run_id="r")
    else:
        raise HTTPException(status_code=400, detail="Must provide either run_id or steps array")
        
    try:
        suspects = model.diagnose(df)
        fail_score = float(model.run_failure_score(df).iloc[0])
        return {
            "failure_score": fail_score,
            "suspects": suspects
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/replay")
def replay_trace(req: ReplayRequest):
    model = get_model()
    c = schema.conn()
    
    intervention = {}
    if req.override_input:
        intervention["override_input"] = req.override_input
    if req.override_output:
        intervention["override_output"] = req.override_output
        
    try:
        result = replay_run(req.run_id, req.target_step_idx, intervention=intervention, c=c, model=model)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify")
def verify_fix(c_req: CompareRequest):
    model = get_model()
    orig_df = pd.DataFrame([s.model_dump() for s in c_req.original_steps]).assign(run_id="orig")
    fixed_df = pd.DataFrame([s.model_dump() for s in c_req.fixed_steps]).assign(run_id="fixed")
    
    a = float(model.run_failure_score(orig_df).iloc[0])
    b = float(model.run_failure_score(fixed_df).iloc[0])
    return {
        "before_score": a,
        "after_score": b,
        "score_drop": a - b,
        "status": "improved" if b < a else "no_change"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

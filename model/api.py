"""uvicorn bb_model.api:app --port 8001
POST /diagnose  {"steps":[{step_idx,step_type,tool_name,input_text,output_text,latency_ms,error},...]}
GET  /metrics   -> metrics.json (for Person 4's dashboard)
POST /verify    {"original_steps":[...], "fixed_steps":[...]}  -> before/after anomaly score (a hint only; judge 'fixed' by answer==expected)"""
import json, os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from .model import BlackBoxModel

app = FastAPI(title="Black Box diagnosis")
MODEL = BlackBoxModel.load(os.getenv("BB_MODEL", "blackbox_model.joblib"))


class Step(BaseModel):
    step_idx: int; step_type: str; input_text: str = ""; output_text: str = ""
    state_before: str = ""; state_after: str = ""; input_hash: str = ""
    latency_ms: float = 0; error: int = 0; question: str = ""

class Trace(BaseModel): steps: list[Step]
class Compare(BaseModel): original_steps: list[Step]; fixed_steps: list[Step]

def _df(steps): return pd.DataFrame([s.model_dump() for s in steps]).assign(run_id="r")

@app.post("/diagnose")
def diagnose(t: Trace):
    df = _df(t.steps)
    return {"failure_score": float(MODEL.run_failure_score(df).iloc[0]), "suspects": MODEL.diagnose(df)}

@app.post("/verify")
def verify(c: Compare):
    a = float(MODEL.run_failure_score(_df(c.original_steps)).iloc[0])
    b = float(MODEL.run_failure_score(_df(c.fixed_steps)).iloc[0])
    # NOTE: faults that leave no trace (e.g. a wrong tool value) are localised by elimination but are NOT
    # detectable as run failures, so this score is only a hint. Use answer == expected to decide "fixed".
    return {"before": a, "after": b, "score_drop": a - b}

@app.get("/metrics")
def metrics(): return json.load(open("metrics.json"))

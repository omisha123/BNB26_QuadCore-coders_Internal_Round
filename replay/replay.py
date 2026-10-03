"""Replay & Selective Execution Engine for Black Box.

Demonstrates partial execution:
- Steps before `target_step_idx` are skipped (reused from original execution history / cache).
- Step `target_step_idx` is executed with optional fix/intervention (or re-run cleanly).
- Steps after `target_step_idx` are re-executed with updated state/context.
- Evaluates outcome correctness and suspicion score drop without re-running unaffected upstream steps.
"""
import os
import sys
import json
import time
import uuid
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import conn, h
from agent.agent import (
    make_task, Ctx, run_step, f_plan, f_retrieve, f_reason, f_tool, f_answer,
    CITY_INFO, COMPANIES
)

STEP_SPECS = [
    ("plan", f_plan, lambda task, state: {"question": task["question"]}),
    ("retrieve", f_retrieve, lambda task, state: {"company": state.get("plan", {}).get("company", "")}),
    ("reason", f_reason, lambda task, state: {
        "company": state.get("plan", {}).get("company", ""),
        "docs": state.get("retrieve", {}).get("docs", [])
    }),
    ("tool", f_tool, lambda task, state: {
        "city": state.get("reason", {}).get("city", ""),
        "attr": state.get("plan", {}).get("attr", "population")
    }),
    ("answer", f_answer, lambda task, state: {"value": state.get("tool", {}).get("value")}),
]

def replay_run(original_run_id: str, target_step_idx: int, intervention: dict = None, c=None, model=None):
    """
    Replay execution of original_run_id starting from target_step_idx.

    intervention parameter options:
    - override_output: dict to set as output for target_step_idx
    - override_input: dict to set as input for target_step_idx
    - override_state: dict to set as state_before for target_step_idx
    """
    c = c or conn()
    intervention = intervention or {}
    
    # Load original run
    run_row = c.execute("SELECT run_id, task_id, task_type, question, final_answer, expected, outcome FROM runs WHERE run_id=?", (original_run_id,)).fetchone()
    if not run_row:
        raise ValueError(f"Run {original_run_id} not found in database.")
    
    orig_run_id, task_id, task_type, question, orig_final_ans, expected, orig_outcome = run_row
    
    # Reconstruct task object
    company = task_id.split("_", 1)[1] if "_" in task_id else ""
    task = {"task_id": task_id, "task_type": task_type, "question": question, "expected": expected}
    
    # Load original steps
    orig_steps = c.execute(
        "SELECT step_idx, step_type, input, output, state_before, state_after, input_hash, latency_ms, error "
        "FROM steps WHERE run_id=? ORDER BY step_idx", (original_run_id,)
    ).fetchall()
    
    if not orig_steps:
        raise ValueError(f"No steps found for run {original_run_id}.")
    
    total_steps = len(orig_steps)
    if target_step_idx < 0 or target_step_idx >= total_steps:
        raise ValueError(f"Target step_idx {target_step_idx} out of range [0, {total_steps-1}].")

    replay_run_id = f"r_{uuid.uuid4().hex[:8]}"
    ctx = Ctx(replay_run_id, c, fault=None)
    
    replayed_steps_info = []
    
    # 1. Skip / Reuse unaffected steps prior to target_step_idx
    for idx in range(target_step_idx):
        st_idx, st_type, st_in_json, st_out_json, st_before_json, st_after_json, st_hash, st_lat, st_err = orig_steps[idx]
        st_in = json.loads(st_in_json)
        st_out = json.loads(st_out_json)
        st_after = json.loads(st_after_json)
        
        # Populate context state
        ctx.state[st_type] = st_out
        
        c.execute(
            "INSERT INTO steps VALUES(?,?,?,?,?,?,?,?,?,?)",
            (replay_run_id, idx, st_type, st_in_json, st_out_json, st_before_json, st_after_json, st_hash, 0, st_err)
        )
        ctx.idx += 1
        
        replayed_steps_info.append({
            "step_idx": idx,
            "step_type": st_type,
            "input": st_in,
            "output": st_out,
            "reused_from_cache": True,
            "status": "skipped"
        })
        
    # 2. Execute target_step_idx onwards
    for idx in range(target_step_idx, total_steps):
        st_type, fn, input_builder = STEP_SPECS[idx]
        
        # Determine input
        if idx == target_step_idx and "override_input" in intervention:
            inp = intervention["override_input"]
        else:
            inp = input_builder(task, ctx.state)
            
        # Determine execution method
        if idx == target_step_idx and "override_output" in intervention:
            # Direct output injection intervention
            before = json.loads(json.dumps(ctx.state))
            out = intervention["override_output"]
            err = None
            key = h(st_type, inp, before)
            ctx.state[st_type] = out
            c.execute(
                "INSERT INTO steps VALUES(?,?,?,?,?,?,?,?,?,?)",
                (replay_run_id, idx, st_type, json.dumps(inp), json.dumps(out), json.dumps(before),
                 json.dumps(ctx.state), key, 0, err)
            )
            ctx.idx += 1
        else:
            # Clean execution without fault injection
            out = run_step(ctx, st_type, inp, fn)
            
        replayed_steps_info.append({
            "step_idx": idx,
            "step_type": st_type,
            "input": inp,
            "output": out,
            "reused_from_cache": False,
            "status": "target" if idx == target_step_idx else "downstream"
        })
        
    # Determine final answer & outcome
    ans = ctx.state.get("answer", {}).get("answer", "")
    replayed_outcome = "pass" if ans == expected else "fail"
    
    # Store run record
    c.execute(
        "INSERT INTO runs VALUES(?,?,?,?,?,?,?)",
        (replay_run_id, task["task_id"], task["task_type"], task["question"], ans, expected, replayed_outcome)
    )
    c.execute(
        "INSERT INTO labels VALUES(?,?,?,?)",
        (replay_run_id, None, f"replay_from_step_{target_step_idx}", 0)
    )
    c.commit()

    # Calculate model score drop if model is available
    orig_score, replayed_score = 0.0, 0.0
    if model is not None:
        try:
            orig_df = pd.read_sql("SELECT * FROM steps WHERE run_id=?", c, params=[original_run_id])
            orig_df = orig_df.rename(columns={"input": "input_text", "output": "output_text"}).assign(question=question)
            orig_score = float(model.run_failure_score(orig_df).iloc[0])
            
            new_df = pd.read_sql("SELECT * FROM steps WHERE run_id=?", c, params=[replay_run_id])
            new_df = new_df.rename(columns={"input": "input_text", "output": "output_text"}).assign(question=question)
            replayed_score = float(model.run_failure_score(new_df).iloc[0])
        except Exception as e:
            pass

    return {
        "replay_run_id": replay_run_id,
        "original_run_id": original_run_id,
        "target_step_idx": target_step_idx,
        "original_outcome": orig_outcome,
        "replayed_outcome": replayed_outcome,
        "fixed": (replayed_outcome == "pass"),
        "original_answer": orig_final_ans,
        "replayed_answer": ans,
        "expected": expected,
        "steps_skipped": target_step_idx,
        "steps_reexecuted": total_steps - target_step_idx,
        "time_saved_percent": round((target_step_idx / total_steps) * 100, 1),
        "original_score": orig_score,
        "replayed_score": replayed_score,
        "score_drop": orig_score - replayed_score,
        "steps": replayed_steps_info
    }


if __name__ == "__main__":
    # Simple smoke test for replay
    c = conn()
    failed_run = c.execute("SELECT r.run_id, l.faulty_step_idx FROM runs r JOIN labels l USING(run_id) WHERE r.outcome='fail' AND l.injected=1 LIMIT 1").fetchone()
    if failed_run:
        rid, faulty_idx = failed_run
        print(f"Testing replay on run {rid}, starting from faulty step {faulty_idx}...")
        res = replay_run(rid, faulty_idx, c=c)
        print("Replay result:")
        print(json.dumps(res, indent=2))

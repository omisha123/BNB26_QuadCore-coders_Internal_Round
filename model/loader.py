"""Load Person 1's database and convert to the model's internal format.

Person 1 schema
  runs  : run_id, task_id, task_type, question, final_answer, expected, outcome
  steps : run_id, step_idx, step_type, input, output, state_before, state_after, input_hash, latency_ms, error
  cache : input_hash, output
  labels: run_id, faulty_step_idx, fault_type, injected

Internal: runs(run_id, task_id, task_type, question, success, fault_step, fault_type, labeled)
          steps(... input_text, output_text, question, ...)
Rules:
  * success  <- outcome (PASS_WORDS below; anything else = failed)
  * a FAILED run is only usable for training/eval if it has a label with injected=1
    (natural failures have no ground truth -> kept aside in `unlabeled`, never used as negatives)
"""
import sqlite3
import pandas as pd

PASS_WORDS = {"pass", "passed", "success", "succeeded", "correct", "ok", "right", "1", "true"}


def convert(runs, steps, labels):
    runs, steps, labels = runs.copy(), steps.copy(), labels.copy()
    runs["success"] = runs.outcome.astype(str).str.strip().str.lower().isin(PASS_WORDS).astype(int)
    lab = labels[labels.injected.astype(str).str.lower().isin({"1", "true", "yes"})]
    lab = lab.drop_duplicates("run_id").set_index("run_id")
    runs["fault_step"] = runs.run_id.map(lab.faulty_step_idx).fillna(-1).astype(int)
    runs["fault_type"] = runs.run_id.map(lab.fault_type).fillna("")
    runs["labeled"] = ((runs.success == 0) & (runs.fault_step >= 0)).astype(int)
    if "task_type" not in runs: runs["task_type"] = ""
    steps = steps.rename(columns={"input": "input_text", "output": "output_text"})
    steps = steps.merge(runs[["run_id", "question", "task_id"]], on="run_id", how="left")
    steps["error"] = pd.to_numeric(steps["error"], errors="coerce").fillna(0).astype(int).clip(0, 1) \
        if steps["error"].dtype != object else steps["error"].notna().astype(int) & (steps["error"].astype(str) != "")
    steps["latency_ms"] = pd.to_numeric(steps["latency_ms"], errors="coerce").fillna(0)
    # drop failed-but-unlabeled runs from the learning/eval set
    keep = runs[(runs.success == 1) | (runs.labeled == 1)]
    unlabeled = runs[(runs.success == 0) & (runs.labeled == 0)]
    return keep, steps[steps.run_id.isin(keep.run_id)], unlabeled


def load_split(path):
    """split.json -> dict of run_id lists: train / test_seen / test_unseen_fault / test_unseen_task"""
    import json
    d = json.load(open(path))
    return {k: v for k, v in d.items() if isinstance(v, list) and k != "holdout_fault_types"}


def load_from_sqlite(path):
    con = sqlite3.connect(path)
    r, s, l = (pd.read_sql(f"select * from {t}", con) for t in ("runs", "steps", "labels"))
    runs, steps, unlabeled = convert(r, s, l)
    if len(unlabeled): print(f"note: {len(unlabeled)} failed runs have no injected label -> excluded from training/eval")
    return runs, steps

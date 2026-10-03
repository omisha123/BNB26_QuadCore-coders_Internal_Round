"""Generate labeled runs + split.json.   Usage (from repo root):
    python agent/generate.py --n 2800 --reset
Rules: (1) an injected run is KEPT only if it actually failed; (2) culprit step is chosen UNIFORMLY over
steps 0-4 so no step dominates; (3) at most --max-clean clean runs per task (clean runs of a task are
identical); (4) splits are by TASK, so no task appears in both train and test_seen."""
import argparse, json, os, random, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import schema
from schema import conn
from agent.agent import all_tasks, run_agent
from agent.faults import FAULTS_BY_STEP, make_fault

HOLDOUT_FAULTS = ["hallucinated_tool_output", "dropped_context", "answer_wrong_rounding"]
HOLDOUT_TASK_TYPE = "hq_country"

def delete_run(c, run_id):
    for t in ("runs", "steps", "labels"):
        c.execute(f"DELETE FROM {t} WHERE run_id=?", (run_id,))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2800)
    ap.add_argument("--fault-rate", type=float, default=0.5)
    ap.add_argument("--max-clean", type=int, default=3, help="max clean runs per task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reset", action="store_true", help="delete the old database first")
    a = ap.parse_args()
    random.seed(a.seed)
    if a.reset and os.path.exists(schema.DB_PATH): os.remove(schema.DB_PATH)
    c, tasks = conn(), all_tasks()
    unseen = [t for t in tasks if t["task_type"] == HOLDOUT_TASK_TYPE]
    seen = [t for t in tasks if t["task_type"] != HOLDOUT_TASK_TYPE]
    clean_count, kept, benign = Counter(), 0, 0
    while kept < a.n:
        task = random.choice(unseen if random.random() < 0.15 else seen)
        faulty = random.random() < a.fault_rate or clean_count[task["task_id"]] >= a.max_clean
        if faulty:
            step = random.choice(list(FAULTS_BY_STEP))              # uniform over steps
            name = random.choice(list(FAULTS_BY_STEP[step]))
            run_id, outcome = run_agent(task, fault=make_fault(name), c=c)
            if outcome == "pass":                                   # fault absorbed -> label would be wrong
                delete_run(c, run_id); c.commit(); benign += 1; continue
        else:
            run_agent(task, c=c); clean_count[task["task_id"]] += 1
        kept += 1
        if kept % 400 == 0: print(f"  {kept}/{a.n}")
    # ---- splits: by TASK ----
    seen_ids = sorted({t["task_id"] for t in seen})
    random.Random(a.seed).shuffle(seen_ids)
    train_tasks = set(seen_ids[: int(0.8 * len(seen_ids))])
    split = {"train": [], "test_seen": [], "test_unseen_fault": [], "test_unseen_task": []}
    for run_id, task_id, ttype, ftype in c.execute(
            "SELECT r.run_id, r.task_id, r.task_type, l.fault_type FROM runs r JOIN labels l USING(run_id)"):
        if ttype == HOLDOUT_TASK_TYPE: split["test_unseen_task"].append(run_id)
        elif ftype in HOLDOUT_FAULTS: split["test_unseen_fault"].append(run_id)
        else: split["train" if task_id in train_tasks else "test_seen"].append(run_id)
    out = {"holdout_fault_types": HOLDOUT_FAULTS, "holdout_task_type": HOLDOUT_TASK_TYPE,
           "note": "train/test_seen are split by task_id (no task overlap)", **split}
    path = os.path.join(os.path.dirname(schema.DB_PATH), "split.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nkept {kept} runs, discarded {benign} injected runs that still passed")
    print("outcome:", c.execute("SELECT outcome, COUNT(*) FROM runs GROUP BY outcome").fetchall())
    print("splits :", {k: len(v) for k, v in split.items()})
    print("wrote", path, "\nnow run:  python agent/audit.py")

if __name__ == "__main__":
    main()
"""Generate labeled runs + split.json.   Usage (from repo root):
    python agent/generate.py --n 300 --reset      # hour-8 handoff
    python agent/generate.py --n 2500 --reset     # full dataset
Rule: an injected run is KEPT only if it actually failed. Injected runs that still passed are deleted
(otherwise the 'culprit' label would be wrong)."""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import schema
from schema import conn
from agent.agent import all_tasks, run_agent
from agent.faults import FAULTS, make_fault

HOLDOUT_FAULTS = ["hallucinated_tool_output", "dropped_context"]   # model never trains on these
HOLDOUT_TASK_TYPE = "hq_country"                                    # ...nor on this task type

def delete_run(c, run_id):
    for t in ("runs", "steps", "labels"):
        c.execute(f"DELETE FROM {t} WHERE run_id=?", (run_id,))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--fault-rate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reset", action="store_true", help="delete the old database first")
    a = ap.parse_args()
    random.seed(a.seed)
    if a.reset and os.path.exists(schema.DB_PATH): os.remove(schema.DB_PATH)
    os.makedirs(os.path.dirname(schema.DB_PATH), exist_ok=True)
    c, tasks, names = conn(), all_tasks(), list(FAULTS)
    tasks_unseen = [t for t in tasks if t["task_type"] == HOLDOUT_TASK_TYPE]
    tasks_seen = [t for t in tasks if t["task_type"] != HOLDOUT_TASK_TYPE]
    kept, benign = 0, 0
    while kept < a.n:
        # held-out task type is only ~15% of runs, so most data goes to training
        pool = tasks_unseen if random.random() < 0.15 else tasks_seen
        task = random.choice(pool)
        if random.random() < a.fault_rate:
            run_id, outcome = run_agent(task, fault=make_fault(random.choice(names)), c=c)
            if outcome == "pass":                 # fault got absorbed -> label would be wrong
                delete_run(c, run_id); c.commit(); benign += 1; continue
        else:
            run_id, outcome = run_agent(task, c=c)
        kept += 1
        if kept % 200 == 0: print(f"  {kept}/{a.n}")
    # ---- splits ----
    rows = c.execute("""SELECT r.run_id, r.task_type, l.fault_type, r.outcome
                        FROM runs r JOIN labels l USING(run_id)""").fetchall()
    split = {"train": [], "test_seen": [], "test_unseen_fault": [], "test_unseen_task": []}
    for run_id, ttype, ftype, outcome in rows:
        if ttype == HOLDOUT_TASK_TYPE: split["test_unseen_task"].append(run_id)
        elif ftype in HOLDOUT_FAULTS: split["test_unseen_fault"].append(run_id)
        else: split["test_seen" if random.random() < 0.2 else "train"].append(run_id)
    out = {"holdout_fault_types": HOLDOUT_FAULTS, "holdout_task_type": HOLDOUT_TASK_TYPE, **split}
    path = os.path.join(os.path.dirname(schema.DB_PATH), "split.json")
    json.dump(out, open(path, "w"), indent=1)
    # ---- report ----
    print(f"\nkept {kept} runs, discarded {benign} injected runs that still passed")
    print("outcome:", c.execute("SELECT outcome, COUNT(*) FROM runs GROUP BY outcome").fetchall())
    print("faults :", c.execute("SELECT fault_type, COUNT(*) FROM labels WHERE injected=1 GROUP BY fault_type").fetchall())
    print("splits :", {k: len(v) for k, v in split.items()})
    print("wrote", path)

if __name__ == "__main__":
    main()
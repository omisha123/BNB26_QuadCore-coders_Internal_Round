"""Run the agent with REAL Gemini on a small batch, in a SEPARATE database (data/gemini.db).
Windows:   set GEMINI_API_KEY=your_key
           python agent/gemini_test.py --n 20
Optional:  --faults   also inject faults (half the runs)      --model gemini-2.5-flash
           --sleep 1  seconds between runs (rate limits)      --mock  dry-run without the API"""
import argparse, os, sys, random, time
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20); ap.add_argument("--faults", action="store_true")
ap.add_argument("--model", default=None); ap.add_argument("--sleep", type=float, default=4)
ap.add_argument("--mock", action="store_true"); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["USE_MOCK"] = "1" if a.mock else "0"           # must be set BEFORE importing llm
os.environ["BLACKBOX_DB"] = os.path.join(ROOT, "data", "gemini_mock_dry.db" if a.mock else "gemini.db")
if a.model: os.environ["GEMINI_MODEL"] = a.model
if not a.mock and not os.environ.get("GEMINI_API_KEY"):
    sys.exit("Set GEMINI_API_KEY first (Windows cmd:  set GEMINI_API_KEY=your_key)")
sys.path.insert(0, ROOT)
import schema; from schema import conn
from agent.agent import all_tasks, run_agent
from agent.faults import FAULTS, make_fault
random.seed(a.seed); c = conn(); tasks = all_tasks()
for t in ("runs", "steps", "labels"): c.execute(f"DELETE FROM {t}")   # keep the CACHE: reruns reuse paid-for calls
c.commit()
print(f"DB: {schema.DB_PATH}\nmode: {'MOCK dry-run' if a.mock else 'REAL Gemini'}\n")
natural_fail, api_err_streak = 0, 0
for i in range(a.n):
    t = random.choice(tasks)
    f = make_fault(random.choice(list(FAULTS))) if a.faults and i % 2 else None
    rid, out = run_agent(t, fault=f, c=c)
    errs = c.execute("SELECT step_idx, error FROM steps WHERE run_id=? AND error IS NOT NULL", (rid,)).fetchall()
    tag = f"fault={f['type']}" if f else "clean"
    print(f"[{i+1:>3}] {out.upper():4} {tag:32} {t['task_type']}")
    if errs:
        print("      API/step ERROR:", errs[0][1][:300])
        api_err_streak += 1
        if api_err_streak >= 3:
            sys.exit("\nStopping: 3 runs in a row hit API errors, so nothing here is a real result.\n"
                     "Fix the error above (wrong model name? try --model <name>; bad key? rate limit? try --sleep 3).")
        continue                      # an API error is NOT a natural failure
    api_err_streak = 0
    if out == "fail" and not f:
        natural_fail += 1
        print("      NATURAL FAILURE. step outputs:")
        for s in c.execute("SELECT step_idx, step_type, output FROM steps WHERE run_id=? ORDER BY step_idx", (rid,)):
            print(f"        {s[0]} {s[1]:9} {s[2][:110]}")
    time.sleep(0 if a.mock else a.sleep)
print("\nsummary:", c.execute("SELECT outcome, COUNT(*) FROM runs GROUP BY outcome").fetchall(),
      "| natural failures:", natural_fail,
      "| step errors:", c.execute("SELECT COUNT(*) FROM steps WHERE error IS NOT NULL").fetchone()[0])
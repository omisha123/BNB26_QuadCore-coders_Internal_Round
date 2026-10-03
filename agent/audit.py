"""Dataset health check. Run after generate.py:  python agent/audit.py
Prints the problems that make a diagnosis model look better than it is."""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import schema
c = schema.conn()
sp = json.load(open(os.path.join(os.path.dirname(schema.DB_PATH), "split.json")))
runs = {r[0]: r for r in c.execute("SELECT run_id, task_id, task_type, outcome FROM runs")}
lab = {r[0]: (r[1], r[2]) for r in c.execute(
    "SELECT run_id, faulty_step_idx, fault_type FROM labels WHERE injected=1")}
def flag(ok, msg): print(("  OK   " if ok else "  WARN ") + msg)

print(f"runs: {len(runs)}   injected: {len(lab)}   steps per run: "
      f"{c.execute('SELECT MIN(n), MAX(n) FROM (SELECT COUNT(*) n FROM steps GROUP BY run_id)').fetchone()}")
print("\n[1] task variety")
clean = Counter(r[1] for r in runs.values() if r[0] not in lab)
print(f"  distinct tasks: {len({r[1] for r in runs.values()})}   clean runs: {sum(clean.values())}   "
      f"max clean runs of one task: {max(clean.values())}")
flag(len({r[1] for r in runs.values()}) >= 300, "at least 300 distinct tasks")

print("\n[2] leakage between splits")
tid = lambda ids: {runs[i][1] for i in ids}
overlap = tid(sp["train"]) & tid(sp["test_seen"])
flag(not overlap, f"tasks shared by train and test_seen: {len(overlap)} (want 0)")
flag(not (tid(sp["train"]) & tid(sp["test_unseen_task"])), "held-out task type never appears in train")
train_faults = {lab[i][1] for i in sp["train"] if i in lab}
flag(not (train_faults & set(sp["holdout_fault_types"])), "held-out fault types never appear in train")

print("\n[3] where do culprits sit?")
steps = Counter(v[0] for v in lab.values()); n = sum(steps.values())
print("  culprit step:", {k: f"{100 * v / n:.0f}%" for k, v in sorted(steps.items())})
flag(max(steps.values()) / n < 0.30, f"always guessing the most common step scores {100 * max(steps.values()) / n:.0f}% (want < 30%)")
flag(steps.get(4, 0) / n > 0.10, f"'blame last step' baseline scores {100 * steps.get(4, 0) / n:.0f}% (want > 10%, so the baseline is meaningful)")

print("\n[4] giveaway markers (error words a model could just search for)")
marked = Counter()
for rid, (idx, ft) in lab.items():
    if any(("NOT_FOUND" in o or "UNKNOWN" in o) for (o,) in c.execute("SELECT output FROM steps WHERE run_id=?", (rid,))):
        marked[ft] += 1
flag(sum(marked.values()) == 0, f"injected runs containing NOT_FOUND/UNKNOWN: {sum(marked.values())} {dict(marked) or ''}")

print("\n[5] faults per type and split")
for ft, k in sorted(Counter(v[1] for v in lab.values()).items(), key=lambda x: -x[1]):
    held = " (held out)" if ft in sp["holdout_fault_types"] else ""
    print(f"  {ft:26} {k:4}{held}")
print("  splits:", {k: len(sp[k]) for k in ("train", "test_seen", "test_unseen_fault", "test_unseen_task")})
print("\n[6] known limits")
print("  - latency_ms is ~0 in mock mode (no timing signal); no token counts / logprobs recorded")
print("  - held-out fault types share step positions with seen faults, so 'unseen fault' is easier than a new step position")
print("  - every run has exactly 5 steps")
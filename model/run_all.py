"""python -m bb_model.run_all --db blackbox.db --split split.json"""
import argparse, json, time
from .loader import load_from_sqlite, load_split
from .model import BlackBoxModel
from .evaluate import evaluate_splits, learning_curve, ablation

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True); ap.add_argument("--split", required=True)
ap.add_argument("--fast", action="store_true", help="skip learning curve + ablation")
a = ap.parse_args()
runs, steps = load_from_sqlite(a.db); split = load_split(a.split)
print(f"{len(runs)} runs ({(runs.success==0).sum()} failed), {len(steps)} steps | " + ", ".join(f"{k}={len(v)}" for k, v in split.items()))
tr = runs[runs.run_id.isin(split["train"])]
t = time.time(); model = BlackBoxModel().fit(steps[steps.run_id.isin(tr.run_id)], tr); print(f"trained in {time.time()-t:.1f}s")
metrics = {"splits": evaluate_splits(model, steps, runs, split)}
if not a.fast:
    metrics["learning_curve"] = learning_curve(steps, runs, split)
    metrics["ablation_top1"] = ablation(steps, runs, split)
json.dump(metrics, open("metrics.json", "w"), indent=2)
model.save("blackbox_model.joblib")     # trained on TRAIN split only, so saved model == reported metrics
print(json.dumps(metrics, indent=2))

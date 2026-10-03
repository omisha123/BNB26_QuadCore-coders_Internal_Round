"""Evaluate the learned diagnosis against KNOWN injected faults, using Person 1's split.json.

  test_seen          new tasks, fault types seen in training
  test_unseen_fault  fault types NEVER seen in training      (does it generalise to new kinds of bugs?)
  test_unseen_task   a task type NEVER seen in training
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .model import BlackBoxModel


def _rank_of_truth(g, true_step):
    order = g.sort_values("score", ascending=False, kind="stable").step_idx.tolist()
    return order.index(true_step) + 1 if true_step in order else len(order) + 1


def baselines(f, failed, train_failed):
    """Dumb strategies the learned model must beat."""
    rng = np.random.default_rng(0)
    prior = int(train_failed.fault_step.mode().iloc[0])  # 'always blame the step that failed most often in training'
    t = failed.fault_step.values
    rnd = [int(rng.choice(f[f.run_id == r].step_idx)) for r in failed.run_id]
    last = [int(f[f.run_id == r].step_idx.max()) for r in failed.run_id]
    return {"random_step": float(np.mean(np.array(rnd) == t)), "always_last_step": float(np.mean(np.array(last) == t)),
            f"always_step_{prior}_(training_majority)": float(np.mean(t == prior))}


def evaluate(model: BlackBoxModel, steps, runs, train_runs=None):
    f = model.score_steps(steps)
    failed = runs[runs.success == 0]
    ranks, pred_steps = [], []
    for r in failed.itertuples():
        g = f[f.run_id == r.run_id]
        ranks.append(_rank_of_truth(g, r.fault_step)); pred_steps.append(int(g.loc[g.score.idxmax(), "step_idx"]))
    ranks, pred = np.array(ranks), np.array(pred_steps)
    true = failed.fault_step.values
    out = dict(n_failed_runs=int(len(failed)), top1=float((ranks == 1).mean()), top3=float((ranks <= 3).mean()),
               mrr=float((1 / ranks).mean()), mean_step_error=float(np.abs(pred - true).mean()),
               # replaying from the predicted step still re-runs the true cause when predicted <= true
               replay_covers_cause=float((pred <= true).mean()))
    per = pd.Series(ranks == 1, index=failed.index).groupby(failed.fault_type.values).agg(["mean", "size"])
    out["per_fault_type_top1"] = {k: dict(top1=round(float(v["mean"]), 3), n=int(v["size"])) for k, v in per.iterrows()}
    if "task_type" in failed:
        per = pd.Series(ranks == 1, index=failed.index).groupby(failed.task_type.values).mean()
        out["per_task_type_top1"] = {k: round(float(v), 3) for k, v in per.items()}
    if runs.success.nunique() > 1:
        sc = f.groupby("run_id").score.max().reindex(runs.run_id)
        out["fail_detection_auc"] = float(roc_auc_score(1 - runs.success.values, sc.values))
    if train_runs is not None:
        out["baselines_top1"] = baselines(f, failed, train_runs[train_runs.success == 0])
    return out


def evaluate_splits(model, steps, runs, split):
    res = {}
    tr = runs[runs.run_id.isin(split["train"])]
    for name in ("test_seen", "test_unseen_fault", "test_unseen_task"):
        r = runs[runs.run_id.isin(split[name])]
        res[name] = evaluate(model, steps[steps.run_id.isin(r.run_id)], r, tr)
    return res


def learning_curve(steps, runs, split, sizes=(100, 200, 400, 800, 1563), seed=0, **kw):
    tr_all = runs[runs.run_id.isin(split["train"])]
    rows = []
    for n in sizes:
        sub = tr_all.sample(min(n, len(tr_all)), random_state=seed)
        if (sub.success == 0).sum() < 10: continue
        m = BlackBoxModel(**kw).fit(steps[steps.run_id.isin(sub.run_id)], sub)
        row = dict(train_runs=int(len(sub)))
        for name in ("test_seen", "test_unseen_fault"):
            r = runs[runs.run_id.isin(split[name])]
            row[name + "_top1"] = round(evaluate(m, steps[steps.run_id.isin(r.run_id)], r)["top1"], 3)
        rows.append(row)
    return rows


def ablation(steps, runs, split):
    """Which signal groups matter?  (top-1 on each test split)"""
    anom = ("anom", "anom_sum", "others_max_anom", "n_anom_run", "anom_before", "anom_after",
            "anom_rank_in_run", "anom_gap", "prev_anom", "next_anom")
    cfgs = {"full model (default)": dict(),
            "+ cross-run memory": dict(use_memory=True),
            "no 'deviation from normal' / elimination feats": dict(drop_cols=anom),
            "no position info": dict(drop_cols=("pos_rel", "is_last", "type_code")),
            "no neighbour (prev/next) context": dict(drop_cols=tuple(
                f"{p}{c}" for p in ("prev_", "next_", "next2_") for c in
                ("error", "err_in_text", "out_empty", "jac_in_out", "num_consistent_task", "weird_chars_in", "state_changed", "out_in_state", "anom"))),
            }
    tr = runs[runs.run_id.isin(split["train"])]; res = {}
    for name, kw in cfgs.items():
        m = BlackBoxModel(**kw).fit(steps[steps.run_id.isin(tr.run_id)], tr)
        e = evaluate_splits(m, steps, runs, split)
        res[name] = {k: round(v["top1"], 3) for k, v in e.items()}
    return res

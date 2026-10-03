"""Turn raw trace rows into one feature vector per step.
Only OBSERVABLE signals are used (inputs/outputs/latency/error flags) -- never fault labels."""
import re
import json
import numpy as np
import pandas as pd

NUM = re.compile(r"\d+(?:\.\d+)?")
TOK = re.compile(r"[a-z0-9]+")


def _toks(s): return set(TOK.findall(str(s).lower()))
def _nums(s): return set(NUM.findall(str(s)))
def _vals(s):
    """Tokens of the VALUES inside a JSON string (ignore keys like 'company'); falls back to raw text."""
    try:
        o = json.loads(s)
    except Exception:
        return _toks(s)
    out = []
    def walk(x):
        if isinstance(x, dict): [walk(v) for v in x.values()]
        elif isinstance(x, (list, tuple)): [walk(v) for v in x]
        else: out.append(str(x))
    walk(o)
    return _toks(" ".join(out))


def _leaves(s):
    """Raw (case-sensitive) leaf values of a JSON string, as strings."""
    try:
        o = json.loads(s)
    except Exception:
        return [str(s)] if str(s) else []
    out = []
    def walk(x):
        if isinstance(x, dict): [walk(v) for v in x.values()]
        elif isinstance(x, (list, tuple)): [walk(v) for v in x]
        else: out.append(str(x))
    walk(o)
    return out


def _jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


def build_features(steps: pd.DataFrame, embedder=None, step_types=None) -> pd.DataFrame:
    """embedder: optional callable list[str] -> np.ndarray (e.g. sentence-transformers .encode).
    If given, adds cosine(input, output) and cosine(step output, previous output)."""
    steps = steps.sort_values(["run_id", "step_idx"]).reset_index(drop=True).copy()
    steps["input_text"] = steps["input_text"].fillna(""); steps["output_text"] = steps["output_text"].fillna("")
    if "tool_name" not in steps: steps["tool_name"] = ""
    steps["tool_name"] = steps["tool_name"].fillna("")
    steps["step_type"] = steps["step_type"].fillna("").astype(str)
    for c in ("state_before", "state_after"):
        steps[c] = steps[c].fillna("").astype(str) if c in steps else ""
    if "input_hash" not in steps: steps["input_hash"] = steps.input_text.map(hash).astype(str)
    step_types = step_types if step_types is not None else sorted(steps.step_type.unique())
    f = pd.DataFrame({"run_id": steps.run_id, "step_idx": steps.step_idx})
    for t in step_types: f[f"is_{re.sub(r'[^A-Za-z0-9]+', '_', t)}"] = (steps.step_type == t).astype(int)
    f["tool_code"] = steps.tool_name.map({"": 0, "lookup": 1, "calculator": 2, "search": 3}).fillna(4)
    f["type_code"] = steps.step_type.map({t: i for i, t in enumerate(step_types)}).fillna(-1)
    f["in_len"] = steps.input_text.str.len(); f["out_len"] = steps.output_text.str.len()
    f["out_empty"] = ((steps.output_text.str.strip() == "") & (steps.step_type == "tool_result")).astype(int)
    f["error"] = steps.error.astype(int)
    f["err_in_text"] = steps.output_text.str.contains("error|invalid|timeout|irrelevant|unknown", case=False).astype(int)
    f["latency"] = steps.latency_ms
    f["n_nums_in"] = steps.input_text.map(lambda s: len(_nums(s)))
    f["n_nums_out"] = steps.output_text.map(lambda s: len(_nums(s)))
    f["weird_chars_in"] = steps.input_text.str.count(r"[^A-Za-z0-9 =+\-_.:|,]")  # malformed args

    tin = steps.input_text.map(_toks); tout = steps.output_text.map(_toks)
    nin = steps.input_text.map(_nums); nout = steps.output_text.map(_nums)
    qsrc = steps["question"] if "question" in steps else steps.groupby("run_id").input_text.transform("first")
    qnums = qsrc.fillna("").map(_nums)
    g = steps.groupby("run_id")
    f["n_steps"] = g.step_idx.transform("count"); f["pos_rel"] = steps.step_idx / f.n_steps
    f["is_last"] = (steps.step_idx == f.n_steps - 1).astype(int)

    # relations between neighbouring steps (where inconsistencies show up)
    prev_out_t = tout.groupby(steps.run_id).shift(1); prev_out_n = nout.groupby(steps.run_id).shift(1)
    f["jac_in_out"] = [_jac(a, b) for a, b in zip(tin, tout)]
    f["jac_prev_out"] = [_jac(a, b) if isinstance(b, set) else 1.0 for a, b in zip(tout, prev_out_t)]
    f["num_overlap_in_out"] = [len(a & b) for a, b in zip(nin, nout)]
    # does this step's output reuse numbers from the original task statement (first step input)?
    f["num_consistent_task"] = [len(a & b) / max(len(a), 1) for a, b in zip(nout, qnums)]
    # repeated identical call (retry)
    key = steps.tool_name + "|" + steps.input_text
    f["is_retry"] = (steps.step_type == "tool_call").astype(int) * (
        key.groupby([steps.run_id, steps.step_type]).cumcount() > 0).astype(int)

    # latency relative to same tool across the whole dataset
    f["lat_z"] = (f.latency - f.groupby("type_code").latency.transform("mean")) / (
        f.groupby("type_code").latency.transform("std").fillna(1) + 1e-6)

    # --- state-diff features (Person 1's state_before / state_after) ---
    f["state_changed"] = (steps.state_before != steps.state_after).astype(int)
    f["state_len_delta"] = steps.state_after.str.len() - steps.state_before.str.len()
    f["state_tokens_added"] = [len(_toks(a) - _toks(b)) for a, b in zip(steps.state_after, steps.state_before)]
    f["state_tokens_removed"] = [len(_toks(b) - _toks(a)) for a, b in zip(steps.state_after, steps.state_before)]
    f["out_in_state"] = [(_jac(o, _toks(a)) if o else 1.0) for o, a in zip(tout, steps.state_after)]  # did output land in state?
    f["state_num_consistent"] = [len(_nums(a) & q) / max(len(q), 1) for a, q in zip(steps.state_after, qnums)]
    # --- value-level consistency: does the output introduce things its input never mentioned,
    #     and does it keep what its input asked about? ---
    vin = steps.input_text.map(_vals); vout = steps.output_text.map(_vals); qt = qsrc.fillna("").map(_toks)
    f["out_novel_frac"] = [len(o - (i | q)) / max(len(o), 1) for o, i, q in zip(vout, vin, qt)]
    f["in_missing_frac"] = [len(i - o) / max(len(i), 1) for o, i in zip(vout, vin)]
    f["n_out_vals"] = vout.map(len)
    # copy fidelity: are the output's raw values found VERBATIM (case, commas, whitespace) in the input?
    f["exact_copy_frac"] = [(sum(v in i_txt for v in lv) / len(lv)) if lv else 1.0
                            for lv, i_txt in zip(steps.output_text.map(_leaves), steps.input_text)]
    # repeated identical input within a run (loops / retries)
    f["dup_hash_in_run"] = steps.groupby(["run_id", "input_hash"]).cumcount()

    # running context: errors seen so far in this run
    f["errs_before"] = f.groupby("run_id").error.cumsum() - f.error

    # lag / lead context (root causes show their effects DOWNSTREAM)
    for col in ["error", "err_in_text", "out_empty", "jac_in_out", "num_consistent_task", "weird_chars_in", "state_changed", "out_in_state"]:
        grp = f.groupby("run_id")[col]
        f[f"prev_{col}"] = grp.shift(1).fillna(0); f[f"next_{col}"] = grp.shift(-1).fillna(0)
        f[f"next2_{col}"] = grp.shift(-2).fillna(0)
    f["final_err_in_text"] = f.groupby("run_id").err_in_text.transform("last")
    f["final_num_consistent"] = f.groupby("run_id").num_consistent_task.transform("last")

    if embedder is not None:  # semantic consistency features
        ei = embedder(steps.input_text.tolist()); eo = embedder(steps.output_text.tolist())
        ei = ei / (np.linalg.norm(ei, axis=1, keepdims=True) + 1e-9); eo = eo / (np.linalg.norm(eo, axis=1, keepdims=True) + 1e-9)
        f["emb_cos_in_out"] = (ei * eo).sum(1)
        prev = pd.DataFrame(eo).groupby(steps.run_id).shift(1).fillna(0).values
        f["emb_cos_prev_out"] = (eo * prev).sum(1)
    return f


FEATURE_COLS = lambda f: [c for c in f.columns if c not in ("run_id", "step_idx")]

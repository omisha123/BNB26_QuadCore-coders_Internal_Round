"""Step-level root-cause model. Label = 1 only for the injected-fault step of a failed run."""
import joblib
from collections import Counter
import numpy as np
import pandas as pd
import lightgbm as lgb
from .features import build_features, FEATURE_COLS

EXPLAIN = {  # feature -> human sentence, used when a feature pushes the score up
    "out_empty": "the tool returned an empty result",
    "error": "the step reported an error",
    "err_in_text": "the output text looks like an error / irrelevant / unknown",
    "next_error": "the next step failed right after it",
    "next_err_in_text": "the next step's output looks like an error",
    "next2_err_in_text": "an error shows up two steps later",
    "next_out_empty": "the next step got an empty result",
    "weird_chars_in": "the input arguments look malformed",
    "num_consistent_task": "numbers in the output disagree with the task statement",
    "next_num_consistent_task": "the following step's numbers disagree with the task",
    "final_num_consistent": "the final answer is inconsistent with the task numbers",
    "final_err_in_text": "the final answer is an error/unknown",
    "jac_in_out": "the output shares little with its own input",
    "jac_prev_out": "the output doesn't follow from the previous step",
    "next_jac_in_out": "the next step ignores this step's output",
    "lat_z": "unusual latency for this tool",
    "tool_code": "unexpected tool for this point in the plan",
    "is_last": "it is the final answer step",
    "is_tool_call": "it is a tool call", "is_tool_result": "it is a tool result",
    "errs_before": "errors had already occurred earlier in the run",
    "latency": "this step was unusually slow", "pos_rel": "its position in the run is typical of failures",
    "n_nums_out": "the number of values in its output is abnormal", "out_len": "the output length is abnormal",
    "state_changed": "the shared state changed at this step",
    "next_state_changed": "the state changed unexpectedly right after it",
    "state_num_consistent": "the state holds numbers that disagree with the question",
    "out_in_state": "its output never made it into the agent state",
    "next_out_in_state": "the next step's output didn't make it into the state",
    "state_tokens_added": "it wrote unusual content into the state",
    "dup_hash_in_run": "it repeated an identical input seen earlier (loop/retry)",
    "out_novel_frac": "its output contains things that appear nowhere in its input or the question",
    "in_missing_frac": "its output ignores what its input asked about",
    "exact_copy_frac": "it altered a value it should have copied exactly (case / formatting / rounding)",
    "anom": "this step looks abnormal compared with the same step in passing runs",
    "anom_gap": "it is far more abnormal than any other step in the run",
    "others_max_anom": "every other step looks normal, so the fault must be here",
    "anom_before": "something abnormal already happened upstream",
    "mem_agree": "past runs with the same input produced a different output",
    "emb_cos_in_out": "output is semantically unrelated to the input",
    "emb_cos_prev_out": "output is semantically unrelated to the previous step",
}


class BlackBoxModel:
    def __init__(self, embedder=None, use_memory=False, drop_cols=(), **params):
        self.embedder, self.use_memory, self.drop_cols = embedder, use_memory, tuple(drop_cols)
        self.baseline, self.mem, self.mem_task = None, None, None
        self.params = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           scale_pos_weight=5, random_state=0, verbose=-1)
        self.params.update(params)
        self.clf, self.cols, self.step_types = None, None, None

    def _labels(self, f, runs):
        fs = runs.set_index("run_id").fault_step
        return (f.step_idx.values == f.run_id.map(fs).values).astype(int)

    DEV_BASE = ["out_novel_frac", "in_missing_frac", "jac_in_out", "jac_prev_out", "state_tokens_added",
                "state_tokens_removed", "out_len", "in_len", "state_len_delta", "n_out_vals", "exact_copy_frac"]

    def _fit_baseline(self, f, success_by_run):
        """What does each step TYPE look like in PASSING runs? Later steps are scored by distance from that."""
        ok = f[f.run_id.map(success_by_run).fillna(0).astype(bool)]
        g = ok.groupby("type_code")[self.DEV_BASE]
        self.baseline = (g.mean(), g.std().fillna(0))

    def _add_dev(self, f):
        mu, sd = self.baseline
        dev = pd.DataFrame(index=f.index)
        for c in self.DEV_BASE:
            m = f.type_code.map(mu[c]).fillna(0); d = f.type_code.map(sd[c]).fillna(0)
            dev[c] = ((f[c] - m).abs() / (d + 0.1)).clip(0, 20)
        f["anom"] = dev.max(axis=1)
        f["anom_sum"] = dev.sum(axis=1)
        g = f.groupby("run_id").anom
        f["others_max_anom"] = [max([a for j, a in enumerate(v) if j != i] or [0])
                                for _, v in g for i in range(len(v))]
        f["n_anom_run"] = g.transform(lambda x: (x > 3).sum())
        f["anom_before"] = g.transform(lambda x: x.shift(1).cummax()).fillna(0)
        f["anom_after"] = g.transform(lambda x: x[::-1].shift(1).cummax()[::-1]).fillna(0)
        f["anom_rank_in_run"] = g.rank(ascending=False)
        f["anom_gap"] = f.anom - f.others_max_anom
        f["prev_anom"] = g.shift(1).fillna(0); f["next_anom"] = g.shift(-1).fillna(0)
        return f

    def _build_memory(self, s):
        self.mem, self.mem_task = {}, {}
        for r in s.itertuples():
            k = (r.step_type, r.input_hash)
            self.mem.setdefault(k, Counter())[r.output_text] += 1
            self.mem_task.setdefault((k, r.task_id), Counter())[r.output_text] += 1

    def _add_memory(self, f, s, leave_task_out):
        """Do other (past) runs agree with this step's output for the same input? Leave-task-out in training
        so train rows look like test rows (unseen task)."""
        n, agree = [], []
        for r in s.itertuples():
            k = (r.step_type, r.input_hash); c = Counter(self.mem.get(k, {}))
            if leave_task_out: c.subtract(self.mem_task.get((k, r.task_id), {}))
            tot = sum(v for v in c.values() if v > 0)
            n.append(tot); agree.append(c.get(r.output_text, 0) / tot if tot > 0 else np.nan)
        f["mem_n"], f["mem_agree"] = n, agree
        return f

    def _feat(self, steps, leave_task_out=False):
        s = steps.sort_values(["run_id", "step_idx"]).reset_index(drop=True)
        if "task_id" not in s: s["task_id"] = s.run_id
        s["output_text"] = s.output_text.fillna(""); s["step_type"] = s.step_type.fillna("").astype(str)
        if "input_hash" not in s: s["input_hash"] = s.input_text.map(hash).astype(str)
        f = build_features(s, self.embedder, self.step_types)
        if self.baseline is not None: f = self._add_dev(f)
        if self.use_memory and self.mem is not None: f = self._add_memory(f, s, leave_task_out)
        return f.drop(columns=[c for c in self.drop_cols if c in f])

    def fit(self, steps, runs):
        self.step_types = sorted(steps.step_type.fillna("").astype(str).unique())
        raw = build_features(steps.sort_values(["run_id", "step_idx"]).reset_index(drop=True), self.embedder, self.step_types)
        self._fit_baseline(raw, runs.set_index("run_id").success)
        if self.use_memory:
            s = steps.sort_values(["run_id", "step_idx"]).reset_index(drop=True)
            if "task_id" not in s: s["task_id"] = s.run_id
            s["output_text"] = s.output_text.fillna(""); self._build_memory(s)
        f = self._feat(steps, leave_task_out=True); self.cols = FEATURE_COLS(f)
        y = self._labels(f, runs)
        self.clf = lgb.LGBMClassifier(**self.params).fit(f[self.cols], y)
        return self

    def score_steps(self, steps):
        """Return per-step 'suspicion' = P(this step is the root cause)."""
        f = self._feat(steps)
        for c in self.cols:
            if c not in f: f[c] = 0  # step type unseen in training
        f["score"] = self.clf.predict_proba(f[self.cols])[:, 1]
        return f

    def diagnose(self, steps, top_k=3):
        """steps: DataFrame of ONE run that is KNOWN to have failed. Returns ranked suspects with reasons."""
        f = self.score_steps(steps)
        contrib = self.clf.predict(f[self.cols], pred_contrib=True)[:, :-1]
        order = np.argsort(-f.score.values)[:top_k]
        total = f.score.sum() + 1e-9
        out = []
        for rank, i in enumerate(order):
            conf = float(f.score.iloc[i] / total)
            if conf < 0.05:
                reasons = ["little evidence against this step"]
            elif f.anom.iloc[i] < 3 and f.others_max_anom.iloc[i] < 3 if "anom" in f else False:
                reasons = ["no step in the run shows a visible problem, and this step's output cannot be checked "
                           "from the trace alone, so it is the most likely cause (confirm by replaying it)"]
            else:
                top = np.argsort(-contrib[i])[:8]
                reasons = [EXPLAIN[self.cols[j]] for j in top if contrib[i, j] > 0.05 and self.cols[j] in EXPLAIN][:3]
                reasons = reasons or ["pattern similar to past failures"]
            out.append(dict(step_idx=int(f.step_idx.iloc[i]), score=float(f.score.iloc[i]), confidence=conf, reasons=reasons))
        return out

    def run_failure_score(self, steps):
        f = self.score_steps(steps)
        return f.groupby("run_id").score.max()

    def save(self, path): joblib.dump({"clf": self.clf, "cols": self.cols, "params": self.params, "step_types": self.step_types,
                     "baseline": self.baseline, "mem": self.mem, "mem_task": self.mem_task,
                     "use_memory": self.use_memory, "drop_cols": self.drop_cols}, path)
    @classmethod
    def load(cls, path, embedder=None):
        d = joblib.load(path); m = cls(embedder); m.clf, m.cols, m.params, m.step_types = d["clf"], d["cols"], d["params"], d["step_types"]
        m.baseline, m.mem, m.mem_task = d["baseline"], d["mem"], d["mem_task"]
        m.use_memory, m.drop_cols = d["use_memory"], d["drop_cols"]; return m

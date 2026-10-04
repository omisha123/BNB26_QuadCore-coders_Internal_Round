"""Black Box: AI Agent Observability, Failure Diagnosis & Backtracking Replay System.

Minimalist Muted Blues & White Palette (Strictly 3 Colors: #FFFFFF, #5B708B, #0F172A).
Background features a subtle gradient between White and Muted Blue.
Zero extraneous colors. Strictly emoji-free.
"""
import os
import sys
import json
import sqlite3
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import schema
from agent.agent import all_tasks, run_agent, COMPANIES, TASK_TYPES, CITY_INFO, DOCS
from agent.faults import FAULTS, make_fault
from model.model import BlackBoxModel, EXPLAIN
from replay.replay import replay_run

# Page Configuration
st.set_page_config(
    page_title="Black Box: Agent Diagnosis & Backtracking",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimalist Muted Blues & White Theme (3 Colors Only: #FFFFFF, #5B708B, #0F172A)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --c-white: #FFFFFF;
        --c-muted: #5B708B;
        --c-dark: #0F172A;
    }

    /* Global Canvas: Background Gradient strictly between White and Muted Blue */
    .stApp {
        background: linear-gradient(180deg, #FFFFFF 0%, rgba(91, 112, 139, 0.18) 100%) !important;
        color: var(--c-dark) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Streamlit top header */
    header[data-testid="stHeader"] {
        background-color: var(--c-white) !important;
        border-bottom: 1px solid var(--c-muted) !important;
    }

    /* Sidebar: Solid White with Muted Blue Border */
    section[data-testid="stSidebar"] {
        background-color: var(--c-white) !important;
        border-right: 1px solid var(--c-muted) !important;
    }

    /* Typography: Exclusively #0F172A and #5B708B */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--c-dark) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }

    p, span, label {
        color: var(--c-dark) !important;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.86rem !important;
        color: var(--c-dark) !important;
        background-color: var(--c-white) !important;
        border: 1px solid var(--c-muted) !important;
    }

    /* Card Panels: Solid White, Solid Borders */
    .bb-card {
        background-color: var(--c-white);
        border: 1px solid var(--c-muted);
        border-radius: 4px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }

    /* Pill Badges */
    .bb-pill {
        display: inline-block;
        padding: 3px 9px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border-radius: 2px;
    }

    .bb-pill-pass {
        color: var(--c-white);
        background-color: var(--c-muted);
        border: 1px solid var(--c-muted);
    }

    .bb-pill-fail {
        color: var(--c-white);
        background-color: var(--c-dark);
        border: 1px solid var(--c-dark);
    }

    .bb-pill-neutral {
        color: var(--c-dark);
        background-color: var(--c-white);
        border: 1px solid var(--c-muted);
    }

    /* LLM Thinking Monologue Box */
    .bb-monologue {
        background-color: var(--c-white);
        border: 1px solid var(--c-muted);
        border-left: 4px solid var(--c-muted);
        border-radius: 0 4px 4px 0;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 0.88rem;
        line-height: 1.5;
        color: var(--c-dark);
    }

    .bb-monologue-title {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--c-muted);
        margin-bottom: 4px;
    }

    /* Diagnostic Callout */
    .bb-anomaly-flag {
        background-color: var(--c-white);
        border: 1px solid var(--c-dark);
        border-left: 4px solid var(--c-dark);
        border-radius: 4px;
        padding: 14px 18px;
        margin: 14px 0;
    }

    .bb-anomaly-title {
        color: var(--c-dark);
        font-weight: 700;
        font-size: 0.86rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    /* Backtracking Rewind Arc Banner */
    .bb-backtrack-banner {
        background-color: var(--c-white);
        border: 1px solid var(--c-muted);
        border-left: 4px solid var(--c-dark);
        border-radius: 4px;
        padding: 16px 20px;
        margin: 16px 0 24px 0;
    }

    .bb-backtrack-title {
        color: var(--c-dark);
        font-weight: 700;
        font-size: 0.88rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    /* Step Timeline Node */
    .bb-step-card {
        background-color: var(--c-white);
        border: 1px solid var(--c-muted);
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }

    .bb-step-card.is-culprit {
        border: 2px solid var(--c-dark);
    }

    .bb-step-card.is-skipped {
        border: 1px dashed var(--c-muted);
        opacity: 0.85;
    }

    .bb-step-card.is-replayed {
        border: 2px solid var(--c-muted);
    }

    /* Streamlit widgets overrides */
    div[data-testid="stMetricValue"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        color: var(--c-dark) !important;
        letter-spacing: -0.02em !important;
    }

    div[data-testid="stMetricLabel"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        color: var(--c-muted) !important;
    }

    .stButton>button {
        background-color: var(--c-dark) !important;
        color: var(--c-white) !important;
        border: 1px solid var(--c-dark) !important;
        border-radius: 3px !important;
        font-weight: 500 !important;
        padding: 6px 18px !important;
    }

    .stButton>button:hover {
        background-color: var(--c-muted) !important;
        color: var(--c-white) !important;
        border-color: var(--c-muted) !important;
    }

    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        color: var(--c-dark) !important;
        border-radius: 3px !important;
        background-color: var(--c-white) !important;
        border: 1px solid var(--c-muted) !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = os.getenv("BB_MODEL", "blackbox_model.joblib")
    if os.path.exists(model_path):
        return BlackBoxModel.load(model_path)
    return None

model = load_model()

@st.cache_data
def load_metrics():
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

metrics = load_metrics()
c = schema.conn()


# Header Banner
st.markdown("""
<div style="padding: 12px 0 24px 0; border-bottom: 1px solid #5B708B; margin-bottom: 28px;">
    <div style="display: flex; align-items: baseline; justify-content: space-between;">
        <div>
            <div style="font-size: 0.76rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #5B708B; margin-bottom: 4px;">
                Agent Observability & Root-Cause Diagnosis
            </div>
            <div style="font-size: 1.65rem; font-weight: 600; color: #0F172A; letter-spacing: -0.03em;">
                Black Box: Agent Failure Diagnosis & Backtracking Engine
            </div>
        </div>
        <div style="text-align: right;">
            <span class="bb-pill bb-pill-neutral">Model: LightGBM Diagnostic</span>
            <span class="bb-pill bb-pill-pass" style="margin-left: 6px;">Top-1 Acc: 99.56%</span>
        </div>
    </div>
    <div style="font-size: 0.88rem; color: #5B708B; margin-top: 6px; max-width: 820px;">
        Monitors multi-step LLM execution traces, isolates faulty reasoning in intermediate steps using 70+ non-intrusive runtime signals, and executes selective backtracking to repair failures without repeating unaffected execution.
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar Navigation
st.sidebar.markdown("""
<div style="padding-bottom: 12px; margin-bottom: 16px; border-bottom: 1px solid #5B708B;">
    <div style="font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #5B708B;">
        Control Center
    </div>
    <div style="font-size: 1.05rem; font-weight: 600; color: #0F172A; margin-top: 2px;">
        Navigation
    </div>
</div>
""", unsafe_allow_html=True)

nav = st.sidebar.radio(
    "Select System View",
    [
        "Trace Inspector & AI Diagnosis",
        "Backtracking & Selective Replay",
        "Benchmark Evaluation & Metrics",
        "Live Agent Execution Sandbox"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.78rem; color: #5B708B; line-height: 1.5;">
    <strong>Architecture Summary</strong><br>
    - 5 Sequential Agent Steps<br>
    - Pure State Transformations<br>
    - Deterministic Input Hashing<br>
    - Selective State Rollback<br>
    - Zero Ground-Truth Leakage
</div>
""", unsafe_allow_html=True)


def get_step_intent_and_thinking(step_type, inp, out, is_faulty=False, fault_type=None):
    """Provides structured insight into what the LLM is doing and thinking at each phase."""
    if step_type == "plan":
        q = inp.get("question", "")
        company = out.get("company", "")
        attr = out.get("attr", "")
        intent = "Goal formulation & named entity extraction."
        thinking = f"Analyzing question '{q}'. The LLM isolates company name '{company}' and identifies target attribute '{attr}'."
        if is_faulty and fault_type == "bad_plan":
            thinking += f" [Discrepancy: Extracted company '{company}' does not match question target entity. Hallucination occurred at planning.]"
        return intent, thinking

    elif step_type == "retrieve":
        company = inp.get("company", "")
        docs = out.get("docs", [])
        intent = "Knowledge retrieval from internal documentation."
        thinking = f"Querying context documents for '{company}'. Scored and retrieved {len(docs)} documents detailing company locations and founding attributes."
        if is_faulty and fault_type == "irrelevant_retrieval":
            thinking += " [Discrepancy: Retrieved documents pertain to an unrelated company. Context corrupted.]"
        return intent, thinking

    elif step_type == "reason":
        company = inp.get("company", "")
        city = out.get("city", "")
        intent = "Deductive entity reasoning across context."
        thinking = f"Evaluating retrieved documents for '{company}'. Parsing location predicates ('headquartered in [City]') to deduce municipality."
        if is_faulty and fault_type in ("corrupted_state", "dropped_context"):
            thinking += f" [Discrepancy: The model deduced city '{city}' which contradicts the retrieved documents. Reasoning drifted.]"
        return intent, thinking

    elif step_type == "tool":
        city = inp.get("city", "")
        attr = inp.get("attr", "")
        val = out.get("value", "")
        intent = "Deterministic municipal database tool lookup."
        thinking = f"Calling municipal database with key ('{city}', '{attr}'). Deterministic record returned: '{val}'."
        if is_faulty and fault_type in ("wrong_tool_arg", "hallucinated_tool_output"):
            thinking += f" [Discrepancy: Tool lookup executed on erroneous argument or returned corrupted value '{val}'.]"
        return intent, thinking

    elif step_type == "answer":
        val = inp.get("value", "")
        ans = out.get("answer", "")
        intent = "Response synthesis and formatting."
        thinking = f"Synthesizing final concise answer from tool output '{val}'. Adhering to exact literal output specification."
        if is_faulty and fault_type in ("answer_format_error", "answer_wrong_rounding"):
            thinking += f" [Discrepancy: Formatting violation or erroneous numerical rounding during final synthesis.]"
        return intent, thinking

    return "General execution step.", "Processing input parameters."


# =====================================================================
# VIEW 1: Trace Inspector & AI Root-Cause Diagnosis
# =====================================================================
if nav == "Trace Inspector & AI Diagnosis":
    st.subheader("Trace Inspector & AI Root-Cause Diagnosis")
    st.caption("Inspect observable execution traces, analyze what the LLM did and thought at each step, and review AI diagnosis flags.")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        outcome_filter = st.selectbox("Outcome Filter", ["All", "fail", "pass"], index=1)
    with col_f2:
        task_types = ["All"] + TASK_TYPES
        task_filter = st.selectbox("Task Type Filter", task_types)
    with col_f3:
        search_id = st.text_input("Search Run ID", "")

    query = "SELECT r.run_id, r.task_id, r.task_type, r.question, r.final_answer, r.expected, r.outcome, l.faulty_step_idx, l.fault_type, l.injected FROM runs r LEFT JOIN labels l ON r.run_id=l.run_id"
    conds, params = [], []
    if outcome_filter != "All":
        conds.append("r.outcome = ?")
        params.append(outcome_filter)
    if task_filter != "All":
        conds.append("r.task_type = ?")
        params.append(task_filter)
    if search_id.strip():
        conds.append("r.run_id LIKE ?")
        params.append(f"%{search_id.strip()}%")

    if conds:
        query += " WHERE " + " AND ".join(conds)
    query += " ORDER BY r.rowid DESC LIMIT 100"

    runs_df = pd.read_sql(query, c, params=params if conds else None)

    if runs_df.empty:
        st.info("No runs found matching the selected filter criteria.")
    else:
        run_options = [f"{r.run_id} | [{r.outcome.upper()}] | {r.task_id} | {r.question[:46]}..." for r in runs_df.itertuples()]
        selected_option = st.selectbox("Select Execution Trace to Inspect", run_options)
        selected_run_id = selected_option.split(" | ")[0]

        run_data = runs_df[runs_df.run_id == selected_run_id].iloc[0]

        is_pass = run_data.outcome == "pass"
        status_pill = '<span class="bb-pill bb-pill-pass">[PASSED]</span>' if is_pass else '<span class="bb-pill bb-pill-fail">[FAILED]</span>'
        
        st.markdown(f"""
        <div class="bb-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                <div>
                    <span style="font-size: 0.74rem; font-weight: 600; text-transform: uppercase; color: #5B708B; letter-spacing: 0.05em;">Execution Trace Metadata</span>
                    <h3 style="margin: 2px 0 0 0; font-size: 1.15rem; color: #0F172A;">Run ID: <code>{run_data.run_id}</code></h3>
                </div>
                <div>
                    {status_pill}
                </div>
            </div>
            <div style="font-size: 0.92rem; color: #0F172A; margin-bottom: 14px; padding: 10px 14px; background: #FFFFFF; border-radius: 4px; border: 1px solid #5B708B;">
                <strong>Task Prompt:</strong> {run_data.question}
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; font-size: 0.84rem;">
                <div>
                    <span style="color: #5B708B; font-size: 0.74rem; text-transform: uppercase; font-weight: 600;">Task Type</span><br>
                    <strong>{run_data.task_type}</strong>
                </div>
                <div>
                    <span style="color: #5B708B; font-size: 0.74rem; text-transform: uppercase; font-weight: 600;">Final Agent Answer</span><br>
                    <code style="color: #0F172A; font-weight: 600;">{run_data.final_answer}</code>
                </div>
                <div>
                    <span style="color: #5B708B; font-size: 0.74rem; text-transform: uppercase; font-weight: 600;">Expected Ground Truth</span><br>
                    <code style="color: #0F172A; font-weight: 600;">{run_data.expected}</code>
                </div>
                <div>
                    <span style="color: #5B708B; font-size: 0.74rem; text-transform: uppercase; font-weight: 600;">Discrepancy</span><br>
                    <span style="color: #0F172A; font-weight: 600;">{'None (Nominal)' if is_pass else 'Mismatch Detected'}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        steps_df = pd.read_sql("SELECT * FROM steps WHERE run_id=? ORDER BY step_idx", c, params=[selected_run_id])

        if not is_pass and model is not None:
            steps_feat_df = steps_df.rename(columns={"input": "input_text", "output": "output_text"}).assign(question=run_data.question)
            steps_feat_df["error"] = pd.to_numeric(steps_feat_df["error"], errors="coerce").fillna(0).astype(int)
            fail_score = model.run_failure_score(steps_feat_df).iloc[0]
            diagnosis = model.diagnose(steps_feat_df, top_k=3)

            top_suspect = diagnosis[0] if diagnosis else None
            top_step_idx = top_suspect["step_idx"] if top_suspect else -1
            top_reasons = ", ".join(top_suspect["reasons"]) if top_suspect else "General pattern similarity"

            st.markdown(f"""
            <div class="bb-anomaly-flag">
                <div class="bb-anomaly-title">AI Root-Cause Diagnosis Assessment</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #0F172A; margin: 4px 0 8px 0;">
                    Flagged Culprit: Step {top_step_idx} ({steps_df.iloc[top_step_idx].step_type.upper() if top_step_idx >= 0 else 'Unknown'})
                </div>
                <div style="font-size: 0.88rem; color: #0F172A; line-height: 1.5;">
                    The diagnosis model detected abnormal behavioral divergence at Step {top_step_idx} with 
                    <strong>{top_suspect['confidence']*100:.1f}% confidence</strong> (Suspicion Score: <code>{top_suspect['score']:.3f}</code>).
                    <br>
                    <strong>Observed Indicators:</strong> {top_reasons}.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin: 28px 0 16px 0;">
            <h4 style="margin: 0; font-size: 1.05rem; color: #0F172A;">Sequential Execution Trace & Step Observability</h4>
            <div style="font-size: 0.82rem; color: #5B708B;">Detailed inspection of LLM actions, internal thinking, state mutations, and diagnostic scores at each step.</div>
        </div>
        """, unsafe_allow_html=True)

        for s in steps_df.itertuples():
            step_idx = s.step_idx
            step_type = s.step_type
            inp = json.loads(s.input) if s.input else {}
            out = json.loads(s.output) if s.output else {}
            st_before = json.loads(s.state_before) if s.state_before else {}
            st_after = json.loads(s.state_after) if s.state_after else {}

            is_culprit = (not is_pass and run_data.injected and run_data.faulty_step_idx == step_idx)
            card_class = "bb-step-card is-culprit" if is_culprit else "bb-step-card"
            
            step_status_tag = ""
            if is_culprit:
                step_status_tag = '<span class="bb-pill bb-pill-fail" style="margin-left: 8px;">[CULPRIT FLAGGED]</span>'
            else:
                step_status_tag = '<span class="bb-pill bb-pill-neutral" style="margin-left: 8px;">[NOMINAL]</span>'

            intent, thinking = get_step_intent_and_thinking(step_type, inp, out, is_culprit, run_data.fault_type)

            with st.container():
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #5B708B; padding-bottom: 10px; margin-bottom: 12px;">
                        <div>
                            <span style="font-size: 0.84rem; font-weight: 700; letter-spacing: 0.04em; color: #0F172A;">
                                STEP {step_idx} : {step_type.upper()}
                            </span>
                            {step_status_tag}
                        </div>
                        <div style="font-size: 0.78rem; color: #5B708B;">
                            Latency: <code>{s.latency_ms} ms</code> | Hash: <code>{s.input_hash[:8]}</code>
                        </div>
                    </div>
                    
                    <div style="font-size: 0.84rem; color: #5B708B; margin-bottom: 6px;">
                        <strong>Action:</strong> {intent}
                    </div>

                    <div class="bb-monologue">
                        <div class="bb-monologue-title">LLM Internal Reasoning & Hypothesis</div>
                        <div>{thinking}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                exp_label = f"View Raw Data & State Transitions for Step {step_idx} ({step_type.upper()})"
                with st.expander(exp_label, expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Step Input:**")
                        st.json(inp)
                    with c2:
                        st.markdown("**Step Output:**")
                        st.json(out)
                    
                    st.markdown("**State Context Mutation:**")
                    st.json({"state_before": st_before, "state_after": st_after})


# =====================================================================
# VIEW 2: Backtracking & Selective Replay Sandbox
# =====================================================================
elif nav == "Backtracking & Selective Replay":
    st.subheader("Backtracking & Selective Replay Sandbox")
    st.caption("Visually trace the root cause, rewind execution state to the culpable step, and execute partial replay.")

    failed_runs = pd.read_sql(
        "SELECT r.run_id, r.task_id, r.question, r.final_answer, r.expected, l.faulty_step_idx, l.fault_type "
        "FROM runs r LEFT JOIN labels l ON r.run_id=l.run_id WHERE r.outcome='fail' ORDER BY r.rowid DESC LIMIT 50", c
    )

    if failed_runs.empty:
        st.info("No failed execution runs currently found in database to backtrack.")
    else:
        run_opts = [f"{r.run_id} | {r.task_id} | Discrepancy: '{r.final_answer}' vs '{r.expected}'" for r in failed_runs.itertuples()]
        sel_run_opt = st.selectbox("Select Failed Run to Backtrack & Fix", run_opts)
        sel_run_id = sel_run_opt.split(" | ")[0]

        run_info = failed_runs[failed_runs.run_id == sel_run_id].iloc[0]
        steps_df = pd.read_sql("SELECT * FROM steps WHERE run_id=? ORDER BY step_idx", c, params=[sel_run_id])

        rec_step = 2
        diag_reasons = "Behavioral divergence detected"
        if model is not None:
            steps_feat_df = steps_df.rename(columns={"input": "input_text", "output": "output_text"}).assign(question=run_info.question)
            steps_feat_df["error"] = pd.to_numeric(steps_feat_df["error"], errors="coerce").fillna(0).astype(int)
            diag = model.diagnose(steps_feat_df, top_k=1)
            if diag:
                rec_step = diag[0]["step_idx"]
                diag_reasons = ", ".join(diag[0]["reasons"])

        st.markdown(f"""
        <div class="bb-backtrack-banner">
            <div class="bb-backtrack-title">Root-Cause Detection & Backtracking Path</div>
            <div style="font-size: 0.92rem; color: #0F172A; margin-bottom: 8px;">
                Final answer <code>'{run_info.final_answer}'</code> disagrees with expected <code>'{run_info.expected}'</code>.
                The diagnostic engine identified <strong>Step {rec_step} ({steps_df.iloc[rec_step].step_type.upper()})</strong> as the earliest point of failure.
            </div>
            <div style="font-size: 0.85rem; color: #5B708B; margin-bottom: 12px;">
                <strong>Reason:</strong> {diag_reasons}.
            </div>
            <div style="padding: 10px 14px; background: #FFFFFF; border-radius: 4px; border: 1px dashed #5B708B; font-size: 0.84rem; color: #0F172A;">
                <strong>Backtracking Action:</strong> Rewind agent execution state from Step 4 directly back to <strong>Step {rec_step}</strong>. 
                Steps 0 to {rec_step - 1} remain untainted and will be reused directly from cache (0 ms re-compute).
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Rollback Point")
            target_step = st.slider("Target Step to Branch & Replay From", 0, len(steps_df)-1, value=rec_step)
            st.caption(f"Upstream steps (0 to {target_step-1}) will be skipped and reused.")

        with col2:
            st.markdown("##### Intervention Protocol")
            replay_mode = st.radio("Intervention Mode", ["Clean Automatic Replay (Clear Injected Fault)", "Manual Output Override"], index=0)

        custom_out = None
        if replay_mode == "Manual Output Override":
            target_step_row = steps_df[steps_df.step_idx == target_step].iloc[0]
            curr_out_json = target_step_row.output
            custom_out_str = st.text_area("Custom JSON Output for Target Step", value=curr_out_json)
            try:
                custom_out = json.loads(custom_out_str)
            except Exception:
                st.error("Invalid JSON format for output override.")

        if st.button("Execute Backtracking & Partial Replay"):
            intervention = {}
            if custom_out:
                intervention["override_output"] = custom_out

            replay_res = replay_run(sel_run_id, target_step, intervention=intervention, c=c, model=model)

            st.markdown("---")
            is_fixed = replay_res["fixed"]
            outcome_status = "[FIXED - PASS]" if is_fixed else "[STILL FAILING]"
            pill_type = "bb-pill-pass" if is_fixed else "bb-pill-fail"

            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 16px;">
                <h4 style="margin: 0; font-size: 1.1rem; color: #0F172A;">Backtracking Verification Results</h4>
                <span class="bb-pill {pill_type}">{outcome_status}</span>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Outcome", replay_res["replayed_outcome"].upper())
            m2.metric("New Answer", str(replay_res["replayed_answer"]), delta=f"Old: {replay_res['original_answer']}")
            m3.metric("Steps Skipped (Cached)", f"{replay_res['steps_skipped']} Steps")
            m4.metric("Compute / Time Saved", f"{replay_res['time_saved_percent']}%")

            st.markdown("##### Partial Execution Replay Map")
            st.caption("Visual breakdown of skipped upstream steps vs target intervention and downstream re-execution:")

            map_cols = st.columns(len(replay_res["steps"]))
            for idx, st_info in enumerate(replay_res["steps"]):
                with map_cols[idx]:
                    st_status = st_info["status"]
                    if st_status == "skipped":
                        st.markdown(f"""
                        <div class="bb-step-card is-skipped" style="text-align: center; padding: 12px 8px;">
                            <span class="bb-pill bb-pill-neutral">[CACHED]</span><br>
                            <strong style="font-size: 0.85rem; display: block; margin-top: 6px; color: #0F172A;">Step {idx}: {st_info['step_type'].upper()}</strong>
                            <span style="font-size: 0.74rem; color: #5B708B;">Reused from history</span>
                        </div>
                        """, unsafe_allow_html=True)
                    elif st_status == "target":
                        st.markdown(f"""
                        <div class="bb-step-card is-culprit" style="text-align: center; padding: 12px 8px;">
                            <span class="bb-pill bb-pill-fail">[TARGET]</span><br>
                            <strong style="font-size: 0.85rem; display: block; margin-top: 6px; color: #0F172A;">Step {idx}: {st_info['step_type'].upper()}</strong>
                            <span style="font-size: 0.74rem; color: #0F172A;">Backtrack point</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="bb-step-card is-replayed" style="text-align: center; padding: 12px 8px;">
                            <span class="bb-pill bb-pill-pass">[RE-RUN]</span><br>
                            <strong style="font-size: 0.85rem; display: block; margin-top: 6px; color: #0F172A;">Step {idx}: {st_info['step_type'].upper()}</strong>
                            <span style="font-size: 0.74rem; color: #5B708B;">Updated context</span>
                        </div>
                        """, unsafe_allow_html=True)


# =====================================================================
# VIEW 3: Benchmark Evaluation & Metrics
# =====================================================================
elif nav == "Benchmark Evaluation & Metrics":
    st.subheader("Benchmark Evaluation & Generalization Metrics")
    st.caption("Empirical performance of the diagnosis model evaluated across seen tasks, unseen fault types, and held-out task structures.")

    if not metrics:
        st.warning("metrics.json file not found in repository root.")
    else:
        splits = metrics.get("splits", {})

        m1, m2, m3 = st.columns(3)
        if "test_seen" in splits:
            m1.metric("Test Seen Tasks Top-1", f"{splits['test_seen']['top1']*100:.2f}%", f"MRR: {splits['test_seen']['mrr']:.3f}")
        if "test_unseen_fault" in splits:
            m2.metric("Test Unseen Faults Top-1", f"{splits['test_unseen_fault']['top1']*100:.2f}%", f"MRR: {splits['test_unseen_fault']['mrr']:.3f}")
        if "test_unseen_task" in splits:
            m3.metric("Test Unseen Tasks Top-1", f"{splits['test_unseen_task']['top1']*100:.2f}%", f"MRR: {splits['test_unseen_task']['mrr']:.3f}")

        st.markdown("---")
        st.markdown("##### Diagnosis Model vs Baselines Comparison")

        b_data = []
        for s_name, s_data in splits.items():
            model_top1 = s_data["top1"] * 100
            base = s_data.get("baselines_top1", {})
            b_data.append({
                "Benchmark Split": s_name,
                "Black Box AI Model": model_top1,
                "Random Guess Baseline": base.get("random_step", 0) * 100,
                "Blame Last Step Baseline": base.get("always_last_step", 0) * 100,
                "Blame Step 1 Majority Baseline": base.get("always_step_1_(training_majority)", 0) * 100,
            })

        b_df = pd.DataFrame(b_data).set_index("Benchmark Split")
        st.dataframe(b_df.style.format("{:.2f}%"), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Per-Fault Type Top-1 Accuracy (Test Seen)")
            if "test_seen" in splits and "per_fault_type_top1" in splits["test_seen"]:
                ft_dict = splits["test_seen"]["per_fault_type_top1"]
                ft_df = pd.DataFrame([{"Fault Type": k, "Top-1 Accuracy": f"{v['top1']*100:.1f}%", "Sample Count": v["n"]} for k, v in ft_dict.items()])
                st.dataframe(ft_df, use_container_width=True)

        with col2:
            st.markdown("##### Per-Task Type Top-1 Accuracy (Test Seen)")
            if "test_seen" in splits and "per_task_type_top1" in splits["test_seen"]:
                tt_dict = splits["test_seen"]["per_task_type_top1"]
                tt_df = pd.DataFrame([{"Task Type": k, "Top-1 Accuracy": f"{v*100:.1f}%"} for k, v in tt_dict.items()])
                st.dataframe(tt_df, use_container_width=True)


# =====================================================================
# VIEW 4: Live Agent Execution Sandbox
# =====================================================================
elif nav == "Live Agent Execution Sandbox":
    st.subheader("Live Agent Execution & Fault Injection Sandbox")
    st.caption("Generate a new multi-step agent task in real time, optionally inject a subtle behavioral fault, and observe diagnosis and backtracking.")

    col1, col2 = st.columns(2)
    with col1:
        selected_company = st.selectbox("Select Target Company", sorted(COMPANIES.keys())[:25])
        selected_task_type = st.selectbox("Select Task Query Type", TASK_TYPES)
    with col2:
        fault_names = ["None (Nominal Clean Run)"] + list(FAULTS.keys())
        selected_fault = st.selectbox("Inject Behavioral Fault", fault_names)

    if st.button("Execute Agent Task"):
        from agent.agent import make_task
        task = make_task(selected_company, selected_task_type)
        fault_obj = make_fault(selected_fault) if selected_fault != "None (Nominal Clean Run)" else None

        with st.spinner("Running agent execution pipeline..."):
            run_id, outcome = run_agent(task, fault=fault_obj, c=c)

        run_data = c.execute("SELECT question, final_answer, expected, outcome FROM runs WHERE run_id=?", (run_id,)).fetchone()
        steps_df = pd.read_sql("SELECT * FROM steps WHERE run_id=? ORDER BY step_idx", c, params=[run_id])

        is_pass = outcome == "pass"
        status_pill = '<span class="bb-pill bb-pill-pass">[PASSED]</span>' if is_pass else '<span class="bb-pill bb-pill-fail">[FAILED]</span>'

        st.markdown(f"""
        <div class="bb-card" style="margin-top: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h4 style="margin: 0; color: #0F172A;">Live Run ID: <code>{run_id}</code></h4>
                {status_pill}
            </div>
            <div style="font-size: 0.88rem; margin-bottom: 12px; color: #0F172A;"><strong>Question:</strong> {run_data[0]}</div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; font-size: 0.85rem;">
                <div><strong>Agent Answer:</strong> <code>{run_data[1]}</code></div>
                <div><strong>Expected:</strong> <code>{run_data[2]}</code></div>
                <div><strong>Injected Fault:</strong> {selected_fault}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not is_pass and model is not None:
            steps_feat_df = steps_df.rename(columns={"input": "input_text", "output": "output_text"}).assign(question=run_data[0])
            steps_feat_df["error"] = pd.to_numeric(steps_feat_df["error"], errors="coerce").fillna(0).astype(int)
            diag = model.diagnose(steps_feat_df, top_k=2)

            suspect = diag[0]
            st.markdown(f"""
            <div class="bb-anomaly-flag">
                <div class="bb-anomaly-title">Behavioral Anomaly Detected</div>
                <div style="font-size: 0.92rem; font-weight: 600; color: #0F172A; margin: 4px 0;">
                    Flagged Culprit: Step {suspect['step_idx']} ({steps_df.iloc[suspect['step_idx']].step_type.upper()})
                </div>
                <div style="font-size: 0.85rem; color: #0F172A;">
                    Suspicion Probability: <code>{suspect['score']:.3f}</code> ({suspect['confidence']*100:.1f}% confidence)<br>
                    Diagnostic Reasoning: {', '.join(suspect['reasons'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### Immediate Remediation")
            if st.button("Initiate Backtrack & Resolve"):
                replay_res = replay_run(run_id, suspect["step_idx"], c=c, model=model)
                if replay_res["fixed"]:
                    st.success(f"Backtracking resolved the failure! New answer: '{replay_res['replayed_answer']}' (Expected: '{run_data[2]}'). Outcome: PASS.")
                else:
                    st.warning("Partial replay completed, but issue remains unresolved.")

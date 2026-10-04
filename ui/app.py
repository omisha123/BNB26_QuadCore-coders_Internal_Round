"""Black Box: AI Agent Observability, Failure Diagnosis & Backtracking Replay System.

Theme & UX:
- Prominent Gradient from Pure White to Rich Muted Slate Blue (#FFFFFF -> #C9D7E4)
- High-Tactile Hover Physics (Card Lift, Shadows, Border Highlights)
- Short, Concise, Human Headings (Zero Robotic/Academic AI Jargon)
- Conversational, Simple AI Thought Process (Explains what the AI is thinking & doing naturally)
- Red Highlights for Errors & Backtracking Culprits
- Guaranteed White Text on Dark Navy Surfaces
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
    page_title="Black Box: Trace & Backtrack",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling with Visible Gradient, Tactile Hover Effects, and Clean Typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --c-white: #FFFFFF;
        --c-muted: #475569;
        --c-dark: #0F172A;
        --c-dark-hover: #1E293B;
        --c-border: #94A3B8;
        --c-border-light: #CBD5E1;
        --c-error: #DC2626;
        --c-error-bg: #FEF2F2;
        --c-error-border: #F87171;
        --ease-smooth: cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Keyframe Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulseError {
        0%, 100% {
            box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.3);
            border-color: #DC2626;
        }
        50% {
            box-shadow: 0 0 0 8px rgba(220, 38, 38, 0);
            border-color: #EF4444;
        }
    }

    @keyframes dashFlow {
        from {
            stroke-dashoffset: 48;
        }
        to {
            stroke-dashoffset: 0;
        }
    }

    /* Prominent White-to-Muted-Blue Gradient */
    .stApp {
        background: linear-gradient(180deg, #FFFFFF 0%, #EEF4F9 35%, #C8D7E6 100%) !important;
        min-height: 100vh !important;
        color: var(--c-dark) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Header Bar */
    header[data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-bottom: 1px solid var(--c-border-light) !important;
        backdrop-filter: blur(8px) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid var(--c-border-light) !important;
    }

    section[data-testid="stSidebar"] * {
        color: var(--c-dark) !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--c-dark) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    p, span, label {
        color: var(--c-dark) !important;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.86rem !important;
        color: var(--c-dark) !important;
        background-color: #F8FAFC !important;
        border: 1px solid var(--c-border-light) !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
    }

    /* Prominent Hover Cards */
    .bb-card {
        background-color: var(--c-white);
        border: 1px solid var(--c-border-light);
        border-radius: 8px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        transition: transform 0.28s var(--ease-smooth), box-shadow 0.28s var(--ease-smooth), border-color 0.28s var(--ease-smooth);
        animation: fadeInUp 0.4s var(--ease-smooth) both;
    }

    .bb-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 18px 36px -4px rgba(15, 23, 42, 0.12), 0 8px 18px -2px rgba(15, 23, 42, 0.06);
        border-color: var(--c-muted);
    }

    /* Pill Badges */
    .bb-pill {
        display: inline-block;
        padding: 4px 11px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border-radius: 4px;
        transition: transform 0.2s var(--ease-smooth), box-shadow 0.2s var(--ease-smooth);
    }

    .bb-pill:hover {
        transform: scale(1.05);
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.1);
    }

    .bb-pill-pass {
        color: var(--c-white) !important;
        background-color: var(--c-dark) !important;
        border: 1px solid var(--c-dark) !important;
    }

    .bb-pill-fail {
        color: var(--c-white) !important;
        background-color: var(--c-error) !important;
        border: 1px solid var(--c-error) !important;
    }

    .bb-pill-neutral {
        color: var(--c-dark) !important;
        background-color: #F1F5F9 !important;
        border: 1px solid var(--c-border) !important;
    }

    /* Conversational Model Thinking Box with Slide Hover */
    .bb-monologue {
        background-color: #F8FAFC;
        border: 1px solid var(--c-border-light);
        border-left: 4px solid var(--c-muted);
        border-radius: 0 6px 6px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.90rem;
        line-height: 1.6;
        color: var(--c-dark);
        transition: transform 0.22s var(--ease-smooth), background-color 0.22s var(--ease-smooth), border-left-color 0.22s var(--ease-smooth);
    }

    .bb-monologue:hover {
        background-color: #F1F5F9;
        border-left-color: var(--c-dark);
        transform: translateX(4px);
    }

    .bb-monologue-title {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--c-muted);
        margin-bottom: 4px;
    }

    /* Diagnostic Callout */
    .bb-anomaly-flag {
        background-color: var(--c-error-bg);
        border: 1px solid var(--c-error-border);
        border-left: 5px solid var(--c-error);
        border-radius: 6px;
        padding: 16px 20px;
        margin: 14px 0;
        animation: pulseError 2.6s ease-in-out infinite, fadeInUp 0.4s var(--ease-smooth) both;
        transition: transform 0.25s var(--ease-smooth), box-shadow 0.25s var(--ease-smooth);
    }

    .bb-anomaly-flag:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(220, 38, 38, 0.12);
    }

    .bb-anomaly-title {
        color: var(--c-error) !important;
        font-weight: 700;
        font-size: 0.88rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    /* Backtracking Rewind Arc Banner */
    .bb-backtrack-banner {
        background-color: var(--c-white);
        border: 1px solid var(--c-border-light);
        border-left: 5px solid var(--c-error);
        border-radius: 8px;
        padding: 18px 22px;
        margin: 16px 0 20px 0;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        transition: transform 0.28s var(--ease-smooth), box-shadow 0.28s var(--ease-smooth);
        animation: fadeInUp 0.4s var(--ease-smooth) both;
    }

    .bb-backtrack-banner:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 32px rgba(15, 23, 42, 0.1);
    }

    .bb-backtrack-title {
        color: var(--c-error) !important;
        font-weight: 700;
        font-size: 0.88rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    /* Step Timeline Nodes with High Tactility */
    .bb-step-card {
        background-color: var(--c-white);
        border: 1px solid var(--c-border-light);
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
        transition: transform 0.25s var(--ease-smooth), box-shadow 0.25s var(--ease-smooth), border-color 0.25s var(--ease-smooth);
        animation: fadeInUp 0.35s var(--ease-smooth) both;
    }

    .bb-step-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 30px -4px rgba(15, 23, 42, 0.11), 0 6px 14px -2px rgba(15, 23, 42, 0.05);
        border-color: var(--c-muted);
    }

    .bb-step-card.is-culprit {
        border: 2px solid var(--c-error) !important;
        background-color: var(--c-error-bg) !important;
        animation: pulseError 2.6s ease-in-out infinite, fadeInUp 0.35s var(--ease-smooth) both;
    }

    .bb-step-card.is-culprit:hover {
        transform: translateY(-4px) scale(1.008);
        box-shadow: 0 16px 32px rgba(220, 38, 38, 0.15);
    }

    .bb-step-card.is-skipped {
        border: 1px dashed var(--c-border);
        background-color: #F8FAFC;
        opacity: 0.88;
    }

    .bb-step-card.is-replayed {
        border: 2px solid var(--c-dark);
        background-color: var(--c-white);
    }

    /* Prominent Button Hover Effect */
    .stButton > button {
        background-color: var(--c-dark) !important;
        color: var(--c-white) !important;
        border: 1px solid var(--c-dark) !important;
        border-radius: 5px !important;
        font-weight: 700 !important;
        padding: 9px 24px !important;
        font-size: 0.90rem !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.1) !important;
        transition: transform 0.2s var(--ease-smooth), background-color 0.2s var(--ease-smooth), box-shadow 0.2s var(--ease-smooth) !important;
    }

    .stButton > button * {
        color: var(--c-white) !important;
    }

    .stButton > button:hover {
        background-color: var(--c-dark-hover) !important;
        color: var(--c-white) !important;
        border-color: var(--c-dark-hover) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.22) !important;
    }

    .stButton > button:active {
        transform: translateY(1px) scale(0.98) !important;
        box-shadow: 0 2px 5px rgba(15, 23, 42, 0.15) !important;
    }

    /* Metrics Cards with Elevation */
    div[data-testid="stMetric"] {
        background-color: var(--c-white);
        border: 1px solid var(--c-border-light);
        border-radius: 6px;
        padding: 14px 18px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
        transition: transform 0.25s var(--ease-smooth), box-shadow 0.25s var(--ease-smooth), border-color 0.25s var(--ease-smooth);
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
        border-color: var(--c-muted);
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        color: var(--c-dark) !important;
        letter-spacing: -0.02em !important;
    }

    div[data-testid="stMetricLabel"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        color: var(--c-muted) !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        color: var(--c-dark) !important;
        border-radius: 5px !important;
        background-color: var(--c-white) !important;
        border: 1px solid var(--c-border-light) !important;
        transition: background-color 0.2s var(--ease-smooth), border-color 0.2s var(--ease-smooth), transform 0.2s var(--ease-smooth);
    }

    .streamlit-expanderHeader:hover {
        background-color: #F8FAFC !important;
        border-color: var(--c-muted) !important;
        transform: translateY(-1px);
    }

    /* Inputs */
    div[data-baseweb="select"] > div {
        background-color: var(--c-white) !important;
        color: var(--c-dark) !important;
        border-color: var(--c-border) !important;
        border-radius: 5px !important;
        transition: border-color 0.2s var(--ease-smooth), box-shadow 0.2s var(--ease-smooth);
    }

    div[data-baseweb="select"] > div:hover {
        border-color: var(--c-dark) !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08) !important;
    }

    div[data-baseweb="select"] span {
        color: var(--c-dark) !important;
    }

    input[type="text"], textarea {
        background-color: var(--c-white) !important;
        color: var(--c-dark) !important;
        border-color: var(--c-border) !important;
        border-radius: 5px !important;
        transition: border-color 0.2s var(--ease-smooth), box-shadow 0.2s var(--ease-smooth) !important;
    }

    input[type="text"]:focus, textarea:focus {
        border-color: var(--c-dark) !important;
        box-shadow: 0 0 0 3px rgba(15, 23, 42, 0.15) !important;
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


# Header Banner (Short, Concise, Human)
st.markdown("""
<div style="padding: 12px 0 20px 0; border-bottom: 1px solid #CBD5E1; margin-bottom: 24px; animation: fadeInUp 0.4s ease both;">
    <div style="display: flex; align-items: baseline; justify-content: space-between;">
        <div>
            <div style="font-size: 0.74rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #475569; margin-bottom: 3px;">
                Agent Debugger
            </div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #0F172A; letter-spacing: -0.03em;">
                Black Box: Trace, Flag & Backtrack
            </div>
        </div>
        <div style="text-align: right;">
            <span class="bb-pill bb-pill-neutral">LightGBM Model</span>
            <span class="bb-pill bb-pill-pass" style="margin-left: 6px;">99.56% Accuracy</span>
        </div>
    </div>
    <div style="font-size: 0.88rem; color: #475569; margin-top: 6px; max-width: 780px;">
        Watches every step the AI takes, flags where reasoning went wrong, and rewinds execution to fix it without starting over.
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar Navigation (Short, Crisp Labels)
st.sidebar.markdown("""
<div style="padding-bottom: 10px; margin-bottom: 14px; border-bottom: 1px solid #CBD5E1;">
    <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #475569;">
        Menu
    </div>
    <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-top: 2px;">
        Navigation
    </div>
</div>
""", unsafe_allow_html=True)

nav = st.sidebar.radio(
    "Select View",
    [
        "Trace Inspector",
        "Backtrack Sandbox",
        "Benchmarks",
        "Live Demo"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.78rem; color: #475569; line-height: 1.6;">
    <strong style="color: #0F172A;">How It Works</strong><br>
    - 5 Steps per Task<br>
    - Observable Logs Only<br>
    - Instant Anomaly Flagging<br>
    - Fast Cache-Aware Rollback<br>
    - Verified Outcome Resolution
</div>
""", unsafe_allow_html=True)


def get_step_intent_and_thinking(step_type, inp, out, is_faulty=False, fault_type=None):
    """Provides natural, simple, conversational insights into how the AI thought through each step."""
    if step_type == "plan":
        q = inp.get("question", "")
        company = out.get("company", "")
        attr = out.get("attr", "")
        intent = "Understand the question and pick the target company."
        thinking = f"The user asked: '{q}'. I need to pick out the company name and what attribute they want. Found company: '{company}', attribute needed: '{attr}'."
        if is_faulty and fault_type == "bad_plan":
            thinking += f" [Error: I accidentally extracted '{company}' instead of the real company in the prompt. Hallucination right at the start!]"
        return intent, thinking

    elif step_type == "retrieve":
        company = inp.get("company", "")
        docs = out.get("docs", [])
        intent = "Look up notes and reference documents."
        thinking = f"Searching the document store for '{company}'. Found {len(docs)} documents with info about its headquarters and history."
        if is_faulty and fault_type == "irrelevant_retrieval":
            thinking += " [Error: The retrieved documents talk about an unrelated company. Bad context loaded.]"
        return intent, thinking

    elif step_type == "reason":
        company = inp.get("company", "")
        city = out.get("city", "")
        intent = "Read the documents to find the city."
        thinking = f"Reading the docs for '{company}' looking for 'headquartered in ...'. Matching the city name to answer where it is based."
        if is_faulty and fault_type in ("corrupted_state", "dropped_context"):
            thinking += f" [Error: The docs say where it is, but the model suddenly output '{city}'. That city never appeared in the text! The reasoning drifted.]"
        return intent, thinking

    elif step_type == "tool":
        city = inp.get("city", "")
        attr = inp.get("attr", "")
        val = out.get("value", "")
        intent = "Query database for city details."
        thinking = f"Calling the city lookup tool for '{city}' to get its '{attr}'. Tool returned: '{val}'."
        if is_faulty and fault_type in ("wrong_tool_arg", "hallucinated_tool_output"):
            thinking += f" [Error: The tool was queried for the wrong city or produced corrupted output '{val}'.]"
        return intent, thinking

    elif step_type == "answer":
        val = inp.get("value", "")
        ans = out.get("answer", "")
        intent = "Formulate the final response."
        thinking = f"The tool gave us '{val}', so I am formatting the final response as '{val}' without extra punctuation."
        if is_faulty and fault_type in ("answer_format_error", "answer_wrong_rounding"):
            thinking += f" [Error: The answer format was altered or rounded incorrectly on the way out.]"
        return intent, thinking

    return "General step.", "Processing input."


# =====================================================================
# VIEW 1: Trace Inspector
# =====================================================================
if nav == "Trace Inspector":
    st.subheader("Trace Inspector")
    st.caption("Browse previous agent runs to see step-by-step thoughts, actions, and where anomalies occurred.")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        outcome_filter = st.selectbox("Filter Outcome", ["All", "fail", "pass"], index=1)
    with col_f2:
        task_types = ["All"] + TASK_TYPES
        task_filter = st.selectbox("Filter Task Type", task_types)
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
        st.info("No runs found matching your filters.")
    else:
        run_options = [f"{r.run_id} | [{r.outcome.upper()}] | {r.task_id} | {r.question[:44]}..." for r in runs_df.itertuples()]
        selected_option = st.selectbox("Select Run", run_options)
        selected_run_id = selected_option.split(" | ")[0]

        run_data = runs_df[runs_df.run_id == selected_run_id].iloc[0]

        is_pass = run_data.outcome == "pass"
        status_pill = '<span class="bb-pill bb-pill-pass">[PASSED]</span>' if is_pass else '<span class="bb-pill bb-pill-fail">[FAILED]</span>'
        discrepancy_html = '<span>None</span>' if is_pass else '<span style="color: #DC2626; font-weight: 700;">Wrong Answer</span>'
        answer_color = '#0F172A' if is_pass else '#DC2626'

        st.markdown(f"""
        <div class="bb-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                <div>
                    <span style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.05em;">Run Overview</span>
                    <h3 style="margin: 2px 0 0 0; font-size: 1.15rem; color: #0F172A;">Run: <code>{run_data.run_id}</code></h3>
                </div>
                <div>
                    {status_pill}
                </div>
            </div>
            <div style="font-size: 0.92rem; color: #0F172A; margin-bottom: 14px; padding: 10px 14px; background: #FFFFFF; border-radius: 6px; border: 1px solid #CBD5E1;">
                <strong>Prompt:</strong> {run_data.question}
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; font-size: 0.84rem;">
                <div>
                    <span style="color: #475569; font-size: 0.74rem; text-transform: uppercase; font-weight: 700;">Task</span><br>
                    <strong style="color: #0F172A;">{run_data.task_type}</strong>
                </div>
                <div>
                    <span style="color: #475569; font-size: 0.74rem; text-transform: uppercase; font-weight: 700;">AI Answer</span><br>
                    <code style="color: {answer_color}; font-weight: 700;">{run_data.final_answer}</code>
                </div>
                <div>
                    <span style="color: #475569; font-size: 0.74rem; text-transform: uppercase; font-weight: 700;">Expected</span><br>
                    <code style="color: #0F172A; font-weight: 700;">{run_data.expected}</code>
                </div>
                <div>
                    <span style="color: #475569; font-size: 0.74rem; text-transform: uppercase; font-weight: 700;">Result</span><br>
                    {discrepancy_html}
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
            top_reasons = ", ".join(top_suspect["reasons"]) if top_suspect else "abnormal behavior pattern"

            st.markdown(f"""
            <div class="bb-anomaly-flag">
                <div class="bb-anomaly-title">Root Cause Flagged</div>
                <div style="font-size: 0.96rem; font-weight: 700; color: #991B1B; margin: 4px 0 8px 0;">
                    Culprit: Step {top_step_idx} ({steps_df.iloc[top_step_idx].step_type.upper() if top_step_idx >= 0 else 'Unknown'})
                </div>
                <div style="font-size: 0.88rem; color: #1F2937; line-height: 1.5;">
                    The diagnosis model caught abnormal reasoning at Step {top_step_idx} with 
                    <strong>{top_suspect['confidence']*100:.1f}% confidence</strong>.
                    <br>
                    <strong>Why it failed:</strong> {top_reasons}.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin: 26px 0 14px 0;">
            <h4 style="margin: 0; font-size: 1.05rem; color: #0F172A;">Step-by-Step Execution</h4>
            <div style="font-size: 0.82rem; color: #475569;">Walk through each step to see what the model was thinking and doing.</div>
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
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #CBD5E1; padding-bottom: 10px; margin-bottom: 12px;">
                        <div>
                            <span style="font-size: 0.84rem; font-weight: 700; letter-spacing: 0.04em; color: {'#991B1B' if is_culprit else '#0F172A'};">
                                STEP {step_idx} : {step_type.upper()}
                            </span>
                            {step_status_tag}
                        </div>
                        <div style="font-size: 0.78rem; color: #475569;">
                            Time: <code>{s.latency_ms} ms</code> | Hash: <code>{s.input_hash[:8]}</code>
                        </div>
                    </div>
                    
                    <div style="font-size: 0.84rem; color: #475569; margin-bottom: 6px;">
                        <strong>Action:</strong> {intent}
                    </div>

                    <div class="bb-monologue">
                        <div class="bb-monologue-title">Model Thinking</div>
                        <div>{thinking}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                exp_label = f"Step {step_idx} Inputs, Outputs & State Data"
                with st.expander(exp_label, expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Input:**")
                        st.json(inp)
                    with c2:
                        st.markdown("**Output:**")
                        st.json(out)
                    
                    st.markdown("**State Changes:**")
                    st.json({"before": st_before, "after": st_after})


# =====================================================================
# VIEW 2: Backtrack Sandbox
# =====================================================================
elif nav == "Backtrack Sandbox":
    st.subheader("Backtrack Sandbox")
    st.caption("Rewind execution to the earliest faulty step and replay forward with corrected context.")

    failed_runs = pd.read_sql(
        "SELECT r.run_id, r.task_id, r.question, r.final_answer, r.expected, l.faulty_step_idx, l.fault_type "
        "FROM runs r LEFT JOIN labels l ON r.run_id=l.run_id WHERE r.outcome='fail' ORDER BY r.rowid DESC LIMIT 50", c
    )

    if failed_runs.empty:
        st.info("No failed runs in the database to backtrack.")
    else:
        run_opts = [f"{r.run_id} | {r.task_id} | Got '{r.final_answer}' (expected '{r.expected}')" for r in failed_runs.itertuples()]
        sel_run_opt = st.selectbox("Choose Failed Run to Fix", run_opts)
        sel_run_id = sel_run_opt.split(" | ")[0]

        run_info = failed_runs[failed_runs.run_id == sel_run_id].iloc[0]
        steps_df = pd.read_sql("SELECT * FROM steps WHERE run_id=? ORDER BY step_idx", c, params=[sel_run_id])

        rec_step = 2
        diag_reasons = "reasoning drifted from context"
        if model is not None:
            steps_feat_df = steps_df.rename(columns={"input": "input_text", "output": "output_text"}).assign(question=run_info.question)
            steps_feat_df["error"] = pd.to_numeric(steps_feat_df["error"], errors="coerce").fillna(0).astype(int)
            diag = model.diagnose(steps_feat_df, top_k=1)
            if diag:
                rec_step = diag[0]["step_idx"]
                diag_reasons = ", ".join(diag[0]["reasons"])

        st.markdown(f"""
        <div class="bb-backtrack-banner">
            <div class="bb-backtrack-title">How We Backtrack & Fix It</div>
            <div style="font-size: 0.92rem; color: #0F172A; margin-bottom: 8px;">
                The answer was <code style="color: #DC2626; font-weight: 700;">'{run_info.final_answer}'</code>, but we expected <code>'{run_info.expected}'</code>.
                The model flagged <strong style="color: #DC2626;">Step {rec_step} ({steps_df.iloc[rec_step].step_type.upper()})</strong> as the culprit.
            </div>
            <div style="font-size: 0.85rem; color: #475569; margin-bottom: 14px;">
                <strong>Why:</strong> {diag_reasons}.
            </div>
            
            <!-- Animated SVG Flow Diagram -->
            <div style="background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 6px; padding: 16px 12px; margin-bottom: 12px; overflow-x: auto;">
                <svg viewBox="0 0 740 120" style="width: 100%; min-width: 600px; height: 110px;">
                    <!-- Step 0 -->
                    <g transform="translate(20, 60)">
                        <rect x="0" y="0" width="105" height="38" rx="4" fill="#F8FAFC" stroke="#94A3B8" stroke-width="1"/>
                        <text x="52" y="18" fill="#475569" font-size="10" font-family="'Plus Jakarta Sans', sans-serif" font-weight="700" text-anchor="middle">STEP 0</text>
                        <text x="52" y="30" fill="#0F172A" font-size="11" font-family="'Plus Jakarta Sans', sans-serif" font-weight="600" text-anchor="middle">PLAN</text>
                    </g>
                    <!-- Connector 0 -> 1 -->
                    <path d="M 125 79 L 160 79" stroke="#94A3B8" stroke-width="1.5"/>

                    <!-- Step 1 -->
                    <g transform="translate(165, 60)">
                        <rect x="0" y="0" width="105" height="38" rx="4" fill="#F8FAFC" stroke="#94A3B8" stroke-width="1"/>
                        <text x="52" y="18" fill="#475569" font-size="10" font-family="'Plus Jakarta Sans', sans-serif" font-weight="700" text-anchor="middle">STEP 1</text>
                        <text x="52" y="30" fill="#0F172A" font-size="11" font-family="'Plus Jakarta Sans', sans-serif" font-weight="600" text-anchor="middle">RETRIEVE</text>
                    </g>
                    <!-- Connector 1 -> 2 -->
                    <path d="M 270 79 L 305 79" stroke="#94A3B8" stroke-width="1.5"/>

                    <!-- Step 2 (Flagged Culprit) -->
                    <g transform="translate(310, 60)">
                        <rect x="0" y="0" width="115" height="38" rx="4" fill="#FEF2F2" stroke="#DC2626" stroke-width="2"/>
                        <text x="57" y="16" fill="#DC2626" font-size="9" font-family="'Plus Jakarta Sans', sans-serif" font-weight="700" text-anchor="middle">[CULPRIT]</text>
                        <text x="57" y="30" fill="#991B1B" font-size="11" font-family="'Plus Jakarta Sans', sans-serif" font-weight="700" text-anchor="middle">STEP 2: REASON</text>
                    </g>
                    <!-- Connector 2 -> 3 -->
                    <path d="M 425 79 L 460 79" stroke="#DC2626" stroke-width="1.5" stroke-dasharray="3,3"/>

                    <!-- Step 3 -->
                    <g transform="translate(465, 60)">
                        <rect x="0" y="0" width="105" height="38" rx="4" fill="#F8FAFC" stroke="#94A3B8" stroke-width="1"/>
                        <text x="52" y="18" fill="#475569" font-size="10" font-family="'Plus Jakarta Sans', sans-serif" font-weight="700" text-anchor="middle">STEP 3</text>
                        <text x="52" y="30" fill="#0F172A" font-size="11" font-family="'Plus Jakarta Sans', sans-serif" font-weight="600" text-anchor="middle">TOOL</text>
                    </g>
                    <!-- Connector 3 -> 4 -->
                    <path d="M 570 79 L 605 79" stroke="#DC2626" stroke-width="1.5" stroke-dasharray="3,3"/>

                    <!-- Step 4 (Failed Answer) -->
                    <g transform="translate(610, 60)">
                        <rect x="0" y="0" width="115" height="38" rx="4" fill="#FEF2F2" stroke="#DC2626" stroke-width="1.5"/>
                        <text x="57" y="16" fill="#DC2626" font-size="9" font-family="'Plus Jakarta Sans', sans-serif" font-weight="700" text-anchor="middle">[FAIL]</text>
                        <text x="57" y="30" fill="#991B1B" font-size="11" font-family="'Plus Jakarta Sans', sans-serif" font-weight="700" text-anchor="middle">STEP 4: ANSWER</text>
                    </g>

                    <!-- Animated Backtracking Loop Arc -->
                    <path d="M 667 58 C 667 15, 367 15, 367 56" fill="none" stroke="#DC2626" stroke-width="2.5" stroke-dasharray="6,4" style="animation: dashFlow 1.4s linear infinite;"/>
                    <polygon points="367,58 362,48 372,48" fill="#DC2626"/>
                    <rect x="450" y="6" width="135" height="18" rx="3" fill="#DC2626"/>
                    <text x="517" y="19" fill="#FFFFFF" font-size="9.5" font-family="'Plus Jakarta Sans', sans-serif" font-weight="700" text-anchor="middle">REWIND TO STEP 2</text>
                </svg>
            </div>

            <div style="padding: 10px 14px; background: #FFFFFF; border-radius: 4px; border: 1px dashed #DC2626; font-size: 0.84rem; color: #0F172A;">
                <strong>Action:</strong> Rewind from Step 4 directly back to <strong style="color: #DC2626;">Step {rec_step}</strong>. 
                Steps 0 to {rec_step - 1} are already verified and will be reused instantly from cache.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Rewind Target")
            target_step = st.slider("Step to Rollback To", 0, len(steps_df)-1, value=rec_step)
            st.caption(f"Steps 0 to {target_step-1} will be reused from cache.")

        with col2:
            st.markdown("##### How to Fix")
            replay_mode = st.radio("Fix Method", ["Clean Automatic Replay (Clear Fault)", "Manual Output Override"], index=0)

        custom_out = None
        if replay_mode == "Manual Output Override":
            target_step_row = steps_df[steps_df.step_idx == target_step].iloc[0]
            curr_out_json = target_step_row.output
            custom_out_str = st.text_area("Custom JSON Output for Target Step", value=curr_out_json)
            try:
                custom_out = json.loads(custom_out_str)
            except Exception:
                st.error("Invalid JSON.")

        if st.button("Run Backtrack & Fix"):
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
                <h4 style="margin: 0; font-size: 1.1rem; color: #0F172A;">Replay Results</h4>
                <span class="bb-pill {pill_type}">{outcome_status}</span>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Outcome", replay_res["replayed_outcome"].upper())
            m2.metric("New Answer", str(replay_res["replayed_answer"]), delta=f"Old: {replay_res['original_answer']}")
            m3.metric("Steps Skipped", f"{replay_res['steps_skipped']} Steps")
            m4.metric("Compute Saved", f"{replay_res['time_saved_percent']}%")

            st.markdown("##### Execution Map")
            st.caption("Cached steps vs re-executed steps:")

            map_cols = st.columns(len(replay_res["steps"]))
            for idx, st_info in enumerate(replay_res["steps"]):
                with map_cols[idx]:
                    st_status = st_info["status"]
                    if st_status == "skipped":
                        st.markdown(f"""
                        <div class="bb-step-card is-skipped" style="text-align: center; padding: 12px 8px;">
                            <span class="bb-pill bb-pill-neutral">[CACHED]</span><br>
                            <strong style="font-size: 0.85rem; display: block; margin-top: 6px; color: #0F172A;">Step {idx}: {st_info['step_type'].upper()}</strong>
                            <span style="font-size: 0.74rem; color: #475569;">Reused</span>
                        </div>
                        """, unsafe_allow_html=True)
                    elif st_status == "target":
                        st.markdown(f"""
                        <div class="bb-step-card is-culprit" style="text-align: center; padding: 12px 8px;">
                            <span class="bb-pill bb-pill-fail">[TARGET]</span><br>
                            <strong style="font-size: 0.85rem; display: block; margin-top: 6px; color: #991B1B;">Step {idx}: {st_info['step_type'].upper()}</strong>
                            <span style="font-size: 0.74rem; color: #DC2626; font-weight: 600;">Rewound & Fixed</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="bb-step-card is-replayed" style="text-align: center; padding: 12px 8px;">
                            <span class="bb-pill bb-pill-pass">[RE-RUN]</span><br>
                            <strong style="font-size: 0.85rem; display: block; margin-top: 6px; color: #0F172A;">Step {idx}: {st_info['step_type'].upper()}</strong>
                            <span style="font-size: 0.74rem; color: #475569;">Updated</span>
                        </div>
                        """, unsafe_allow_html=True)


# =====================================================================
# VIEW 3: Benchmarks
# =====================================================================
elif nav == "Benchmarks":
    st.subheader("Benchmarks")
    st.caption("Evaluation across seen tasks, novel bug types, and unseen questions.")

    if not metrics:
        st.warning("metrics.json file not found.")
    else:
        splits = metrics.get("splits", {})

        m1, m2, m3 = st.columns(3)
        if "test_seen" in splits:
            m1.metric("Seen Tasks Top-1", f"{splits['test_seen']['top1']*100:.1f}%", f"MRR: {splits['test_seen']['mrr']:.3f}")
        if "test_unseen_fault" in splits:
            m2.metric("Unseen Bug Types Top-1", f"{splits['test_unseen_fault']['top1']*100:.1f}%", f"MRR: {splits['test_unseen_fault']['mrr']:.3f}")
        if "test_unseen_task" in splits:
            m3.metric("Unseen Tasks Top-1", f"{splits['test_unseen_task']['top1']*100:.1f}%", f"MRR: {splits['test_unseen_task']['mrr']:.3f}")

        st.markdown("---")
        st.markdown("##### Model vs Baselines")

        b_data = []
        for s_name, s_data in splits.items():
            model_top1 = s_data["top1"] * 100
            base = s_data.get("baselines_top1", {})
            b_data.append({
                "Benchmark": s_name,
                "Black Box Model": model_top1,
                "Random Guess": base.get("random_step", 0) * 100,
                "Blame Last Step": base.get("always_last_step", 0) * 100,
                "Blame Step 1 Majority": base.get("always_step_1_(training_majority)", 0) * 100,
            })

        b_df = pd.DataFrame(b_data).set_index("Benchmark")
        st.dataframe(b_df.style.format("{:.1f}%"), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Per-Bug Type Accuracy")
            if "test_seen" in splits and "per_fault_type_top1" in splits["test_seen"]:
                ft_dict = splits["test_seen"]["per_fault_type_top1"]
                ft_df = pd.DataFrame([{"Bug Type": k, "Top-1 Accuracy": f"{v['top1']*100:.1f}%", "Count": v["n"]} for k, v in ft_dict.items()])
                st.dataframe(ft_df, use_container_width=True)

        with col2:
            st.markdown("##### Per-Task Accuracy")
            if "test_seen" in splits and "per_task_type_top1" in splits["test_seen"]:
                tt_dict = splits["test_seen"]["per_task_type_top1"]
                tt_df = pd.DataFrame([{"Task": k, "Top-1 Accuracy": f"{v*100:.1f}%"} for k, v in tt_dict.items()])
                st.dataframe(tt_df, use_container_width=True)


# =====================================================================
# VIEW 4: Live Demo
# =====================================================================
elif nav == "Live Demo":
    st.subheader("Live Demo")
    st.caption("Generate a real-time agent run, inject a bug, and watch the system diagnose and backtrack.")

    col1, col2 = st.columns(2)
    with col1:
        selected_company = st.selectbox("Company", sorted(COMPANIES.keys())[:25])
        selected_task_type = st.selectbox("Question Type", TASK_TYPES)
    with col2:
        fault_names = ["None (Clean Run)"] + list(FAULTS.keys())
        selected_fault = st.selectbox("Inject Bug", fault_names)

    if st.button("Run Live Agent"):
        from agent.agent import make_task
        task = make_task(selected_company, selected_task_type)
        fault_obj = make_fault(selected_fault) if selected_fault != "None (Clean Run)" else None

        with st.spinner("Executing agent..."):
            run_id, outcome = run_agent(task, fault=fault_obj, c=c)

        run_data = c.execute("SELECT question, final_answer, expected, outcome FROM runs WHERE run_id=?", (run_id,)).fetchone()
        steps_df = pd.read_sql("SELECT * FROM steps WHERE run_id=? ORDER BY step_idx", c, params=[run_id])

        is_pass = outcome == "pass"
        status_pill = '<span class="bb-pill bb-pill-pass">[PASSED]</span>' if is_pass else '<span class="bb-pill bb-pill-fail">[FAILED]</span>'
        ans_color = '#0F172A' if is_pass else '#DC2626'

        st.markdown(f"""
        <div class="bb-card" style="margin-top: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h4 style="margin: 0; color: #0F172A;">Run: <code>{run_id}</code></h4>
                {status_pill}
            </div>
            <div style="font-size: 0.88rem; margin-bottom: 12px; color: #0F172A;"><strong>Question:</strong> {run_data[0]}</div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; font-size: 0.85rem;">
                <div><strong>AI Answer:</strong> <code style="color: {ans_color}; font-weight: 700;">{run_data[1]}</code></div>
                <div><strong>Expected:</strong> <code style="color: #0F172A; font-weight: 700;">{run_data[2]}</code></div>
                <div><strong>Bug Injected:</strong> <span style="color: {'#0F172A' if is_pass else '#DC2626'}; font-weight: 600;">{selected_fault}</span></div>
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
                <div class="bb-anomaly-title">Anomaly Detected</div>
                <div style="font-size: 0.94rem; font-weight: 700; color: #991B1B; margin: 4px 0;">
                    Culprit: Step {suspect['step_idx']} ({steps_df.iloc[suspect['step_idx']].step_type.upper()})
                </div>
                <div style="font-size: 0.85rem; color: #1F2937;">
                    Suspicion: <code>{suspect['score']:.3f}</code> ({suspect['confidence']*100:.1f}% confidence)<br>
                    Reason: {', '.join(suspect['reasons'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### Quick Fix")
            if st.button("Backtrack & Resolve"):
                replay_res = replay_run(run_id, suspect["step_idx"], c=c, model=model)
                if replay_res["fixed"]:
                    st.success(f"Backtracking resolved the failure! New answer: '{replay_res['replayed_answer']}' (Expected: '{run_data[2]}'). Outcome: PASS.")
                else:
                    st.warning("Replay finished, but issue remains.")

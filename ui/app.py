"""Black Box - AI-Powered Debugging & Selective Replay System UI.

Run with: py -3.13 -m streamlit run ui/app.py
"""
import os
import sys
import json
import sqlite3
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import schema
from agent.agent import all_tasks, run_agent, COMPANIES, TASK_TYPES
from agent.faults import FAULTS, make_fault
from model.model import BlackBoxModel
from replay.replay import replay_run

# Page Config
st.set_page_config(
    page_title="Black Box - AI Debugging System",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme & sleek UI
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2e3440;
        text-align: center;
    }
    .status-pass {
        color: #43b581;
        font-weight: bold;
    }
    .status-fail {
        color: #f04747;
        font-weight: bold;
    }
    .step-box-skipped {
        background-color: #2b2d42;
        border-left: 4px solid #8d99ae;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .step-box-target {
        background-color: #1d3557;
        border-left: 4px solid #e63946;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .step-box-downstream {
        background-color: #1b4332;
        border-left: 4px solid #52b788;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 8px;
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

# Header
st.title("◼ Black Box: AI Agent Debugging & Replay System")
st.caption("Learns from agent execution traces to pinpoint failure-causing steps and verify partial replay fixes without repeating unaffected execution.")

# Sidebar Navigation
st.sidebar.title("Navigation")
nav = st.sidebar.radio(
    "Select View",
    [
        "🔍 Trace Inspector & Diagnosis",
        "🔄 Replay & Fix Sandbox",
        "📈 Evaluation & Metrics",
        "⚡ Live Agent Execution Demo"
    ]
)

c = schema.conn()

# --- View 1: Trace Inspector & Diagnosis ---
if nav == "🔍 Trace Inspector & Diagnosis":
    st.header("🔍 Execution Trace Inspector & AI Root-Cause Diagnosis")
    
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
        st.warning("No runs found matching the selected filters.")
    else:
        run_options = [f"{r.run_id} | {r.outcome.upper()} | {r.task_id} | Question: {r.question[:40]}..." for r in runs_df.itertuples()]
        selected_option = st.selectbox("Select Execution Run", run_options)
        selected_run_id = selected_option.split(" | ")[0]
        
        run_data = runs_df[runs_df.run_id == selected_run_id].iloc[0]
        
        # Meta Card
        st.markdown("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Run ID", run_data.run_id)
        m2.metric("Task Type", run_data.task_type)
        m3.metric("Outcome", run_data.outcome.upper(), delta="PASSED" if run_data.outcome=="pass" else "FAILED", delta_color="normal" if run_data.outcome=="pass" else "inverse")
        m4.metric("Final Answer", str(run_data.final_answer))
        m5.metric("Expected", str(run_data.expected))

        st.markdown(f"**Task Question:** `{run_data.question}`")
        
        # Fetch steps
        steps_df = pd.read_sql("SELECT * FROM steps WHERE run_id=? ORDER BY step_idx", c, params=[selected_run_id])
        
        # Diagnosis
        if run_data.outcome == "fail" and model is not None:
            st.subheader("🤖 AI Root-Cause Diagnosis")
            
            steps_feat_df = steps_df.rename(columns={"input": "input_text", "output": "output_text"}).assign(question=run_data.question)
            steps_feat_df["error"] = pd.to_numeric(steps_feat_df["error"], errors="coerce").fillna(0).astype(int)
            fail_score = model.run_failure_score(steps_feat_df).iloc[0]
            diagnosis = model.diagnose(steps_feat_df, top_k=3)
            
            d_col1, d_col2 = st.columns([1, 2])
            with d_col1:
                st.metric("Failure Suspicion Score", f"{fail_score:.3f}")
                if run_data.injected and run_data.faulty_step_idx is not None:
                    st.info(f"**Ground Truth Benchmark Label:**\n- Faulty Step: Step {run_data.faulty_step_idx}\n- Fault Type: `{run_data.fault_type}`")
            with d_col2:
                st.write("**Ranked Suspicious Steps:**")
                for rank, suspect in enumerate(diagnosis, 1):
                    step_num = suspect["step_idx"]
                    conf = suspect["confidence"]
                    reasons = ", ".join(suspect["reasons"])
                    badge = "🎯 TOP SUSPECT" if rank == 1 else f"Suspect #{rank}"
                    st.write(f"- **{badge} (Step {step_num})**: Probability `{suspect['score']:.3f}` ({conf*100:.1f}% confidence)")
                    st.write(f"  *Reasons:* {reasons}")
        
        # Step Timeline
        st.subheader("📜 Step-by-Step Execution Trace")
        
        for s in steps_df.itertuples():
            step_idx = s.step_idx
            step_type = s.step_type
            inp = json.loads(s.input) if s.input else {}
            out = json.loads(s.output) if s.output else {}
            st_before = json.loads(s.state_before) if s.state_before else {}
            st_after = json.loads(s.state_after) if s.state_after else {}
            
            is_faulty = (run_data.outcome == "fail" and run_data.injected and run_data.faulty_step_idx == step_idx)
            header = f"Step {step_idx}: {step_type.upper()}"
            if is_faulty:
                header += " ⚠️ [GROUND TRUTH FAULTY STEP]"
                
            with st.expander(header, expanded=(is_faulty or step_idx==0)):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Input:**")
                    st.json(inp)
                with c2:
                    st.write("**Output:**")
                    st.json(out)
                st.write("**State After Step:**")
                st.json(st_after)


# --- View 2: Replay & Fix Sandbox ---
elif nav == "🔄 Replay & Fix Sandbox":
    st.header("🔄 Replay & Selective Execution Sandbox")
    st.markdown("Re-run execution starting from a suspicious step while skipping unaffected upstream steps.")

    failed_runs = pd.read_sql("SELECT r.run_id, r.task_id, r.question, r.final_answer, r.expected, l.faulty_step_idx FROM runs r LEFT JOIN labels l ON r.run_id=l.run_id WHERE r.outcome='fail' ORDER BY r.rowid DESC LIMIT 50", c)
    
    if failed_runs.empty:
        st.warning("No failed runs in database to replay.")
    else:
        run_opts = [f"{r.run_id} | {r.task_id} | Question: {r.question[:40]}..." for r in failed_runs.itertuples()]
        sel_run_opt = st.selectbox("Select Failed Run to Replay", run_opts)
        sel_run_id = sel_run_opt.split(" | ")[0]
        
        run_info = failed_runs[failed_runs.run_id == sel_run_id].iloc[0]
        steps_df = pd.read_sql("SELECT * FROM steps WHERE run_id=? ORDER BY step_idx", c, params=[sel_run_id])
        
        # Determine AI recommendation
        rec_step = 2
        if model is not None:
            steps_feat_df = steps_df.rename(columns={"input": "input_text", "output": "output_text"}).assign(question=run_info.question)
            steps_feat_df["error"] = pd.to_numeric(steps_feat_df["error"], errors="coerce").fillna(0).astype(int)
            diag = model.diagnose(steps_feat_df, top_k=1)
            if diag:
                rec_step = diag[0]["step_idx"]

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown("### Original Failed Run Details")
            st.write(f"- **Run ID:** `{run_info.run_id}`")
            st.write(f"- **Question:** {run_info.question}")
            st.write(f"- **Original Answer:** `{run_info.final_answer}` (FAILED)")
            st.write(f"- **Expected Answer:** `{run_info.expected}`")
            st.info(f"💡 **AI Recommendation:** Replay starting from **Step {rec_step}**")

        with r_col2:
            st.markdown("### Replay Settings")
            target_step = st.slider("Target Step Index to Branch From", 0, len(steps_df)-1, value=rec_step)
            
            replay_mode = st.radio("Intervention Mode", ["Clean Automatic Replay (Clear Fault)", "Custom Output Override"], index=0)
            
            custom_out = None
            if replay_mode == "Custom Output Override":
                target_step_row = steps_df[steps_df.step_idx == target_step].iloc[0]
                curr_out_json = target_step_row.output
                custom_out_str = st.text_area("Custom JSON Output for Target Step", value=curr_out_json)
                try:
                    custom_out = json.loads(custom_out_str)
                except Exception as e:
                    st.error("Invalid JSON string for output override.")

        if st.button("🚀 Run Partial Replay Execution"):
            intervention = {}
            if custom_out:
                intervention["override_output"] = custom_out
                
            replay_res = replay_run(sel_run_id, target_step, intervention=intervention, c=c, model=model)
            
            st.success("Replay Completed Successfully!")
            st.markdown("---")
            st.subheader("📊 Replay Verification Results")
            
            res_c1, res_c2, res_c3, res_c4 = st.columns(4)
            res_c1.metric("Outcome", replay_res["replayed_outcome"].upper(), delta="FIXED (PASS)" if replay_res["fixed"] else "STILL FAILING", delta_color="normal" if replay_res["fixed"] else "inverse")
            res_c2.metric("New Answer", str(replay_res["replayed_answer"]), delta=f"Original: {replay_res['original_answer']}")
            res_c3.metric("Steps Skipped (Cached)", f"{replay_res['steps_skipped']} steps")
            res_c4.metric("Time & Compute Saved", f"{replay_res['time_saved_percent']}%")
            
            # Step Execution Map
            st.subheader("🗺️ Execution Map")
            map_cols = st.columns(len(replay_res["steps"]))
            for idx, st_info in enumerate(replay_res["steps"]):
                with map_cols[idx]:
                    if st_info["status"] == "skipped":
                        st.info(f"**Step {idx}: {st_info['step_type']}**\n\n⚡ SKIPPED (Cached)")
                    elif st_info["status"] == "target":
                        st.error(f"**Step {idx}: {st_info['step_type']}**\n\n🛠️ REPLAY TARGET")
                    else:
                        st.success(f"**Step {idx}: {st_info['step_type']}**\n\n🔄 RE-EXECUTED")


# --- View 3: Evaluation & Metrics ---
elif nav == "📈 Evaluation & Metrics":
    st.header("📈 Benchmark Evaluation & Feature Ablation Results")
    st.caption("Demonstrating model diagnosis accuracy evaluated against known injected failures across multiple splits.")

    if not metrics:
        st.warning("metrics.json not found. Run model/run_all.py to generate metrics.")
    else:
        splits = metrics.get("splits", {})
        
        # Summary metrics
        m1, m2, m3 = st.columns(3)
        if "test_seen" in splits:
            m1.metric("Test Seen Tasks Top-1 Acc", f"{splits['test_seen']['top1']*100:.1f}%", f"MRR: {splits['test_seen']['mrr']:.3f}")
        if "test_unseen_fault" in splits:
            m2.metric("Test Unseen Faults Top-1 Acc", f"{splits['test_unseen_fault']['top1']*100:.1f}%", f"MRR: {splits['test_unseen_fault']['mrr']:.3f}")
        if "test_unseen_task" in splits:
            m3.metric("Test Unseen Task Type Top-1 Acc", f"{splits['test_unseen_task']['top1']*100:.1f}%", f"MRR: {splits['test_unseen_task']['mrr']:.3f}")
            
        st.markdown("---")
        st.subheader("🎯 Model vs Baselines Comparison")
        
        b_data = []
        for s_name, s_data in splits.items():
            model_top1 = s_data["top1"] * 100
            base = s_data.get("baselines_top1", {})
            b_data.append({
                "Split": s_name,
                "Black Box AI Model": model_top1,
                "Random Step Baseline": base.get("random_step", 0) * 100,
                "Blame Last Step Baseline": base.get("always_last_step", 0) * 100,
                "Blame Majority Step Baseline": base.get("always_step_1_(training_majority)", 0) * 100,
            })
        
        b_df = pd.DataFrame(b_data).set_index("Split")
        st.bar_chart(b_df)
        
        st.subheader("🔬 Feature Ablation Study (Top-1 Accuracy)")
        if "ablation_top1" in metrics:
            abl_df = pd.DataFrame(metrics["ablation_top1"]).T * 100
            st.dataframe(abl_df.style.format("{:.1f}%").highlight_max(axis=0))
            
        st.subheader("📉 Learning Curve (Top-1 Accuracy vs Training Runs)")
        if "learning_curve" in metrics:
            lc_df = pd.DataFrame(metrics["learning_curve"]).set_index("train_runs")
            st.line_chart(lc_df)


# --- View 4: Live Agent Execution Demo ---
elif nav == "⚡ Live Agent Execution Demo":
    st.header("⚡ Live Agent Execution & Fault Injection Sandbox")
    st.markdown("Run a live multi-step agent task, optionally inject a fault, and watch Black Box diagnose it instantly.")

    demo_c1, demo_c2 = st.columns(2)
    with demo_c1:
        selected_company = st.selectbox("Select Target Company", sorted(COMPANIES.keys())[:20])
        selected_task_type = st.selectbox("Select Question Task Type", TASK_TYPES)
    with demo_c2:
        fault_names = ["None (Clean Run)"] + list(FAULTS.keys())
        selected_fault = st.selectbox("Inject Fault Type", fault_names)

    if st.button("▶️ Execute Live Agent Task"):
        task = {"task_id": f"{selected_task_type}_{selected_company}", "task_type": selected_task_type,
                "question": f"What is the {selected_task_type.replace('hq_', '')} of the city where {selected_company} is headquartered?",
                "expected": str(schema.conn().execute("SELECT outcome FROM runs LIMIT 1").fetchone() or "N/A")}
        
        # Real task object generator
        from agent.agent import make_task
        task = make_task(selected_company, selected_task_type)
        
        fault_obj = make_fault(selected_fault) if selected_fault != "None (Clean Run)" else None
        
        with st.spinner("Executing agent steps..."):
            run_id, outcome = run_agent(task, fault=fault_obj, c=c)
            
        st.success(f"Agent Execution Completed! Run ID: `{run_id}` | Outcome: **{outcome.upper()}**")
        
        # Show trace and immediate diagnosis
        run_data = c.execute("SELECT question, final_answer, expected, outcome FROM runs WHERE run_id=?", (run_id,)).fetchone()
        steps_df = pd.read_sql("SELECT * FROM steps WHERE run_id=? ORDER BY step_idx", c, params=[run_id])
        
        st.write(f"- **Question:** {run_data[0]}")
        st.write(f"- **Agent Answer:** `{run_data[1]}`")
        st.write(f"- **Expected Answer:** `{run_data[2]}`")
        
        if outcome == "fail" and model is not None:
            steps_feat_df = steps_df.rename(columns={"input": "input_text", "output": "output_text"}).assign(question=run_data[0])
            steps_feat_df["error"] = pd.to_numeric(steps_feat_df["error"], errors="coerce").fillna(0).astype(int)
            diag = model.diagnose(steps_feat_df, top_k=2)
            st.error("🚨 AI Root-Cause Diagnosis:")
            for suspect in diag:
                st.write(f"- **Step {suspect['step_idx']}**: Probability `{suspect['score']:.3f}` ({suspect['confidence']*100:.1f}% confidence)")
                st.write(f"  *Reasons:* {', '.join(suspect['reasons'])}")


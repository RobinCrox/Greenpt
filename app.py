import json
from dataclasses import asdict
from io import StringIO
from contextlib import redirect_stdout
 
import pandas as pd
import streamlit as st
 
from experiment import (
    run_condition_a,
    run_condition_b,
    run_condition_c,
    _build_summary_markdown,
)
 
 
st.set_page_config(
    page_title="GreenPT Brainstorming Experiment",
    page_icon="🌱",
    layout="wide",
)
 
 
# ---------- Styling ----------
 
st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
 
    .subtitle {
        color: #5f6f64;
        margin-bottom: 2rem;
    }
 
    .result-card {
        border: 1px solid #d9e6dd;
        border-radius: 16px;
        padding: 1.2rem;
        background: #f8fbf8;
        min-height: 420px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
 
    .condition-label {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
 
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 0.7rem;
        margin-top: 0.7rem;
        border: 1px solid #e5eee8;
    }
 
    .small-muted {
        color: #6c756f;
        font-size: 0.85rem;
    }
</style>
    """,
    unsafe_allow_html=True,
)
 
 
# ---------- Helper functions ----------
 
def get_final_assistant_message(result):
    """Return the final assistant answer from a ConditionResult."""
    assistant_messages = [
        msg["content"]
        for msg in result.conversation
        if msg["role"] == "assistant"
    ]
 
    if not assistant_messages:
        return "No assistant response generated."
 
    return assistant_messages[-1]
 
 
def result_to_metrics(result):
    """Convert result object into a metrics dictionary."""
    return {
        "Condition": result.condition,
        "Label": result.label,
        "Turns": result.num_turns,
        "Input tokens": result.total_input_tokens,
        "Output tokens": result.total_output_tokens,
        "Response chars": result.total_response_chars,
        "Cost USD": round(result.estimated_cost_usd, 5),
        "Efficiency": round(result.prompt_efficiency, 3),
        "Duration seconds": round(result.duration_seconds, 2),
    }
 
 
def run_with_captured_logs(function, goal):
    """
    Runs one condition while capturing terminal print output,
    so Streamlit does not become messy.
    """
    buffer = StringIO()
 
    with redirect_stdout(buffer):
        result = function(goal)
 
    logs = buffer.getvalue()
    return result, logs
 
 
def show_result_card(result, logs):
    """Display one condition result inside a visual box."""
    final_answer = get_final_assistant_message(result)
 
    with st.container(border=True):
        st.markdown(
            f'<div class="condition-label">Condition {result.condition}: {result.label}</div>'
            f'<div class="small-muted">Final assistant output</div>',
            unsafe_allow_html=True,
        )
        st.markdown(final_answer)
 
    with st.expander(f"Metrics — Condition {result.condition}"):
        col1, col2, col3 = st.columns(3)
 
        col1.metric("Input tokens", f"{result.total_input_tokens:,}")
        col2.metric("Output tokens", f"{result.total_output_tokens:,}")
        col3.metric("Cost", f"${result.estimated_cost_usd:.5f}")
 
        col4, col5, col6 = st.columns(3)
 
        col4.metric("Turns", result.num_turns)
        col5.metric("Efficiency", f"{result.prompt_efficiency:.3f}")
        col6.metric("Duration", f"{result.duration_seconds:.2f}s")
 
    with st.expander(f"Full conversation — Condition {result.condition}"):
        for msg in result.conversation:
            role = "User" if msg["role"] == "user" else "Assistant"
            st.markdown(f"**{role}:**")
            st.markdown(msg["content"])
 
    with st.expander(f"Debug logs — Condition {result.condition}"):
        st.code(logs)
 
 
# ---------- UI ----------
 
st.markdown('<div class="main-title">🌱 GreenPT Brainstorming Experiment UI</div>', unsafe_allow_html=True)
 
st.markdown(
    """
<div class="subtitle">
    Compare three brainstorming interaction methods: freeform chat, specialised freeform agent,
    and specialised structured agent.
</div>
    """,
    unsafe_allow_html=True,
)
 
with st.container():
    st.subheader("Prompt")
 
    goal = st.text_area(
        "Enter the brainstorming goal you want to test:",
        value="How might we get kids to eat more vegetables?",
        height=120,
    )
 
    run_button = st.button("Run all three conditions", type="primary")
 
 
if "results" not in st.session_state:
    st.session_state.results = None
 
if "logs" not in st.session_state:
    st.session_state.logs = None
 
 
if run_button:
    if not goal.strip():
        st.warning("Please enter a brainstorming goal first.")
    else:
        results = {}
        logs = {}
 
        progress = st.progress(0)
        status = st.empty()
 
        status.info("Running Condition A: Freeform Chat...")
        results["A"], logs["A"] = run_with_captured_logs(run_condition_a, goal)
        progress.progress(33)
 
        status.info("Running Condition B: Specialised Agent — Freeform...")
        results["B"], logs["B"] = run_with_captured_logs(run_condition_b, goal)
        progress.progress(66)
 
        status.info("Running Condition C: Specialised Agent — Structured...")
        results["C"], logs["C"] = run_with_captured_logs(run_condition_c, goal)
        progress.progress(100)
 
        status.success("All conditions completed.")
 
        st.session_state.results = results
        st.session_state.logs = logs
 
 
if st.session_state.results:
    results = st.session_state.results
    logs = st.session_state.logs
 
    st.divider()
 
    st.subheader("Results comparison")
 
    col_a, col_b, col_c = st.columns(3)
 
    with col_a:
        show_result_card(results["A"], logs["A"])
 
    with col_b:
        show_result_card(results["B"], logs["B"])
 
    with col_c:
        show_result_card(results["C"], logs["C"])
 
    st.divider()
 
    st.subheader("Metrics overview")
 
    metrics_df = pd.DataFrame(
        [
            result_to_metrics(results["A"]),
            result_to_metrics(results["B"]),
            result_to_metrics(results["C"]),
        ]
    )
 
    st.dataframe(metrics_df, use_container_width=True)
 
    st.download_button(
        label="Download metrics as CSV",
        data=metrics_df.to_csv(index=False),
        file_name="greenpt_metrics.csv",
        mime="text/csv",
    )
 
    json_payload = json.dumps(
        {
            key: asdict(value)
            for key, value in results.items()
        },
        indent=2,
        ensure_ascii=False,
    )
 
    st.download_button(
        label="Download full results as JSON",
        data=json_payload,
        file_name="greenpt_full_results.json",
        mime="application/json",
    )

    summary_md = _build_summary_markdown(
        [results["A"], results["B"], results["C"]]
    )

    st.download_button(
        label="Download results summary as Markdown",
        data=summary_md,
        file_name="greenpt_results_summary.md",
        mime="text/markdown",
    )
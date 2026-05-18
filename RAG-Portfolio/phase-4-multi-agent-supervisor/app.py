import streamlit as st

from src.graph import build_graph
from src.config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    ANTHROPIC_MODELS, OPENAI_MODELS,
    get_ollama_models,
)

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Multi-Agent Research System")
st.caption("Supervisor coordinates a Research Agent and Writer Agent to produce structured reports")

# --- Sidebar ---
with st.sidebar:
    st.header("Model")

    provider = st.selectbox(
        "Provider",
        ["ollama", "anthropic", "openai"],
        index=0,
        key="provider_select",
    )

    if provider == "ollama":
        available_models = get_ollama_models()
        default_idx = next(
            (i for i, m in enumerate(available_models) if "llama3.1" in m), 0
        )
        model = st.selectbox("Model", available_models, index=default_idx, key="model_select")
    elif provider == "anthropic":
        model = st.selectbox("Model", ANTHROPIC_MODELS, key="model_select")
        if not ANTHROPIC_API_KEY:
            st.warning("ANTHROPIC_API_KEY not set in .env")
    elif provider == "openai":
        model = st.selectbox("Model", OPENAI_MODELS, key="model_select")
        if not OPENAI_API_KEY:
            st.warning("OPENAI_API_KEY not set in .env")

    active_p = st.session_state.get("active_provider", "ollama")
    active_m = st.session_state.get("active_model", "llama3.1")
    st.caption(f"Active: **{active_p} / {active_m}**")

    st.divider()
    st.header("How it works")
    st.markdown(
        "**Three agents, one task:**\n\n"
        "🎯 **Supervisor** — receives the task, decides who acts next\n\n"
        "🔍 **Research Agent** — searches AI papers + web, writes notes\n\n"
        "✍️ **Writer Agent** — turns notes into a polished report\n\n"
        "---\n"
        "**Flow:**\n"
        "```\n"
        "Task → Supervisor\n"
        "     → Research Agent\n"
        "     → Supervisor\n"
        "     → Writer Agent\n"
        "     → Supervisor\n"
        "     → Final Report\n"
        "```"
    )
    st.divider()
    if st.button("🗑️ Clear", use_container_width=True):
        for key in ["workflow_log", "final_report", "research_notes"]:
            st.session_state.pop(key, None)
        st.rerun()

# --- Session state ---
if "workflow_log" not in st.session_state:
    st.session_state.workflow_log = []
if "final_report" not in st.session_state:
    st.session_state.final_report = ""
if "research_notes" not in st.session_state:
    st.session_state.research_notes = ""

# --- Main UI ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Research Task")
    task = st.text_area(
        "Enter your research topic or question:",
        placeholder=(
            "Examples:\n"
            "• How does the attention mechanism work in transformers?\n"
            "• What are the key differences between RAG and fine-tuning?\n"
            "• Explain GPT-3's few-shot learning approach"
        ),
        height=120,
        key="task_input",
    )

    run_btn = st.button(
        "🚀 Run Research",
        use_container_width=True,
        disabled=not task.strip(),
        type="primary",
    )

    if run_btn and task.strip():
        # Reset previous run
        st.session_state.workflow_log = []
        st.session_state.final_report = ""
        st.session_state.research_notes = ""
        st.session_state.active_provider = provider
        st.session_state.active_model = model

        graph = build_graph(provider, model)
        initial_state = {
            "task": task.strip(),
            "messages": [],
            "research_notes": "",
            "final_report": "",
            "next": "",
        }

        # Stream and display each step
        log_placeholder = st.empty()

        with st.spinner("Agents working..."):
            for step in graph.stream(initial_state, stream_mode="updates"):
                node_name = list(step.keys())[0]
                node_output = step[node_name]

                # Extract the latest message from this node
                msgs = node_output.get("messages", [])
                msg_text = msgs[-1].content if msgs else ""

                if node_name == "supervisor":
                    entry = f"🎯 **Supervisor:** {msg_text}"
                elif node_name == "research":
                    entry = f"🔍 **Research Agent:** {msg_text}"
                    if node_output.get("research_notes"):
                        st.session_state.research_notes = node_output["research_notes"]
                elif node_name == "write":
                    entry = f"✍️ **Writer Agent:** {msg_text}"
                    if node_output.get("final_report"):
                        st.session_state.final_report = node_output["final_report"]
                else:
                    entry = f"⚙️ **{node_name}:** {msg_text}"

                st.session_state.workflow_log.append(entry)

        st.rerun()

# --- Workflow log ---
if st.session_state.workflow_log:
    with col1:
        st.subheader("Agent Workflow")
        for entry in st.session_state.workflow_log:
            st.markdown(entry)

        if st.session_state.research_notes:
            with st.expander("🔍 Research Notes (click to expand)"):
                st.markdown(st.session_state.research_notes)

# --- Final report ---
with col2:
    st.subheader("Final Report")
    if st.session_state.final_report:
        st.markdown(st.session_state.final_report)
        st.divider()
        st.download_button(
            label="⬇️ Download Report",
            data=st.session_state.final_report,
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.info("Your report will appear here once the agents complete their work.")
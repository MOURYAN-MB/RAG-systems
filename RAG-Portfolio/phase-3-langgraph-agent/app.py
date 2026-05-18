import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.agent import build_agent
from src.config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    ANTHROPIC_MODELS, OPENAI_MODELS,
    get_ollama_models,
)

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Research Agent")
st.caption("Powered by LangGraph — watches the agent think, search, and reason in real time")

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
        default_idx = available_models.index("llama3.1") if "llama3.1" in available_models else 0
        model = st.selectbox("Model", available_models, index=default_idx, key="model_select")
    elif provider == "anthropic":
        model = st.selectbox("Model", ANTHROPIC_MODELS, key="model_select")
        if not ANTHROPIC_API_KEY:
            st.warning("ANTHROPIC_API_KEY not set in .env")
    elif provider == "openai":
        model = st.selectbox("Model", OPENAI_MODELS, key="model_select")
        if not OPENAI_API_KEY:
            st.warning("OPENAI_API_KEY not set in .env")

    if st.button("Apply Model", use_container_width=True):
        with st.spinner(f"Loading {model}..."):
            st.session_state.agent = build_agent(provider, model)
            st.session_state.active_provider = provider
            st.session_state.active_model = model
            st.session_state.messages = []
            st.session_state.lc_history = []
        st.success(f"Switched to {model}")
        st.rerun()

    active = st.session_state.get("active_model", "llama3.1")
    active_p = st.session_state.get("active_provider", "ollama")
    st.caption(f"Active: **{active_p} / {active}**")

    st.divider()
    st.header("How this agent works")
    st.markdown(
        "Unlike Phase 2 (which always retrieved from PDFs), this agent **decides** what to do:\n\n"
        "1. **Reads your question**\n"
        "2. **Chooses a tool** — AI papers or web search\n"
        "3. **Runs the tool** and reads the result\n"
        "4. **Loops** if it needs more info\n"
        "5. **Answers** with citations when satisfied\n\n"
        "You can see every step in the chat."
    )
    st.divider()
    st.markdown("**Tools available:**")
    st.markdown("- `search_ai_papers` — GPT-3, Transformers, RAG papers")
    st.markdown("- `web_search` — live internet via Tavily")
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.lc_history = []
        st.rerun()

# --- Session state ---
if "agent" not in st.session_state:
    with st.spinner("Loading agent..."):
        st.session_state.agent = build_agent("ollama", "llama3.1")
        st.session_state.active_provider = "ollama"
        st.session_state.active_model = "llama3.1"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "lc_history" not in st.session_state:
    st.session_state.lc_history = []


def _render_message(msg: dict):
    role = msg["role"]
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    elif role == "tool_call":
        with st.chat_message("assistant", avatar="🔧"):
            st.markdown(f"**Calling tool:** `{msg['tool_name']}`")
            st.caption(f"Input: {msg['tool_input']}")
    elif role == "tool_result":
        with st.chat_message("tool", avatar="📄"):
            with st.expander(f"Result from `{msg['tool_name']}`"):
                st.text(msg["content"][:1000] + ("..." if len(msg["content"]) > 1000 else ""))


# --- Render existing chat ---
for msg in st.session_state.messages:
    _render_message(msg)

# --- Chat input ---
question = st.chat_input("Ask about transformers, RAG, GPT-3, or anything AI-related...")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    input_state = {
        "messages": st.session_state.lc_history + [HumanMessage(content=question)]
    }

    final_answer = ""
    with st.spinner("Agent is thinking..."):
        for step in st.session_state.agent.stream(input_state, stream_mode="values"):
            last_msg = step["messages"][-1]

            if isinstance(last_msg, AIMessage):
                if last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        display = {
                            "role": "tool_call",
                            "tool_name": tc["name"],
                            "tool_input": str(tc["args"]),
                        }
                        st.session_state.messages.append(display)
                        _render_message(display)
                else:
                    final_answer = last_msg.content

            elif isinstance(last_msg, ToolMessage):
                tool_name = "tool"
                for prev in reversed(step["messages"][:-1]):
                    if isinstance(prev, AIMessage) and prev.tool_calls:
                        for tc in prev.tool_calls:
                            if tc["id"] == last_msg.tool_call_id:
                                tool_name = tc["name"]
                                break
                        break

                display = {
                    "role": "tool_result",
                    "tool_name": tool_name,
                    "content": last_msg.content,
                }
                st.session_state.messages.append(display)
                _render_message(display)

    if final_answer:
        with st.chat_message("assistant"):
            st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        st.session_state.lc_history.extend([
            HumanMessage(content=question),
            AIMessage(content=final_answer),
        ])
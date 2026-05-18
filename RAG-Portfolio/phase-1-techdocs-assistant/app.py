import streamlit as st
from src.chatbot import ask
from src.retriever import retrieve
from src.ingest import ingest
from src.config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    ANTHROPIC_MODELS, OPENAI_MODELS,
    get_ollama_models,
)

st.set_page_config(
    page_title="TechDocs Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 TechDocs Assistant")
st.caption("A RAG-powered chatbot that answers questions from your documents.")

# --- Sidebar ---
with st.sidebar:
    # --- Model Selection ---
    st.header("Model")

    provider = st.selectbox(
        "Provider",
        ["ollama", "anthropic", "openai"],
        index=0,
        key="provider_select",
    )

    if provider == "ollama":
        available_models = get_ollama_models()
        model = st.selectbox("Model", available_models, key="model_select")
    elif provider == "anthropic":
        model = st.selectbox("Model", ANTHROPIC_MODELS, key="model_select")
        if not ANTHROPIC_API_KEY:
            st.warning("ANTHROPIC_API_KEY not set in .env")
    elif provider == "openai":
        model = st.selectbox("Model", OPENAI_MODELS, key="model_select")
        if not OPENAI_API_KEY:
            st.warning("OPENAI_API_KEY not set in .env")

    active = st.session_state.get("active_model", "llama3.2")
    active_p = st.session_state.get("active_provider", "ollama")
    st.caption(f"Active: **{active_p} / {active}**")

    st.divider()

    # --- Controls ---
    st.header("Controls")

    if st.button("🔄 Re-ingest Documents", use_container_width=True):
        with st.spinner("Ingesting documents..."):
            ingest(force=True)
        st.success("Documents ingested!")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Your question is embedded\n"
        "2. Most relevant chunks are retrieved\n"
        "3. Chunks + question → LLM\n"
        "4. Answer is grounded in your docs"
    )

# --- Session state init ---
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_provider" not in st.session_state:
    st.session_state.active_provider = "ollama"
if "active_model" not in st.session_state:
    st.session_state.active_model = "llama3.2"

# --- Chat display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Input ---
question = st.chat_input("Ask a question about your documents...")

if question:
    # Update active model tracking
    st.session_state.active_provider = provider
    st.session_state.active_model = model

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    chunks = retrieve(question)
    with st.expander(f"📎 {len(chunks)} source chunk(s) retrieved", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(f"**[{i}] Source:** `{chunk['source']}` — distance: `{chunk['distance']}`")
            st.text(chunk["text"])
            st.divider()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, st.session_state.history = ask(
                question, st.session_state.history, provider, model
            )
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
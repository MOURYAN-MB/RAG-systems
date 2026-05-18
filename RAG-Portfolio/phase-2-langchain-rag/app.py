from pathlib import Path
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.loader import load_and_split
from src.vectorstore import build_vectorstore, load_vectorstore, vectorstore_exists
from src.chain import build_chain
from src.config import (
    DATA_DIR, ANTHROPIC_API_KEY, OPENAI_API_KEY,
    ANTHROPIC_MODELS, OPENAI_MODELS, get_ollama_models,
)

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 AI Research Paper Assistant")
st.caption("Ask questions across landmark AI papers — powered by LangChain + Ollama")

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

    if st.button("Apply Model", use_container_width=True):
        if "vectorstore" in st.session_state:
            with st.spinner(f"Loading {model}..."):
                st.session_state.chain = build_chain(
                    st.session_state.vectorstore, provider, model
                )
                st.session_state.active_provider = provider
                st.session_state.active_model = model
                st.session_state.messages = []
                st.session_state.chat_history = []
            st.success(f"Switched to {model}")
            st.rerun()
        else:
            st.warning("Build the index first, then apply a model.")

    active = st.session_state.get("active_model", "llama3.2")
    active_p = st.session_state.get("active_provider", "ollama")
    st.caption(f"Active: **{active_p} / {active}**")

    st.divider()

    # --- Knowledge Base ---
    st.header("Knowledge Base")

    papers = list(DATA_DIR.glob("**/*.pdf"))
    if papers:
        st.markdown(f"**{len(papers)} paper(s) loaded:**")
        for p in papers:
            st.markdown(f"- `{p.name}`")
    else:
        st.warning("No PDFs found in `data/raw/`")

    st.divider()

    if st.button("🔄 Build / Rebuild Index", use_container_width=True):
        with st.spinner("Loading and chunking papers..."):
            chunks = load_and_split()
        with st.spinner(f"Embedding {len(chunks)} chunks..."):
            vs = build_vectorstore(chunks)
            st.session_state.vectorstore = vs
            st.session_state.chain = build_chain(vs, provider, model)
            st.session_state.active_provider = provider
            st.session_state.active_model = model
            st.session_state.messages = []
            st.session_state.chat_history = []
        st.success(f"Done — {len(chunks)} chunks indexed!")
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown("**Phase 2 upgrades vs Phase 1**")
    st.markdown(
        "- Real PDFs with page numbers\n"
        "- `RecursiveCharacterTextSplitter`\n"
        "- MMR retrieval (diversity + relevance)\n"
        "- LCEL chain (modern LangChain)\n"
        "- Multi-document source tracking"
    )

# --- Auto-load existing vectorstore ---
if "chain" not in st.session_state:
    if vectorstore_exists():
        with st.spinner("Loading existing index..."):
            vs = load_vectorstore()
            st.session_state.vectorstore = vs
            st.session_state.chain = build_chain(vs)
            st.session_state.active_provider = "ollama"
            st.session_state.active_model = "llama3.2"
    else:
        st.info("No index found. Click **Build / Rebuild Index** in the sidebar.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 {len(msg['sources'])} source(s)"):
                for src in msg["sources"]:
                    name = Path(src.metadata.get("source", "unknown")).name
                    page = src.metadata.get("page", 0)
                    st.markdown(f"**{name}** — page {page + 1}")
                    st.caption(src.page_content[:300] + "...")
                    st.divider()

# --- Chat input ---
if "chain" in st.session_state:
    question = st.chat_input("Ask about Transformers, RAG, GPT-3, attention mechanisms...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Searching papers..."):
                result = st.session_state.chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history,
                })
                answer = result["answer"]
                sources = result.get("source_documents", [])

            st.markdown(answer)

            if sources:
                with st.expander(f"📄 {len(sources)} source(s) used"):
                    for src in sources:
                        name = Path(src.metadata.get("source", "unknown")).name
                        page = src.metadata.get("page", 0)
                        st.markdown(f"**{name}** — page {page + 1}")
                        st.caption(src.page_content[:300] + "...")
                        st.divider()

        st.session_state.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer),
        ])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
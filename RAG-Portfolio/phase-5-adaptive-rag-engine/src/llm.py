from src.config import OLLAMA_BASE_URL, ANTHROPIC_API_KEY, OPENAI_API_KEY
from src.retrieval import RetrievedChunk

_SYSTEM_PROMPT = """You are an expert research assistant. Answer questions strictly
based on the retrieved context. Always cite the document name and page number.
If the context does not contain the answer, say so clearly — do not guess."""


def build_llm(provider: str = "ollama", model: str = "llama3.1"):
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=0.1)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=ANTHROPIC_API_KEY, temperature=0.1)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=OPENAI_API_KEY, temperature=0.1)
    raise ValueError(f"Unknown provider: {provider}")


def synthesize(query: str, chunks: list[RetrievedChunk], llm) -> str:
    """Combine retrieved chunks with the query and call the LLM."""
    from langchain_core.messages import SystemMessage, HumanMessage

    context = "\n\n---\n\n".join(
        f"[{c.source} | page {c.page + 1}]\n{c.text}" for c in chunks
    )
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content
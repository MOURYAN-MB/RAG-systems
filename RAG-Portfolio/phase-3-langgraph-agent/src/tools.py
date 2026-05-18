import os
from pathlib import Path

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import PHASE2_VECTORSTORE_DIR, EMBEDDING_MODEL, COLLECTION_NAME, TAVILY_API_KEY

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


@tool
def search_ai_papers(query: str) -> str:
    """Search AI research papers (Attention Is All You Need, RAG paper, GPT-3) for
    technical questions about transformers, attention mechanisms, language models,
    retrieval-augmented generation, and few-shot learning."""
    vs = Chroma(
        persist_directory=str(PHASE2_VECTORSTORE_DIR),
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        collection_name=COLLECTION_NAME,
    )
    docs = vs.similarity_search(query, k=4)
    if not docs:
        return "No relevant content found in the AI research papers."
    return "\n\n---\n\n".join(
        f"[{Path(d.metadata.get('source', '?')).name} | page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in docs
    )


web_search = TavilySearchResults(
    max_results=3,
    description=(
        "Search the web for current events, recent AI news, product announcements, "
        "or any topic not covered in the AI research papers."
    ),
)
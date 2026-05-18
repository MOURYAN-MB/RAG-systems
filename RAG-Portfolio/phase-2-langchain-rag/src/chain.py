from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL, ANTHROPIC_API_KEY, OPENAI_API_KEY, TOP_K

_SYSTEM_PROMPT = """You are an expert AI research assistant with deep knowledge of machine learning papers.
Answer questions strictly based on the retrieved context below.
Always cite the paper name and page number when answering.
If the answer is not in the context, clearly say so — do not guess.

Context:
{context}"""


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(
        f"[{Path(d.metadata.get('source', '?')).name}  |  page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in docs
    )


def _build_llm(provider: str, model: str):
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


def build_chain(vectorstore, provider: str = "ollama", model: str = None):
    if model is None:
        model = OLLAMA_MODEL

    llm = _build_llm(provider, model)

    # MMR retrieval: balances relevance + diversity across chunks
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 10},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def run(inputs: dict) -> dict:
        docs = retriever.invoke(inputs["input"])
        answer = (prompt | llm | StrOutputParser()).invoke({
            "context": _format_docs(docs),
            "chat_history": inputs.get("chat_history", []),
            "input": inputs["input"],
        })
        return {"answer": answer, "source_documents": docs}

    return RunnableLambda(run)
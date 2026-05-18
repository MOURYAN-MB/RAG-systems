from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage

from src.state import AgentState
from src.tools import search_ai_papers, web_search
from src.config import OLLAMA_BASE_URL, ANTHROPIC_API_KEY, OPENAI_API_KEY

_SYSTEM_MESSAGE = SystemMessage(content=(
    "You are an AI research assistant with two tools:\n"
    "1. search_ai_papers — searches a database of AI research papers (Transformers, RAG, GPT-3). "
    "Use this for technical questions about model architecture, training, or specific paper content.\n"
    "2. web_search — searches the live internet. "
    "Use this for recent events, news, or topics not in the papers.\n\n"
    "Always use a tool before answering. Think step by step. "
    "After getting tool results, synthesize a clear answer with citations."
))

_TOOLS = [search_ai_papers, web_search]


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


def build_agent(provider: str = "ollama", model: str = "llama3.1"):
    llm = _build_llm(provider, model).bind_tools(_TOOLS)

    def call_model(state: AgentState) -> dict:
        messages = [_SYSTEM_MESSAGE] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(_TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()
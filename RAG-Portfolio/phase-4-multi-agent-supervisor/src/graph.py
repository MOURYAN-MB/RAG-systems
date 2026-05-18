from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.state import SupervisorState
from src.tools import search_ai_papers, web_search
from src.config import OLLAMA_BASE_URL, ANTHROPIC_API_KEY, OPENAI_API_KEY, OLLAMA_MODEL


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


def _make_supervisor_node():
    """
    Deterministic router — reads what's done and decides who goes next.
    No LLM needed here: the logic is simple and must be reliable.
    """
    def supervisor_node(state: SupervisorState) -> dict:
        if not state.get("research_notes"):
            return {
                "next": "research",
                "messages": [AIMessage(
                    content="Task received. Dispatching to Research Agent to gather information.",
                    name="Supervisor",
                )],
            }
        if not state.get("final_report"):
            return {
                "next": "write",
                "messages": [AIMessage(
                    content="Research complete. Dispatching to Writer Agent to produce the report.",
                    name="Supervisor",
                )],
            }
        return {
            "next": "FINISH",
            "messages": [AIMessage(
                content="Report complete. Delivering final output to user.",
                name="Supervisor",
            )],
        }
    return supervisor_node


def _make_research_node(llm):
    """
    Research Agent — directly calls both tools then uses the LLM to
    synthesize findings into structured notes.
    """
    def research_node(state: SupervisorState) -> dict:
        task = state["task"]

        # Call both tools directly (no LLM routing needed — we always want both)
        paper_results = search_ai_papers.invoke({"query": task})

        raw_web = web_search.invoke({"query": task})
        if isinstance(raw_web, list):
            web_text = "\n".join(
                f"[{r.get('title', 'Source')}]\n{r.get('content', '')[:400]}"
                for r in raw_web
            )
        else:
            web_text = str(raw_web)

        # Synthesize both sources into structured research notes
        notes = llm.invoke([
            SystemMessage(content=(
                "You are a research assistant. Synthesize the provided search results "
                "into clear, structured research notes. Include key findings, technical "
                "details, and cite sources."
            )),
            HumanMessage(content=(
                f"Research task: {task}\n\n"
                f"--- AI Research Papers ---\n{paper_results[:3000]}\n\n"
                f"--- Web Search Results ---\n{web_text[:2000]}\n\n"
                "Produce concise, structured research notes."
            )),
        ]).content

        return {
            "research_notes": notes,
            "messages": [AIMessage(
                content=f"Research complete. Gathered findings from AI papers and web search.",
                name="ResearchAgent",
            )],
        }
    return research_node


def _make_writer_node(llm):
    """
    Writer Agent — takes research notes and the original task, produces
    a polished, structured report.
    """
    def writer_node(state: SupervisorState) -> dict:
        task = state["task"]
        notes = state["research_notes"]

        report = llm.invoke([
            SystemMessage(content=(
                "You are a professional technical writer. Write clear, well-structured "
                "reports based on provided research notes. Use markdown formatting."
            )),
            HumanMessage(content=(
                f"Write a comprehensive report for: \"{task}\"\n\n"
                f"Research Notes:\n{notes}\n\n"
                "Format the report with these sections:\n"
                "## Executive Summary\n"
                "## Key Findings\n"
                "## Detailed Analysis\n"
                "## Conclusion\n"
                "## References & Sources"
            )),
        ]).content

        return {
            "final_report": report,
            "messages": [AIMessage(
                content="Report written successfully.",
                name="WriterAgent",
            )],
        }
    return writer_node


def build_graph(provider: str = "ollama", model: str = None):
    if model is None:
        model = OLLAMA_MODEL

    llm = _build_llm(provider, model)

    graph = StateGraph(SupervisorState)

    graph.add_node("supervisor", _make_supervisor_node())
    graph.add_node("research", _make_research_node(llm))
    graph.add_node("write", _make_writer_node(llm))

    graph.set_entry_point("supervisor")

    # Supervisor decides who goes next
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"research": "research", "write": "write", "FINISH": END},
    )

    # After each agent finishes, always return to supervisor
    graph.add_edge("research", "supervisor")
    graph.add_edge("write", "supervisor")

    return graph.compile()
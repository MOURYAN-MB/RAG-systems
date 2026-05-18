# Phase 3 — LangGraph Agentic RAG

> **Goal:** Move from a fixed retrieval pipeline to an agent that *decides* which tool to call — or whether to call any tool at all.

A research assistant where the LLM agent reads the question, decides whether to search AI papers or the live web, calls the appropriate tool, inspects results, and loops until it has a good answer. Every step is visible in real time.

## Architecture

```
User Question
      │
      ▼
 ┌──────────────────────┐
 │   LangGraph Agent    │
 │  (StateGraph loop)   │
 │                      │
 │  ┌─────────────────┐ │
 │  │  LLM decides:   │ │
 │  │  use tool?      │ │
 │  │  which one?     │ │
 │  └────────┬────────┘ │
 │           │          │
 │    ┌──────┴──────┐   │
 │    ▼             ▼   │
 │ search_ai_papers  web_search
 │ (ChromaDB)    (Tavily)
 │    └──────┬──────┘   │
 │           │ results  │
 │    ┌──────▼────────┐ │
 │    │  loop back    │ │
 │    │  or FINISH    │ │
 │    └───────────────┘ │
 └──────────────────────┘
      │
      ▼
 Final Answer + tool trace shown in UI
```

## Key Features

- **Agent decides** — LLM routes to the right tool based on question intent
- **Two tools**: `search_ai_papers` (vector DB) and `web_search` (live Tavily API)
- **Streaming UI** — tool calls, reasoning, and results appear in real time as the agent thinks
- **Multi-step reasoning** — agent loops until it has enough context to answer
- Full transparency: every tool input and output shown in expanders

## What Changed vs Phase 2

| Feature | Phase 2 | Phase 3 |
|---------|---------|---------|
| Retrieval decision | Always retrieves | Agent decides if/what to retrieve |
| Data sources | Static PDF index | Papers + live web |
| Pipeline | Fixed chain | LangGraph state machine |
| Transparency | Final answer only | Every step visible |
| Reasoning | None | Multi-step with tool use |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit (streaming) |
| Orchestration | LangGraph `StateGraph` |
| Tools | ChromaDB search + Tavily web search |
| LLM | Ollama (llama3.1) / Anthropic / OpenAI |
| State | `AgentState` with message history |

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Set OLLAMA_MODEL and TAVILY_API_KEY (free at app.tavily.com)

streamlit run app.py
```

## What This Phase Covers

- LangGraph `StateGraph` — nodes, edges, conditional routing
- Tool calling with LangChain tool decorators
- Agent loop pattern: `agent → tools → agent → ... → END`
- Streaming responses in Streamlit with `st.empty()`
- `AgentState` with append-only message history
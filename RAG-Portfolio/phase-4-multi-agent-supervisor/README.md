# Phase 4 — Multi-Agent Supervisor System

> **Goal:** Decompose a complex task (research a topic, write a report) into specialized agents coordinated by a supervisor.

A three-agent pipeline where a **Supervisor** routes work between a **Research Agent** (gathers information from papers and the web) and a **Writer Agent** (transforms notes into a polished markdown report). Demonstrates task decomposition and the supervisor coordination pattern.

## Architecture

```
User Task: "Write a report on RAG techniques"
      │
      ▼
 ┌─────────────────────────────┐
 │       Supervisor Node       │
 │  (deterministic routing)    │
 │                             │
 │  Step 1 → Research Agent    │
 │  Step 2 → Writer Agent      │
 │  Step 3 → FINISH            │
 └──────────┬──────────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
Research Agent   Writer Agent
 ├ search_papers  Takes research
 ├ web_search     notes → writes
 └ synthesizes    polished report
   findings       in markdown
     │             │
     └──────┬──────┘
            ▼
      Supervisor decides
      next step / FINISH
            │
            ▼
   Final Markdown Report
   (downloadable)
```

## Key Features

- **Supervisor pattern** — deterministic routing (no LLM needed for routing decisions)
- **Specialized agents** — research ≠ writing; each agent is optimized for its task
- **Live workflow log** — watch each agent's actions in real time with step-by-step updates
- **Downloadable report** — final markdown output available for download
- Two-column layout: workflow log on left, final report on right

## What Changed vs Phase 3

| Feature | Phase 3 | Phase 4 |
|---------|---------|---------|
| Agents | 1 (general) | 3 (supervisor + research + writer) |
| Coordination | Agent self-routes | Supervisor routes deterministically |
| Task type | Q&A | Long-form report generation |
| Specialization | None | Research vs writing separation |
| Output | Chat answer | Downloadable markdown report |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit (two-column) |
| Orchestration | LangGraph multi-agent graph |
| Coordination | Supervisor node (deterministic) |
| Tools | `search_ai_papers`, `web_search` (Tavily) |
| LLM | Ollama / Anthropic / OpenAI |
| State | `SupervisorState` (shared across all agents) |

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Set OLLAMA_MODEL and TAVILY_API_KEY

streamlit run app.py
```

## What This Phase Covers

- Multi-agent graph design — multiple specialized nodes sharing state
- Supervisor coordination pattern vs. fully agentic routing
- Task decomposition: breaking complex work into research + synthesis steps
- Shared state management across agents with `SupervisorState`
- Report generation pipeline (research → notes → polished markdown)
"""
Streamlit Analytics Dashboard for the Adaptive RAG Evaluation Engine.

Tab 1 — Query Interface:   Ask questions; see answer + full retrieval breakdown.
Tab 2 — Analytics:         RAGAS score trends, quality alerts, recent query table.
"""
import time
import sys
from pathlib import Path

import streamlit as st
import requests
import pandas as pd
import altair as alt

# ── Project root so src.* imports work when run from repo root ───────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

API_BASE = "http://localhost:8080"

st.set_page_config(
    page_title="Adaptive RAG Dashboard",
    page_icon="🧠",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Adaptive RAG Engine")
    st.caption("Phase 5 — Production RAG with RAGAS evaluation")

    st.subheader("Model")
    provider = st.selectbox("Provider", ["ollama", "google", "anthropic", "openai"])

    model_map = {
        "ollama":    ["llama3.1:latest", "llama3.2:latest", "mistral:latest"],
        "google":    ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"],
        "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
        "openai":    ["gpt-4o-mini", "gpt-4o"],
    }
    model = st.selectbox("Model", model_map[provider])

    st.divider()
    st.subheader("System Health")
    if st.button("Check Health"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=5)
            h = r.json()
            overall = h.get("status", "unknown")
            color = "green" if overall == "healthy" else "orange"
            st.markdown(f"**Overall:** :{color}[{overall}]")
            for svc, status in h.get("services", {}).items():
                icon = "✅" if status == "ok" else "❌"
                st.write(f"{icon} {svc}: {status}")
        except Exception as e:
            st.error(f"API unreachable: {e}")

    st.divider()
    st.subheader("Ingestion")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Rebuild Index"):
            try:
                r = requests.post(f"{API_BASE}/ingest/rebuild", timeout=5)
                st.success(r.json().get("message", "Started"))
            except Exception as e:
                st.error(str(e))
    with col2:
        if st.button("Index Status"):
            try:
                r = requests.get(f"{API_BASE}/ingest/status", timeout=5)
                d = r.json()
                st.info(f"{d['indexed_documents']} docs / {d['total_chunks']} chunks")
            except Exception as e:
                st.error(str(e))


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_query, tab_analytics = st.tabs(["💬 Query Interface", "📊 Analytics"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — QUERY INTERFACE
# ════════════════════════════════════════════════════════════════════════════
with tab_query:
    st.header("Ask a Question")

    question = st.text_input(
        "Question",
        placeholder="e.g. What are the key findings in the uploaded documents?",
    )

    if model and "llama3.1" in model:
        st.info("llama3.1:latest is a large model — responses can take 5-10 minutes locally. "
                "For faster results, switch to **llama3.2** in the sidebar (if pulled).", icon="⏱️")

    if st.button("Submit", type="primary") and question.strip():
        with st.spinner("Retrieving and generating answer… (llama3.1 may take several minutes)"):
            t0 = time.time()
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"question": question, "provider": provider, "model": model},
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        elapsed = time.time() - t0

        # ── Answer ────────────────────────────────────────────────────────
        st.subheader("Answer")
        st.write(data["answer"])
        st.caption(
            f"Latency: {data['latency_ms']} ms  |  "
            f"RAGAS evaluation: {'scheduled ✓' if data['eval_scheduled'] else 'skipped'}"
        )

        # ── Retrieved Sources ──────────────────────────────────────────────
        st.subheader("Retrieved Sources")
        sources = data.get("sources", [])
        if sources:
            for i, src in enumerate(sources, 1):
                with st.expander(
                    f"#{i}  {src['source']}  (page {src['page']})  "
                    f"— rerank score: {src['rerank_score']:.4f}" if src['rerank_score'] else
                    f"#{i}  {src['source']}  (page {src['page']})"
                ):
                    st.write(src["text_preview"])
        else:
            st.info("No sources returned.")

        st.caption(
            f"Model: {provider}/{model}  |  "
            f"Retrieval: hybrid (BM25 + vector + RRF + cross-encoder)"
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.header("Quality Analytics")

    # ── Live quality status ────────────────────────────────────────────────
    try:
        qr = requests.get(f"{API_BASE}/quality", timeout=5)
        qs = qr.json()
    except Exception:
        qs = {"status": "api_unreachable"}

    status_label = qs.get("status", "unknown")
    if status_label == "healthy":
        st.success("System status: HEALTHY")
    elif status_label == "alert":
        st.error("System status: QUALITY ALERT")
    elif status_label == "no_data":
        st.info("System status: No evaluation data yet — run some queries first.")
    else:
        st.warning(f"System status: {status_label}")

    # ── Score gauges ───────────────────────────────────────────────────────
    if status_label not in ("no_data", "api_unreachable"):
        c1, c2, c3 = st.columns(3)
        faith = qs.get('avg_faithfulness')
        recall = qs.get('avg_context_recall')
        c1.metric("Faithfulness",     f"{faith:.2f}"  if faith   is not None else "N/A", help="Target ≥ 0.70")
        c2.metric("Answer Relevancy", f"{qs.get('avg_answer_relevancy') or 0:.2f}")
        c3.metric("Context Recall",   f"{recall:.2f}" if recall  is not None else "N/A", help="Target ≥ 0.60")

        n = qs.get("n", 0)
        window = qs.get("window", 50)
        if n:
            st.caption(f"Sliding window: {n} of last {window} evaluations")

    st.divider()

    # ── Score history from database ────────────────────────────────────────
    st.subheader("Score Trends")
    try:
        from src.database import SessionLocal, Evaluation
        db = SessionLocal()
        try:
            rows = (
                db.query(
                    Evaluation.created_at,
                    Evaluation.faithfulness,
                    Evaluation.answer_relevancy,
                    Evaluation.context_recall,
                )
                .order_by(Evaluation.created_at.desc())
                .limit(100)
                .all()
            )
        finally:
            db.close()

        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "faithfulness", "answer_relevancy", "context_recall"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            # Melt to long format for Altair
            df_long = df.melt(
                id_vars="timestamp",
                value_vars=["faithfulness", "answer_relevancy", "context_recall"],
                var_name="metric",
                value_name="score",
            ).dropna(subset=["score"])

            color_scale = alt.Scale(
                domain=["faithfulness", "answer_relevancy", "context_recall"],
                range=["#E05C5C", "#4C9EE8", "#2ECC71"],  # red, blue, green
            )

            # Clamp scores to [0, 1] — RAGAS occasionally returns slightly above 1.0
            df_long["score"] = df_long["score"].clip(0, 1)

            lines = (
                alt.Chart(df_long)
                .mark_line(point=alt.OverlayMarkDef(size=60))
                .encode(
                    x=alt.X("timestamp:T", title="Time", axis=alt.Axis(format="%b %d %H:%M")),
                    y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1.0], clamp=True)),
                    color=alt.Color("metric:N", scale=color_scale, legend=alt.Legend(title="Metric")),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Time", format="%b %d %H:%M"),
                        alt.Tooltip("metric:N", title="Metric"),
                        alt.Tooltip("score:Q", title="Score", format=".3f"),
                    ],
                )
            )

            # Dashed threshold reference lines with text labels
            thresholds = pd.DataFrame([
                {"metric": "faithfulness",   "threshold": 0.70, "label": "target ≥ 0.70"},
                {"metric": "context_recall", "threshold": 0.60, "label": "target ≥ 0.60"},
            ])
            thresh_lines = (
                alt.Chart(thresholds)
                .mark_rule(strokeDash=[6, 4], opacity=0.6, strokeWidth=1.5)
                .encode(
                    y=alt.Y("threshold:Q"),
                    color=alt.Color("metric:N", scale=color_scale, legend=None),
                    tooltip=[alt.Tooltip("label:N", title="")],
                )
            )
            thresh_labels = (
                alt.Chart(thresholds)
                .mark_text(align="left", dx=4, dy=-6, fontSize=11, opacity=0.7)
                .encode(
                    x=alt.value(0),
                    y=alt.Y("threshold:Q"),
                    text=alt.Text("label:N"),
                    color=alt.Color("metric:N", scale=color_scale, legend=None),
                )
            )

            st.altair_chart(
                (lines + thresh_lines + thresh_labels).properties(height=320).interactive(),
                use_container_width=True,
            )
        else:
            st.info("No evaluation history yet.")
    except Exception as e:
        st.warning(f"Could not load score history: {e}")

    st.divider()

    # ── Recent queries table ───────────────────────────────────────────────
    st.subheader("Recent Queries")
    try:
        from src.database import SessionLocal, Evaluation
        db = SessionLocal()
        try:
            recent = (
                db.query(
                    Evaluation.created_at,
                    Evaluation.query,
                    Evaluation.faithfulness,
                    Evaluation.answer_relevancy,
                    Evaluation.context_recall,
                    Evaluation.latency_ms,
                    Evaluation.model,
                )
                .order_by(Evaluation.created_at.desc())
                .limit(20)
                .all()
            )
        finally:
            db.close()

        if recent:
            df2 = pd.DataFrame(
                recent,
                columns=["Time", "Query", "Faithfulness", "Answer Relevancy", "Context Recall", "Latency (ms)", "Model"],
            )
            df2["Query"] = df2["Query"].str[:80] + "…"
            st.dataframe(df2, use_container_width=True)
        else:
            st.info("No queries logged yet.")
    except Exception as e:
        st.warning(f"Could not load recent queries: {e}")

    st.divider()

    # ── Open alerts ────────────────────────────────────────────────────────
    st.subheader("Quality Alerts")
    try:
        from src.monitor import get_open_alerts
        alerts = get_open_alerts()
        if alerts:
            for a in alerts:
                st.error(
                    f"**Alert #{a['id']}** — {a['reason']}  \n"
                    f"Faithfulness: {a['faithfulness']:.2f}  |  "
                    f"Context Recall: {a['context_recall']:.2f}  |  "
                    f"Opened: {a['created_at']}"
                )
        else:
            st.success("No open alerts.")
    except Exception as e:
        st.warning(f"Could not load alerts: {e}")

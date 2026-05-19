"""
Airflow DAG: quality_monitor_reindex

Runs every 30 minutes:
1. Call GET /quality on the FastAPI service to read sliding-window RAGAS scores.
2. If status == "alert", call POST /ingest/rebuild to trigger re-indexing.
3. Log outcome to the Airflow task log.

Uses the FastAPI REST API (no direct src/ imports) so the DAG has no Python
dependency on the application code or its venv.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

_API_URL = os.getenv("RAG_API_URL", "http://host.docker.internal:8080")

default_args = {
    "owner": "adaptive-rag",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_quality(**ctx):
    """
    Fetch quality status from the RAG API and push to XCom.
    Fails the task (raises) if the API is unreachable.
    """
    import requests
    r = requests.get(f"{_API_URL}/quality", timeout=15)
    r.raise_for_status()
    status = r.json()
    print(f"[quality-check] {status}")
    ctx["ti"].xcom_push(key="quality_status", value=status)


def maybe_reindex(**ctx):
    """
    Trigger re-indexing via POST /ingest/rebuild only when quality is in alert.
    After re-indexing, log the updated quality status.
    """
    import requests
    status = ctx["ti"].xcom_pull(key="quality_status", task_ids="check_quality")

    if status is None or status.get("status") == "no_data":
        print("[reindex] No quality data yet — skipping.")
        return

    if status.get("status") != "alert":
        faith  = status.get("avg_faithfulness",  "N/A")
        recall = status.get("avg_context_recall", "N/A")
        print(f"[reindex] Quality healthy (faithfulness={faith}, context_recall={recall}) — no action.")
        return

    faith  = status.get("avg_faithfulness",  0)
    recall = status.get("avg_context_recall", 0)
    print(
        f"[reindex] QUALITY ALERT — faithfulness={faith}, context_recall={recall}. "
        f"Triggering re-index..."
    )
    r = requests.post(f"{_API_URL}/ingest/rebuild", timeout=300)
    r.raise_for_status()
    result = r.json()
    print(f"[reindex] Re-index triggered: {result}")

    # Confirm recovery
    r2 = requests.get(f"{_API_URL}/quality", timeout=15)
    if r2.ok:
        post = r2.json()
        print(
            f"[reindex] Post-reindex status: {post.get('status')} — "
            f"faithfulness={post.get('avg_faithfulness')}, "
            f"context_recall={post.get('avg_context_recall')}"
        )


with DAG(
    dag_id="quality_monitor_reindex",
    description="Check RAGAS quality via API; re-index if alert threshold breached",
    start_date=datetime(2024, 1, 1),
    schedule_interval="*/30 * * * *",
    catchup=False,
    default_args=default_args,
    tags=["adaptive-rag", "quality"],
) as dag:

    t_check = PythonOperator(
        task_id="check_quality",
        python_callable=check_quality,
    )

    t_reindex = PythonOperator(
        task_id="maybe_reindex",
        python_callable=maybe_reindex,
    )

    t_check >> t_reindex

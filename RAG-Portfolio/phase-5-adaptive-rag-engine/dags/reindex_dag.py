"""
Airflow DAG: quality_monitor_reindex

Runs every 30 minutes:
1. Compute sliding-window RAGAS scores from PostgreSQL.
2. If faithfulness < FAITHFULNESS_THRESHOLD or context_recall < CONTEXT_RECALL_THRESHOLD,
   trigger a full re-index of all documents.
3. Log outcome to the Airflow task log.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Project root on PYTHONPATH so src.* imports work inside Airflow ──────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

default_args = {
    "owner": "adaptive-rag",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_quality(**ctx):
    """Return quality status dict; pushed to XCom for the next task."""
    from src.monitor import compute_quality_status
    status = compute_quality_status()
    print(f"[quality-check] status={status}")
    ctx["ti"].xcom_push(key="quality_status", value=status)


def maybe_reindex(**ctx):
    """Re-index only when quality is below threshold."""
    from src.config import FAITHFULNESS_THRESHOLD, CONTEXT_RECALL_THRESHOLD
    from src.monitor import resolve_alerts
    from src.ingestion import ingest_documents

    status = ctx["ti"].xcom_pull(key="quality_status", task_ids="check_quality")

    if status is None or status.get("status") == "no_data":
        print("[reindex] No quality data yet — skipping.")
        return

    faith  = status.get("avg_faithfulness",   1.0)
    recall = status.get("avg_context_recall",  1.0)
    alert  = status.get("status") == "alert"

    if alert or faith < FAITHFULNESS_THRESHOLD or recall < CONTEXT_RECALL_THRESHOLD:
        print(
            f"[reindex] Quality below threshold "
            f"(faithfulness={faith:.2f}, context_recall={recall:.2f}) — re-indexing."
        )
        ingest_documents()
        resolve_alerts()
        print("[reindex] Re-indexing complete. Alerts resolved.")
    else:
        print(
            f"[reindex] Quality healthy "
            f"(faithfulness={faith:.2f}, context_recall={recall:.2f}) — no action."
        )


with DAG(
    dag_id="quality_monitor_reindex",
    description="Check RAGAS quality scores; re-index if below threshold",
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
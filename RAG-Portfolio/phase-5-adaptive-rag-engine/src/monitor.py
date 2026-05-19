"""
Quality monitor — computes sliding window averages over recent evaluations
and writes alerts when scores drop below configured thresholds.
"""
from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from src.config import (
    FAITHFULNESS_THRESHOLD, CONTEXT_RECALL_THRESHOLD, QUALITY_WINDOW
)
from src.database import Evaluation, Alert, SessionLocal


def compute_quality_status(window: int = QUALITY_WINDOW) -> dict:
    """
    Read the last `window` evaluations and return average scores + alert status.
    """
    db: Session = SessionLocal()
    try:
        rows = (
            db.query(Evaluation)
            .filter(Evaluation.answer_relevancy.isnot(None))
            .order_by(desc(Evaluation.created_at))
            .limit(window)
            .all()
        )

        if not rows:
            return {"status": "no_data", "n": 0}

        rows_with_faith   = [r for r in rows if r.faithfulness   is not None]
        rows_with_recall  = [r for r in rows if r.context_recall is not None]

        avg_faithfulness = (
            sum(r.faithfulness for r in rows_with_faith) / len(rows_with_faith)
            if rows_with_faith else None
        )
        avg_context_recall = (
            sum(r.context_recall for r in rows_with_recall) / len(rows_with_recall)
            if rows_with_recall else None
        )
        avg_answer_rel = sum(r.answer_relevancy or 0 for r in rows) / len(rows)

        faith_alert = (
            avg_faithfulness < FAITHFULNESS_THRESHOLD
            if avg_faithfulness is not None else False
        )
        recall_alert = (
            avg_context_recall < CONTEXT_RECALL_THRESHOLD
            if avg_context_recall is not None else False
        )

        return {
            "status":               "alert" if (faith_alert or recall_alert) else "healthy",
            "avg_faithfulness":     round(avg_faithfulness,    3) if avg_faithfulness    is not None else None,
            "avg_answer_relevancy": round(avg_answer_rel,      3),
            "avg_context_recall":   round(avg_context_recall,  3) if avg_context_recall is not None else None,
            "faithfulness_alert":   faith_alert,
            "recall_alert":         recall_alert,
            "n":                    len(rows),
            "window":               window,
        }
    finally:
        db.close()


def check_and_create_alert(scores: dict) -> bool:
    """
    After each evaluation, check if latest scores crossed a threshold.
    If so, write an alert record. Returns True if alert was created.
    """
    db: Session = SessionLocal()
    created = False
    try:
        if scores.get("faithfulness") is not None:
            if scores["faithfulness"] < FAITHFULNESS_THRESHOLD:
                db.add(Alert(
                    metric="faithfulness",
                    metric_value=scores["faithfulness"],
                    threshold=FAITHFULNESS_THRESHOLD,
                ))
                created = True

        if scores.get("context_recall") is not None:
            if scores["context_recall"] < CONTEXT_RECALL_THRESHOLD:
                db.add(Alert(
                    metric="context_recall",
                    metric_value=scores["context_recall"],
                    threshold=CONTEXT_RECALL_THRESHOLD,
                ))
                created = True

        if created:
            db.commit()
    finally:
        db.close()
    return created


def get_open_alerts() -> list[dict]:
    db: Session = SessionLocal()
    try:
        alerts = (
            db.query(Alert)
            .filter(Alert.status == "open")
            .order_by(desc(Alert.triggered_at))
            .limit(20)
            .all()
        )
        return [
            {
                "id":           a.id,
                "metric":       a.metric,
                "value":        round(a.metric_value, 3),
                "threshold":    a.threshold,
                "triggered_at": str(a.triggered_at),
            }
            for a in alerts
        ]
    finally:
        db.close()


def resolve_alerts():
    """Mark all open alerts as resolved (called after successful re-indexing)."""
    db: Session = SessionLocal()
    try:
        from datetime import datetime, timezone
        db.query(Alert).filter(Alert.status == "open").update(
            {"status": "resolved", "resolved_at": datetime.now(timezone.utc)}
        )
        db.commit()
    finally:
        db.close()

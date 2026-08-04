"""Durable learning and mistake memory for the autonomous organism.

The trading process is intentionally treated as an experiment loop rather than
as an unconstrained self-modifying program.  Predictions, fills, feedback and
mistakes are recorded as append-only JSON lines.  This gives every module a
small, inspectable memory that survives restarts without adding a database
runtime dependency.

The store is deliberately boring:

* malformed historical lines are ignored instead of preventing startup;
* writes are flushed and fsynced before the call returns;
* records contain no credentials or source code;
* statistics are descriptive and are never used to bypass risk controls.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_RECORD_TYPES = {"prediction", "outcome", "mistake", "feedback", "regime"}


def _json_safe(value: Any) -> Any:
    """Convert common runtime values into JSON-safe, bounded values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in list(value.items())[:100]}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in list(value)[:100]]
    if hasattr(value, "value"):
        return _json_safe(value.value)
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    return str(value)


class LearningStore:
    """Append-only prediction/outcome memory with query helpers.

    Parameters
    ----------
    path:
        JSONL path.  If omitted, ``QTS_LEARNING_PATH`` or the repository's
        ``data/autonomy/learning.jsonl`` is used.
    max_records:
        Maximum records retained in memory.  The file remains append-only;
        compaction is intentionally explicit so a deployment can archive it.
    clock:
        Injectable clock for deterministic tests.
    """

    def __init__(
        self,
        path: Optional[str | os.PathLike[str]] = None,
        *,
        max_records: int = 50_000,
        clock=time.time,
    ) -> None:
        default_path = (
            Path(__file__).resolve().parents[1] / "data" / "autonomy" / "learning.jsonl"
        )
        configured = path or os.environ.get("QTS_LEARNING_PATH") or default_path
        self.path = Path(configured).expanduser().resolve()
        self.max_records = max(1, int(max_records))
        self.clock = clock
        self._lock = threading.RLock()
        self.persistence_available = True
        self._records: List[Dict[str, Any]] = []
        self._prediction_index: Dict[str, Dict[str, Any]] = {}
        self._outcome_index: set[str] = set()
        self._load()

    # ------------------------------------------------------------------ IO
    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if (
                        not isinstance(record, dict)
                        or record.get("type") not in _RECORD_TYPES
                    ):
                        continue
                    self._index_record(record)
        except OSError:
            # A read-only or temporarily unavailable learning file must not
            # prevent the signal path from starting.  The error is represented
            # by an empty in-memory store and can be inspected by the caller.
            self.persistence_available = False
            return

        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]

    def _index_record(self, record: Dict[str, Any]) -> None:
        self._records.append(record)
        record_type = record.get("type")
        if record_type == "prediction" and record.get("prediction_id"):
            self._prediction_index[str(record["prediction_id"])] = record
        elif record_type == "outcome" and record.get("prediction_id"):
            self._outcome_index.add(str(record["prediction_id"]))

    def _append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record = _json_safe(record)
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    )
                    stream.flush()
                    try:
                        os.fsync(stream.fileno())
                    except OSError:
                        # Some in-memory filesystems do not expose fsync.
                        pass
            except OSError:
                # Learning is useful but must never take down market analysis.
                # Keep the record in memory so the current process still
                # learns; callers can inspect ``persistence_available``.
                self.persistence_available = False
            self._index_record(record)
            if len(self._records) > self.max_records:
                self._records = self._records[-self.max_records :]
        return record

    # -------------------------------------------------------------- recording
    def record_prediction(
        self,
        *,
        module_name: str,
        symbol: str,
        signal: Optional[str],
        confidence: float,
        regime: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        prediction_id: Optional[str] = None,
    ) -> str:
        """Record one module prediction and return its correlation id."""

        prediction_id = prediction_id or uuid.uuid4().hex
        record = {
            "type": "prediction",
            "prediction_id": prediction_id,
            "timestamp": float(self.clock()),
            "module_name": str(module_name),
            "symbol": str(symbol),
            "signal": str(signal).upper() if signal is not None else None,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "regime": regime or "unknown",
            "context": context or {},
        }
        self._append(record)
        return prediction_id

    def record_outcome(
        self,
        *,
        prediction_id: Optional[str] = None,
        module_name: Optional[str] = None,
        symbol: Optional[str] = None,
        pnl: Optional[float] = None,
        reward: Optional[float] = None,
        correct: Optional[bool] = None,
        reason: str = "",
        regime: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a realized outcome and create a mistake lesson if needed.

        ``record_outcome`` is idempotent for a prediction id.  Exchange
        adapters can therefore emit both a fill event and a reconciled fill
        without double-counting the trade.
        """

        prediction = (
            self._prediction_index.get(str(prediction_id)) if prediction_id else None
        )
        if prediction and str(prediction_id) in self._outcome_index:
            existing = [
                item
                for item in self._records
                if item.get("type") == "outcome"
                and item.get("prediction_id") == str(prediction_id)
            ]
            return (
                dict(existing[-1])
                if existing
                else {"prediction_id": prediction_id, "duplicate": True}
            )

        if module_name is None and prediction:
            module_name = prediction.get("module_name")
        if symbol is None and prediction:
            symbol = prediction.get("symbol")
        if regime is None and prediction:
            regime = prediction.get("regime")

        if pnl is not None:
            pnl = float(pnl)
        if reward is None:
            if correct is not None:
                reward = 1.0 if correct else 0.0
            elif pnl is not None:
                reward = 1.0 if pnl > 0 else 0.0
            else:
                reward = 0.5
        reward = max(0.0, min(1.0, float(reward)))
        if correct is None and pnl is not None:
            correct = pnl > 0

        outcome = self._append(
            {
                "type": "outcome",
                "prediction_id": prediction_id,
                "timestamp": float(self.clock()),
                "module_name": module_name or "unknown",
                "symbol": symbol or "unknown",
                "pnl": pnl,
                "reward": reward,
                "correct": correct,
                "reason": reason,
                "regime": regime or "unknown",
                "metadata": metadata or {},
            }
        )

        needs_lesson = (
            (correct is False) or (pnl is not None and pnl < 0) or reward < 0.5
        )
        if needs_lesson:
            lesson = (
                reason.strip()
                or "Negative realized outcome; reduce confidence or revisit assumptions."
            )
            self._append(
                {
                    "type": "mistake",
                    "mistake_id": uuid.uuid4().hex,
                    "timestamp": float(self.clock()),
                    "prediction_id": prediction_id,
                    "module_name": module_name or "unknown",
                    "symbol": symbol or "unknown",
                    "regime": regime or "unknown",
                    "lesson": lesson,
                    "pnl": pnl,
                }
            )
        return outcome

    def record_mistake(
        self,
        *,
        module_name: str,
        lesson: str,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        prediction_id: Optional[str] = None,
        pnl: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Store an explicit operational lesson for future coding cycles."""
        return self._append(
            {
                "type": "mistake",
                "mistake_id": uuid.uuid4().hex,
                "timestamp": float(self.clock()),
                "prediction_id": prediction_id,
                "module_name": module_name,
                "symbol": symbol or "unknown",
                "regime": regime or "unknown",
                "lesson": str(lesson),
                "pnl": pnl,
            }
        )

    def record_feedback(
        self,
        *,
        module_name: str,
        reward: float,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        reason: str = "manual feedback",
    ) -> Dict[str, Any]:
        """Record feedback when an execution adapter has no prediction id."""

        return self._append(
            {
                "type": "feedback",
                "feedback_id": uuid.uuid4().hex,
                "timestamp": float(self.clock()),
                "module_name": module_name,
                "symbol": symbol or "unknown",
                "regime": regime or "unknown",
                "reward": max(0.0, min(1.0, float(reward))),
                "reason": reason,
            }
        )

    def record_regime(
        self, regime: str, metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self._append(
            {
                "type": "regime",
                "timestamp": float(self.clock()),
                "regime": regime,
                "metrics": metrics or {},
            }
        )

    # --------------------------------------------------------------- queries
    def records(
        self, record_type: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        with self._lock:
            items = [
                r
                for r in self._records
                if record_type is None or r.get("type") == record_type
            ]
            if limit is not None:
                items = items[-int(limit) :]
            return [dict(item) for item in items]

    def mistakes(
        self, module_name: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        items = self.records("mistake")
        if module_name:
            items = [item for item in items if item.get("module_name") == module_name]
        return items[-limit:]

    # Friendly alias for callers that use a noun.
    lessons = mistakes

    def module_stats(
        self,
        module_name: Optional[str] = None,
        *,
        regime: Optional[str] = None,
        window: int = 100,
    ) -> Dict[str, Any]:
        """Return bounded descriptive statistics for a module or the system."""

        outcomes = self.records("outcome")
        feedback = self.records("feedback")
        if module_name:
            outcomes = [r for r in outcomes if r.get("module_name") == module_name]
            feedback = [r for r in feedback if r.get("module_name") == module_name]
        if regime:
            outcomes = [r for r in outcomes if r.get("regime") == regime]
            feedback = [r for r in feedback if r.get("regime") == regime]
        outcomes = outcomes[-window:]
        feedback = feedback[-window:]

        rewards: List[float] = [float(r.get("reward", 0.5)) for r in outcomes]
        rewards.extend(float(r.get("reward", 0.5)) for r in feedback)
        pnls = [
            float(r["pnl"]) for r in outcomes if isinstance(r.get("pnl"), (int, float))
        ]
        wins = sum(
            1
            for r in outcomes
            if r.get("correct") is True
            or (r.get("pnl") is not None and r.get("pnl", 0) > 0)
        )
        losses = sum(
            1
            for r in outcomes
            if r.get("correct") is False
            or (r.get("pnl") is not None and r.get("pnl", 0) < 0)
        )
        mistakes = self.mistakes(module_name=module_name, limit=window)
        if regime:
            mistakes = [r for r in mistakes if r.get("regime") == regime]

        sample_count = len(outcomes) + len(feedback)
        return {
            "module_name": module_name,
            "regime": regime,
            "sample_count": sample_count,
            "outcomes": len(outcomes),
            "feedback_records": len(feedback),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(outcomes) if outcomes else 0.0,
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.5,
            "mistakes": len(mistakes),
            "mistake_rate": len(mistakes) / max(sample_count, 1),
            "last_timestamp": max(
                [float(r.get("timestamp", 0.0)) for r in outcomes + feedback],
                default=0.0,
            ),
        }

    # Query aliases used by strategy integrations.
    get_module_stats = module_stats
    get_mistakes = mistakes

    def summary(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "records": len(self._records),
            "predictions": len(self.records("prediction")),
            "outcomes": len(self.records("outcome")),
            "mistakes": len(self.records("mistake")),
            "feedback": len(self.records("feedback")),
            "persistence_available": getattr(self, "persistence_available", True),
        }

    def compact(self, keep: Optional[int] = None) -> None:
        """Compact the JSONL file explicitly while preserving recent records."""

        keep = int(keep or self.max_records)
        with self._lock:
            records = self._records[-keep:]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(
                        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    )
                stream.flush()
                try:
                    os.fsync(stream.fileno())
                except OSError:
                    pass
            temporary.replace(self.path)
            self._records = records


# Backwards/semantic aliases used by integrations.
MistakeMemory = LearningStore
OutcomeMemory = LearningStore

__all__ = ["LearningStore", "MistakeMemory", "OutcomeMemory"]

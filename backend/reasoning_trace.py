"""Per-request execution trace.

Serves two consumers:
1. The UI reasoning panel (export() — unchanged shape).
2. Production observability: every request appends ONE JSON line to
   data/traces.jsonl via persist(), carrying a request id, per-stage
   latency breakdown, and a summary of the request outcome. Grep-able,
   pandas-loadable, and the substrate for offline error analysis.

Timing uses a monotonic clock; elapsed_ms on each step is the time since
the previous step was recorded (i.e., roughly the cost of the stage that
just finished), so the JSONL doubles as a latency breakdown.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_WRITE_LOCK = threading.Lock()


@dataclass
class TraceStep:

    sequence: int

    stage: str

    title: str

    status: str

    summary: str

    elapsed_ms: float = 0.0

    details: dict = field(default_factory=dict)


def _jsonable(value, _depth=0, _seen=None):
    """Best-effort conversion so persist() never fails: handles sets and
    arbitrary objects, and guards against cyclic / pathologically deep
    structures (stage details may embed graph nodes that reference each
    other) with a visited-set and a depth cap."""
    if _depth > 20:
        return "<max depth>"
    if _seen is None:
        _seen = set()
    if isinstance(value, dict):
        if id(value) in _seen:
            return "<cycle>"
        _seen.add(id(value))
        try:
            return {
                str(k): _jsonable(v, _depth + 1, _seen)
                for k, v in value.items()
            }
        finally:
            _seen.discard(id(value))
    if isinstance(value, (list, tuple, set, frozenset)):
        if id(value) in _seen:
            return "<cycle>"
        _seen.add(id(value))
        try:
            return [_jsonable(v, _depth + 1, _seen) for v in value]
        finally:
            _seen.discard(id(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ReasoningTrace:

    def __init__(self):
        self.request_id = uuid.uuid4().hex[:12]
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._t0 = time.monotonic()
        self._last = self._t0
        self.steps: list[TraceStep] = []
        self.sequence = 0

    def add(
        self,
        stage,
        title,
        summary,
        status="success",
        **details,
    ):
        now = time.monotonic()
        self.sequence += 1
        self.steps.append(
            TraceStep(
                sequence=self.sequence,
                stage=stage,
                title=title,
                status=status,
                summary=summary,
                elapsed_ms=round((now - self._last) * 1000, 1),
                details=details,
            )
        )
        self._last = now

    def export(self):
        return [
            {
                "sequence": s.sequence,
                "stage": s.stage,
                "title": s.title,
                "status": s.status,
                "summary": s.summary,
                "elapsed_ms": s.elapsed_ms,
                "details": s.details,
            }
            for s in self.steps
        ]

    # ------------------------------------------------------------ persist

    def record(self, **summary) -> dict:
        """The full structured trace record for this request."""
        return _jsonable({
            "request_id": self.request_id,
            "started_at": self.started_at,
            "total_ms": round((time.monotonic() - self._t0) * 1000, 1),
            "latency_breakdown_ms": {
                s.stage: s.elapsed_ms for s in self.steps
            },
            "stages": self.export(),
            **summary,
        })

    def persist(self, path: str | Path, **summary) -> dict:
        """Append this request's trace as one JSON line. Never raises —
        observability must not take down the request path."""
        rec = self.record(**summary)
        try:
            line = json.dumps(rec, ensure_ascii=False)
            with _WRITE_LOCK:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as exc:  # pragma: no cover
            print(f"trace persist failed (non-fatal): {exc}")
        return rec

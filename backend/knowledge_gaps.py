"""AXON knowledge-gap store (design §8 / §10).

Every question AXON cannot ground (refused) or can only weakly ground
(escalated) is captured here as a durable knowledge gap — a prioritised
capture workflow: each gap carries a priority, occurrence count, a suggested
subject-matter expert to capture it from, an assignable owner, and a status
(open → assigned → captured / dismissed). Persisted to JSON so the workflow
survives restarts.
"""
from __future__ import annotations

import csv
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

_PRIORITY_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
_ACTIVE = {"open", "assigned"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_experts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["topic_list"] = [t.strip().lower() for t in r.get("topics", "").split(",") if t.strip()]
    return rows


def suggest_sme(query: str, experts: list[dict]) -> dict | None:
    """Recommend the SME whose topics best overlap the unanswered question —
    i.e. the person to capture this knowledge from before it walks out the door."""
    ql = query.lower()
    best, best_score = None, 0
    for e in experts:
        score = sum(1 for t in e["topic_list"] if t in ql)
        if score > best_score:
            best, best_score = e, score
    if not best:
        return None
    return {"name": best["name"], "role": best["role"], "area": best.get("area", "")}


class GapStore:
    def __init__(self, path: Path):
        self.path = path
        self.gaps: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.gaps = json.loads(self.path.read_text())
            except Exception:
                self.gaps = {}

    def _save(self):
        self.path.write_text(json.dumps(self.gaps, indent=2))

    @staticmethod
    def _key(query: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", query.lower()).strip()

    def _find(self, gap_id: str) -> dict | None:
        return next((g for g in self.gaps.values() if g["id"] == gap_id), None)

    def record(self, query: str, reason: str, confidence: float, priority: str,
               suggested_sme: dict | None = None) -> dict:
        """Log (or re-log) a knowledge gap. Repeated questions raise priority —
        a topic asked again and again with no answer is the most urgent to capture."""
        key = self._key(query)
        if not key:
            return {}
        g = self.gaps.get(key)
        if g and g.get("status") in _ACTIVE:
            g["ask_count"] += 1
            g["last_asked"] = _now()
            g["confidence"] = confidence
            if g["ask_count"] >= 3 and g["priority"] == "MEDIUM":
                g["priority"] = "HIGH"
        else:
            g = {"id": uuid.uuid4().hex[:8], "query": query, "reason": reason,
                 "confidence": confidence, "priority": priority, "ask_count": 1,
                 "first_asked": _now(), "last_asked": _now(),
                 "status": "open", "owner": None, "suggested_sme": suggested_sme}
            self.gaps[key] = g
        self._save()
        return g

    def list(self, include_closed: bool = False) -> list[dict]:
        items = [g for g in self.gaps.values()
                 if include_closed or g.get("status") in _ACTIVE]
        items.sort(key=lambda g: (_PRIORITY_RANK.get(g["priority"], 9), -g["ask_count"]))
        return items

    def unresolved_count(self) -> int:
        return sum(1 for g in self.gaps.values() if g.get("status") in _ACTIVE)

    def assign(self, gap_id: str, owner: str) -> dict | None:
        g = self._find(gap_id)
        if not g:
            return None
        g["owner"] = owner
        g["status"] = "assigned"
        g["assigned_at"] = _now()
        self._save()
        return g

    def set_status(self, gap_id: str, status: str) -> dict | None:
        if status not in ("open", "assigned", "captured", "dismissed"):
            return None
        g = self._find(gap_id)
        if not g:
            return None
        g["status"] = status
        if status in ("captured", "dismissed"):
            g["resolved_at"] = _now()
        self._save()
        return g

    # Backward-compatible: resolve == mark captured.
    def resolve(self, gap_id: str) -> bool:
        return self.set_status(gap_id, "captured") is not None

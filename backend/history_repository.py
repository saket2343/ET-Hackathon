"""Persistence for the maintenance_events model.

This is the new system of record for Asset360 maintenance history. Uploaded
maintenance documents are extracted into structured events and committed here,
which is what finally lets a PDF update Asset360 (the design goal).

Storage strategy
----------------
In production this class fronts a real database. When a database is
unavailable (the demo default) it degrades to a structured JSON store at
``data/maintenance_events.json`` — the proper ``maintenance_events`` model the
design doc asks for, NOT a bare CSV.

For backward compatibility every committed event is ALSO mirrored as a row in
the legacy ``data/maintenance_log.csv`` (same ten columns as the seeded file),
so any code path that still reads the CSV keeps working unchanged.
``ingest.load_corpus`` overlays the JSON events on top of the CSV rows keyed by
work order, so there is never any double counting.

The public API (``add`` / ``all`` / ``get``) is identical regardless of the
backend, so swapping the JSON fallback for a database is a one-file change.
"""
from __future__ import annotations

import csv
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Canonical field order of the maintenance_events model.
EVENT_FIELDS = (
    "event_id", "asset_id", "event_type", "date", "work_order",
    "failure_mode", "severity", "root_cause", "corrective_action",
    "preventive_action", "parts_used", "downtime_hours", "engineer", "cost",
    "source_document", "page_number", "confidence", "created_at", "updated_at",
)

# The frozen ten-column contract of data/maintenance_log.csv.
CSV_COLUMNS = (
    "wo_number", "date", "equipment", "type", "failure_mode", "symptom",
    "cause", "action", "parts_used", "downtime_hours",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", (text or "").strip()).strip("-")


def normalize_event(raw: dict) -> dict:
    """Coerce an arbitrary extracted/edited dict into a well-formed event
    matching the maintenance_events model. Missing keys default sensibly;
    parts_used is always a list; numeric fields are best-effort cast."""
    ev: dict = {k: raw.get(k, "") for k in EVENT_FIELDS}

    parts = raw.get("parts_used", [])
    if isinstance(parts, str):
        parts = [p.strip() for p in re.split(r"[,;]", parts) if p.strip()]
    ev["parts_used"] = [str(p).strip() for p in (parts or []) if str(p).strip()]

    try:
        ev["downtime_hours"] = float(raw.get("downtime_hours") or 0)
    except (TypeError, ValueError):
        ev["downtime_hours"] = 0.0

    conf = raw.get("confidence", 0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    ev["confidence"] = conf if conf <= 1.0 else round(conf / 100.0, 4)

    for key in ("asset_id", "event_type", "date", "work_order", "failure_mode",
                "severity", "root_cause", "corrective_action",
                "preventive_action", "engineer", "cost", "source_document"):
        ev[key] = str(raw.get(key, "") or "").strip()

    ev["page_number"] = str(raw.get("page_number", "") or "").strip()
    return ev


def to_maintenance_row(ev: dict) -> dict:
    """Project an event onto the legacy maintenance-log row shape used by
    Asset360 and the knowledge graph, carrying the richer event fields along
    as extra keys (source_document, page_number, confidence, ...) so source
    traceability survives the projection."""
    etype = (ev.get("event_type") or "").lower()
    if "correct" in etype or "breakdown" in etype or "repair" in etype:
        kind = "corrective"
    elif "prevent" in etype or "preventative" in etype:
        kind = "preventive"
    elif "inspect" in etype:
        kind = "inspection"
    else:
        kind = etype or "event"

    parts = ev.get("parts_used") or []
    parts_str = ", ".join(parts) if isinstance(parts, list) else str(parts)

    return {
        "wo_number": ev.get("work_order") or ev.get("event_id", ""),
        "date": ev.get("date", ""),
        "equipment": ev.get("asset_id", ""),
        "type": kind,
        "failure_mode": ev.get("failure_mode", ""),
        "symptom": ev.get("failure_mode", ""),
        "cause": ev.get("root_cause", ""),
        "action": ev.get("corrective_action", ""),
        "parts_used": parts_str,
        "downtime_hours": ev.get("downtime_hours", 0),
        # --- enrichment carried through for the Asset360 UI ----------------
        "event_id": ev.get("event_id", ""),
        "severity": ev.get("severity", ""),
        "root_cause": ev.get("root_cause", ""),
        "corrective_action": ev.get("corrective_action", ""),
        "preventive_action": ev.get("preventive_action", ""),
        "engineer": ev.get("engineer", ""),
        "cost": ev.get("cost", ""),
        "source_document": ev.get("source_document", ""),
        "page_number": ev.get("page_number", ""),
        "confidence": ev.get("confidence", 0),
        "from_document": True,
    }


class MaintenanceEventRepository:
    """File-backed store for the maintenance_events model (database fallback)."""

    def __init__(self, events_path: Path, csv_path: Path):
        self.events_path = Path(events_path)
        self.csv_path = Path(csv_path)

    # ---------------------------------------------------------------- read
    def all(self) -> list[dict]:
        if not self.events_path.exists():
            return []
        try:
            data = json.loads(self.events_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return [normalize_event(e) for e in data] if isinstance(data, list) else []

    def get(self, event_id: str) -> Optional[dict]:
        return next((e for e in self.all() if e.get("event_id") == event_id), None)

    def work_orders(self) -> set[str]:
        return {e.get("work_order", "") for e in self.all() if e.get("work_order")}

    # --------------------------------------------------------------- write
    def add(self, raw: dict) -> dict:
        """Persist one event. Assigns event_id/work_order/timestamps when
        absent, writes the JSON store and mirrors a row into the legacy CSV.
        Returns the stored event."""
        ev = normalize_event(raw)
        ev["event_id"] = ev["event_id"] or f"ME-{uuid.uuid4().hex[:10]}"
        if not ev["work_order"]:
            ev["work_order"] = "WO-" + _slug(ev["event_id"]).upper()[-8:]
        now = _now_iso()
        ev["created_at"] = ev["created_at"] or now
        ev["updated_at"] = now

        events = self.all()
        events = [e for e in events if e.get("event_id") != ev["event_id"]]
        events.append(ev)
        self._write_json(events)
        self._mirror_csv(ev)
        return ev

    def _write_json(self, events: list[dict]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.events_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(events, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self.events_path)

    def _mirror_csv(self, ev: dict) -> None:
        """Append the event to maintenance_log.csv (creating a header if the
        file is missing). Skips if a row with the same work order is already
        present, so re-commits do not duplicate the legacy mirror."""
        row = to_maintenance_row(ev)
        existing_wos: set[str] = set()
        if self.csv_path.exists():
            try:
                with open(self.csv_path, newline="", encoding="utf-8") as f:
                    existing_wos = {r.get("wo_number", "")
                                    for r in csv.DictReader(f)}
            except OSError:
                existing_wos = set()
        if row["wo_number"] in existing_wos:
            return
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})

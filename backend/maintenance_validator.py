"""Validation for extracted maintenance events.

Runs before anything is inserted into Asset360. Checks the four rules from the
design doc — asset exists, part exists, duplicate work order, valid date — and
returns structured findings. Nothing here blocks the UI outright: hard problems
surface as ``error`` findings (the preview dialog asks the user to fix them),
softer concerns as ``warning`` findings (the user may proceed). The user can
always edit and re-validate.
"""
from __future__ import annotations

import re
from datetime import date, datetime

# Asset node types that count as physical equipment (mirrors main.ASSET_TYPES).
_ASSET_TYPES = {"Pump", "Motor", "Valve", "Tank", "Exchanger", "Instrument"}


def _known_assets(corpus, graph) -> set[str]:
    assets = {nid for nid, d in graph.nodes.items()
              if d.get("type") in _ASSET_TYPES}
    assets |= {s["tag"] for s in corpus.pid.get("symbols", [])}
    return assets


def _known_parts(corpus) -> set[str]:
    return {s["part_number"].upper() for s in corpus.spares}


def _existing_work_orders(corpus) -> set[str]:
    return {w.get("wo_number", "") for w in corpus.maintenance if w.get("wo_number")}


def _parse_date(value: str):
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d %B %Y", "%d %b %Y",
                "%B %d, %Y", "%b %d, %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def validate(event: dict, corpus, graph) -> dict:
    """Validate one event against the current system of record.

    Returns ``{"ok": bool, "findings": [{level, field, message}, ...]}`` where
    ``ok`` is False only when there is at least one ``error`` finding.
    """
    findings: list[dict] = []
    assets = _known_assets(corpus, graph)
    parts = _known_parts(corpus)
    existing_wos = _existing_work_orders(corpus)

    # 1) Asset exists ---------------------------------------------------------
    asset_id = (event.get("asset_id") or "").strip()
    if not asset_id:
        findings.append({"level": "error", "field": "asset_id",
                         "message": "No asset selected for this event."})
    elif asset_id not in assets:
        findings.append({
            "level": "warning", "field": "asset_id",
            "message": f"Asset {asset_id} is not in the plant register — "
                       "it will be added as a new asset reference."})

    # 2) Part exists ----------------------------------------------------------
    for part in event.get("parts_used") or []:
        base = re.split(r"\s*x\d+$", str(part).strip(), flags=re.I)[0].upper()
        if base and base not in parts:
            findings.append({
                "level": "warning", "field": "parts_used",
                "message": f"Part {part} is not in the spares catalogue — "
                           "usage will still be recorded."})

    # 3) Duplicate work order -------------------------------------------------
    wo = (event.get("work_order") or "").strip()
    if wo and wo in existing_wos:
        findings.append({
            "level": "warning", "field": "work_order",
            "message": f"Work order {wo} already exists in the history — "
                       "committing will update the existing entry."})

    # 4) Valid date -----------------------------------------------------------
    raw_date = (event.get("date") or "").strip()
    if not raw_date:
        findings.append({"level": "error", "field": "date",
                         "message": "Event date is missing."})
    else:
        parsed = _parse_date(raw_date)
        if parsed is None:
            findings.append({"level": "error", "field": "date",
                             "message": f"Date '{raw_date}' is not a "
                                        "recognizable date."})
        elif parsed > date.today():
            findings.append({"level": "warning", "field": "date",
                             "message": f"Date {raw_date} is in the future."})

    ok = not any(f["level"] == "error" for f in findings)
    return {"ok": ok, "findings": findings}


def validate_all(events: list[dict], corpus, graph) -> list[dict]:
    """Attach a ``validation`` block to each event; return the annotated list."""
    out = []
    for ev in events:
        annotated = dict(ev)
        annotated["validation"] = validate(ev, corpus, graph)
        out.append(annotated)
    return out

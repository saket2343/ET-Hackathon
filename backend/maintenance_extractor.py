"""Structured maintenance information extraction.

Turns the free text of an uploaded maintenance document into structured events
matching the maintenance_events schema. LLM-first (asks the configured provider
for strict JSON) with a fully deterministic regex fallback, so extraction still
produces a usable draft when no LLM is configured — the user reviews and edits
everything before anything is committed, so a rough draft is fine.

Never inserts anything; it only produces candidate events for the preview
dialog.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import llm
from history_repository import EVENT_FIELDS, normalize_event

_ASSET_TAG = re.compile(r"\b[A-Z]{1,3}-\d{2,3}\b")
_PART_TAG = re.compile(r"\b(?:BRG|MS|CE|VG\d+|SEAL|GKT|BLT)-?[A-Z0-9]+\b", re.I)
_WO_TAG = re.compile(r"\bWO[-\s]?(\d{3,5})\b", re.I)

_EXTRACT_SYSTEM = """You are an information-extraction engine for an industrial
maintenance system. Extract maintenance facts from the supplied document text
into STRICT JSON. Output ONLY a JSON array (no prose, no markdown fences).

Each array element describes ONE maintenance event for ONE asset and MUST use
exactly these keys:
{
 "asset_id": "", "event_type": "", "date": "", "work_order": "",
 "failure_mode": "", "severity": "", "root_cause": "", "corrective_action": "",
 "preventive_action": "", "parts_used": [], "downtime_hours": 0,
 "engineer": "", "cost": "", "page_number": "", "confidence": 0
}

Rules:
- If the document references multiple assets, emit one object per asset.
- asset_id is an equipment tag like P-101, M-101, E-201.
- date in YYYY-MM-DD if determinable, else the date string as written.
- parts_used is a JSON array of part numbers/strings.
- downtime_hours is a number.
- confidence is your 0-1 confidence in the extraction.
- Use "" (or [] / 0) for anything not stated. Never invent facts.
"""


def _strip_json(raw: str) -> str:
    """Pull the JSON payload out of a model response that may be fenced or
    prefixed with prose."""
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("[")
    if start == -1:
        start = raw.find("{")
    end = max(raw.rfind("]"), raw.rfind("}"))
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw


def _parse_events(raw: str) -> list[dict]:
    try:
        data = json.loads(_strip_json(raw))
    except (json.JSONDecodeError, ValueError):
        return []
    if isinstance(data, dict):
        data = [data]
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def _doc_pages(corpus, doc_no: str) -> list[tuple[int, str]]:
    return [(c.page, c.text) for c in corpus.chunks if c.doc_no == doc_no]


def candidate_assets(corpus, doc_no: str, known_assets: set[str]) -> list[str]:
    """Asset tags the document mentions, known plant assets first, ordered by
    mention frequency."""
    from collections import Counter
    from document_classifier import _is_asset_tag
    text = "\n".join(t for _, t in _doc_pages(corpus, doc_no))
    counts = Counter(t for t in _ASSET_TAG.findall(text) if _is_asset_tag(t))
    known = [t for t, _ in counts.most_common() if t in known_assets]
    unknown = [t for t, _ in counts.most_common() if t not in known_assets]
    return known + unknown


def _page_for(corpus, doc_no: str, needle: str) -> str:
    for page, text in _doc_pages(corpus, doc_no):
        if needle and needle.lower() in text.lower():
            return str(page)
    pages = _doc_pages(corpus, doc_no)
    return str(pages[0][0]) if pages else ""


def _guess_date(text: str) -> str:
    iso = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text)
    if iso:
        return f"{iso.group(1)}-{int(iso.group(2)):02d}-{int(iso.group(3)):02d}"
    dmy = re.search(
        r"\b(\d{1,2})\s+"
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(20\d{2})\b",
        text, re.I)
    if dmy:
        months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                  "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
        return (f"{dmy.group(3)}-{months[dmy.group(2).lower()[:3]]:02d}"
                f"-{int(dmy.group(1)):02d}")
    return ""


def _first(patterns: list[str], text: str) -> str:
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return (m.group(1) if m.groups() else m.group(0)).strip(" .:-")
    return ""


def _deterministic(corpus, doc_no: str, asset_ids: list[str],
                   classification: dict) -> list[dict]:
    """Regex/keyword extraction — the no-LLM fallback. Produces one draft
    event per asset from the document's cues."""
    text = "\n".join(t for _, t in _doc_pages(corpus, doc_no))
    date = _guess_date(text)
    wo = _WO_TAG.search(text)
    wo_number = f"WO-{wo.group(1)}" if wo else ""
    parts = sorted({p.upper() for p in _PART_TAG.findall(text)})
    dt = re.search(r"\bdowntime[:\s]*([\d.]+)\s*(?:h|hours?)\b", text, re.I) \
        or re.search(r"\b([\d.]+)\s*hours?\s+downtime\b", text, re.I)
    downtime = float(dt.group(1)) if dt else 0.0
    event_type = classification.get("type", "Maintenance Report")

    failure = _first([r"failure\s*mode[:\s]+([^\n.]{3,60})",
                      r"\b(bearing failure|seal leak|tube fouling|"
                      r"misalignment|overheating|vibration)\b"], text)
    cause = _first([r"root\s*cause[:\s]+([^\n.]{3,80})",
                    r"\bcaused?\s+by\s+([^\n.]{3,80})",
                    r"\bcause[:\s]+([^\n.]{3,80})"], text)
    action = _first([r"corrective\s*action[:\s]+([^\n.]{3,100})",
                     r"\baction\s*(?:taken)?[:\s]+([^\n.]{3,100})",
                     r"\b(replaced?[^\n.]{3,80})"], text)
    preventive = _first([r"preventive\s*action[:\s]+([^\n.]{3,100})",
                         r"recommend(?:ed|ation)?[:\s]+([^\n.]{3,100})"], text)
    severity = _first([r"severity[:\s]+([^\n.]{2,20})",
                       r"\b(critical|high|medium|low)\s+severity\b"], text)
    engineer = _first([r"engineer[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                       r"(?:performed|carried out|inspected)\s+by[:\s]+"
                       r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"], text)
    cost = _first([r"cost[:\s]*((?:USD|INR|Rs|\$|₹)?\s?[\d,]+(?:\.\d+)?)"], text)

    events = []
    for asset in asset_ids:
        anchor = asset if asset in text else (failure or action or "")
        events.append(normalize_event({
            "asset_id": asset,
            "event_type": event_type,
            "date": date,
            "work_order": wo_number,
            "failure_mode": failure,
            "severity": severity,
            "root_cause": cause,
            "corrective_action": action,
            "preventive_action": preventive,
            "parts_used": parts,
            "downtime_hours": downtime,
            "engineer": engineer,
            "cost": cost,
            "source_document": corpus.docs.get(doc_no, {}).get("title", doc_no),
            "page_number": _page_for(corpus, doc_no, anchor),
            "confidence": round(0.55 + 0.1 * bool(failure) + 0.1 * bool(action)
                                + 0.1 * bool(date), 2),
        }))
    return events


def extract(corpus, doc_no: str, *, known_assets: set[str],
            classification: dict, asset_ids: list[str] | None = None,
            use_llm: bool = True) -> list[dict]:
    """Extract structured maintenance events from an ingested document.

    ``asset_ids`` restricts extraction to the chosen assets (the multi-asset
    picker). When omitted, every candidate asset the document mentions is used.
    Returns a list of normalized events, each carrying source_document and
    page_number for traceability.
    """
    title = corpus.docs.get(doc_no, {}).get("title", doc_no)
    if asset_ids is None:
        asset_ids = candidate_assets(corpus, doc_no, known_assets)
    if not asset_ids:
        asset_ids = [""]  # single event with no resolved asset (user edits it)

    events: list[dict] = []
    if use_llm:
        pages = _doc_pages(corpus, doc_no)
        body = "\n\n".join(f"[page {p}] {t}" for p, t in pages)[:9000]
        prompt = (f"Document title: {title}\n"
                  f"Known plant asset tags: {', '.join(sorted(known_assets))}\n"
                  f"Restrict extraction to these assets if present: "
                  f"{', '.join(a for a in asset_ids if a)}\n\n"
                  f"Document text:\n{body}")
        try:
            raw = llm.complete(prompt, _EXTRACT_SYSTEM)
        except Exception:
            raw = None
        if raw:
            wanted = {a for a in asset_ids if a}
            for obj in _parse_events(raw):
                obj.setdefault("source_document", title)
                ev = normalize_event(obj)
                if wanted and ev["asset_id"] and ev["asset_id"] not in wanted:
                    continue
                if not ev.get("page_number"):
                    ev["page_number"] = _page_for(
                        corpus, doc_no, ev.get("failure_mode")
                        or ev.get("corrective_action") or ev.get("asset_id"))
                ev["source_document"] = title
                events.append(ev)

    # Fall back (or backfill missing assets) deterministically so every
    # selected asset yields a draft event even if the LLM missed it.
    covered = {e["asset_id"] for e in events if e["asset_id"]}
    missing = [a for a in asset_ids if a and a not in covered]
    if not events or missing:
        targets = missing or asset_ids
        events.extend(_deterministic(corpus, doc_no, targets, classification))

    return events

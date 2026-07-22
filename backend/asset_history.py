"""Asset360 Digital History generator.

This module is deliberately additive: uploads are indexed by ``ingest`` first
and this generator is only called after the user explicitly approves history
generation.  It creates a durable, source-traceable asset folder that can be
merged on later uploads without replacing earlier history.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import document_classifier as classifier
import maintenance_extractor as extractor
from history_repository import MaintenanceEventRepository, normalize_event


ASSET_TYPES = {
    "Pump", "Compressor", "Exchanger", "Valve", "Tank", "Motor",
    "Instrument", "Fan", "Blower", "Turbine", "Generator", "Equipment",
}
_TAG = re.compile(r"\b(?:[A-Z]{1,3}-\d{2,4}|(?:Tank|Motor|Pump|Compressor|"
                  r"Valve|Exchanger|Fan|Blower)-\d{2,4})\b", re.I)
_PREFIX_TYPE = {
    "P": "Pump", "C": "Compressor", "E": "Exchanger", "V": "Valve",
    "T": "Tank", "M": "Motor", "FI": "Instrument", "PI": "Instrument",
    "TI": "Instrument", "TT": "Instrument", "LT": "Instrument",
    "PT": "Instrument", "FT": "Instrument", "TANK": "Tank",
    "MOTOR": "Motor", "PUMP": "Pump", "COMPRESSOR": "Compressor",
    "VALVE": "Valve", "EXCHANGER": "Exchanger", "FAN": "Fan",
    "BLOWER": "Blower",
}
_FIELDS = {
    "manufacturer": [r"manufacturer\s*[:#-]\s*([^\n]{2,80})", r"make\s*[:#-]\s*([^\n]{2,80})"],
    "vendor": [r"vendor\s*[:#-]\s*([^\n]{2,80})", r"supplier\s*[:#-]\s*([^\n]{2,80})"],
    "model_number": [r"model\s*(?:number|no\.?|#)?\s*[:#-]\s*([^\n]{2,80})"],
    "serial_number": [r"serial\s*(?:number|no\.?|#)?\s*[:#-]\s*([^\n]{2,80})"],
    "area": [r"\barea\s*[:#-]\s*([^\n]{2,80})"],
    "location": [r"\blocation\s*[:#-]\s*([^\n]{2,80})"],
    "department": [r"\bdepartment\s*[:#-]\s*([^\n]{2,80})"],
    "pressure_rating": [r"pressure\s*(?:rating)?\s*[:#-]\s*([^\n]{2,80})"],
    "temperature_rating": [r"temperature\s*(?:rating)?\s*[:#-]\s*([^\n]{2,80})"],
    "capacity": [r"\bcapacity\s*[:#-]\s*([^\n]{2,80})"],
    "flow_rate": [r"flow\s*(?:rate)?\s*[:#-]\s*([^\n]{2,80})"],
    "motor_power": [r"motor\s*power\s*[:#-]\s*([^\n]{2,80})", r"\bpower\s*[:#-]\s*([^\n]{2,80})"],
    "installation_date": [r"install(?:ation|ed)?\s*date\s*[:#-]\s*([^\n]{2,50})"],
    "service_date": [r"service\s*date\s*[:#-]\s*([^\n]{2,50})"],
    "inspection_date": [r"inspection\s*date\s*[:#-]\s*([^\n]{2,50})"],
    "failure_date": [r"failure\s*date\s*[:#-]\s*([^\n]{2,50})"],
    "engineer": [r"engineer\s*[:#-]\s*([^\n]{2,80})"],
    "supervisor": [r"supervisor\s*[:#-]\s*([^\n]{2,80})"],
    "operating_hours": [r"operating\s*hours?\s*[:#-]\s*([^\n]{2,50})"],
    "warranty_expiry": [r"warranty\s*(?:expiry|expiration|end)\s*[:#-]\s*([^\n]{2,50})"],
    "maintenance_frequency": [r"maintenance\s*frequency\s*[:#-]\s*([^\n]{2,80})"],
    "operating_conditions": [r"operating\s*conditions?\s*[:#-]\s*([^\n]{2,120})"],
    "criticality": [r"criticality\s*[:#-]\s*([^\n]{2,40})"],
    "health_score": [r"health\s*score\s*[:#-]?\s*(\d{1,3})"],
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_id(asset_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", asset_id.upper()).strip("-")


def _first(patterns: list[str], text: str) -> str:
    for pattern in patterns:
        found = re.search(pattern, text, re.I)
        if found:
            return found.group(1).strip(" .;:-")
    return ""


def infer_type(asset_id: str, text: str = "") -> str:
    prefix = asset_id.upper().split("-", 1)[0]
    if prefix in _PREFIX_TYPE:
        return _PREFIX_TYPE[prefix]
    around = text[max(0, text.lower().find(asset_id.lower()) - 100):].lower()
    for word, asset_type in (("compressor", "Compressor"), ("pump", "Pump"),
                             ("heat exchanger", "Exchanger"), ("valve", "Valve"),
                             ("tank", "Tank"), ("motor", "Motor"), ("fan", "Fan")):
        if word in around:
            return asset_type
    return "Equipment"


def detect_assets(text: str) -> list[dict]:
    """Regex asset recognition, retaining only tags that look like equipment."""
    counts = Counter(m.group(0).upper() for m in _TAG.finditer(text))
    out = []
    for tag, count in counts.most_common():
        prefix = tag.split("-", 1)[0]
        # Manuals commonly mention model numbers, standards and certificates
        # that share the TAG-NNN shape but are not physical assets.
        if prefix in {"SOP", "SAF", "WO", "RCA", "BRG", "MS", "GKT", "BLT",
                      "API", "ISO", "ASME", "ANSI", "CE", "CX", "IEC", "NEMA"}:
            continue
        asset_type = infer_type(tag, text)
        out.append({"asset_id": tag, "asset_tag": tag, "asset_type": asset_type,
                    "asset_name": f"{tag} {asset_type}",
                    "confidence": round(min(0.98, 0.78 + 0.04 * min(count, 4)), 2)})
    return out


def extract_metadata(text: str, asset: dict, classification: dict, title: str) -> dict:
    """Extract only stated metadata; blank values are intentionally retained."""
    metadata = {key: _first(patterns, text) for key, patterns in _FIELDS.items()}
    metadata.update({
        "asset_id": asset["asset_id"], "asset_tag": asset["asset_tag"],
        "equipment_type": asset["asset_type"], "equipment_name": asset["asset_name"],
        "document_classification": classification.get("type", "General Document"),
        "classification_confidence": classification.get("confidence", 0),
        "confidence_score": asset["confidence"], "source_title": title,
    })
    return metadata


def _merge(existing: dict, incoming: dict) -> dict:
    result = dict(existing or {})
    for key, value in incoming.items():
        if key in {"source_documents", "classifications"}:
            result[key] = sorted(set((result.get(key) or []) + (value or [])))
        elif value not in (None, "", [], {}):
            result[key] = value
    return result


def _md_escape(value: object) -> str:
    return str(value or "—").replace("\n", " ").strip()


class AssetHistoryGenerator:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.assets_dir = self.data_dir / "assets"
        self.repo = MaintenanceEventRepository(self.data_dir / "maintenance_events.json",
                                               self.data_dir / "maintenance_log.csv")

    def _folder(self, asset_id: str) -> Path:
        return self.assets_dir / _safe_id(asset_id)

    def _read_metadata(self, asset_id: str) -> dict:
        path = self._folder(asset_id) / "metadata.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _document_text(self, corpus, doc_no: str) -> str:
        return "\n\n".join(chunk.text for chunk in corpus.chunks if chunk.doc_no == doc_no)

    def _store_events(self, events: list[dict], source: str) -> list[dict]:
        existing = self.repo.all()
        seen = {(e.get("asset_id"), e.get("source_document"), e.get("date"),
                 e.get("failure_mode"), e.get("corrective_action")) for e in existing}
        added = []
        for raw in events:
            event = normalize_event(raw)
            event["source_document"] = source
            key = (event["asset_id"], event["source_document"], event["date"],
                   event["failure_mode"], event["corrective_action"])
            # A dated, substantive record is safe to persist after the user's
            # explicit Generate History approval.  Undated snippets remain in
            # the generated report but are not presented as history events.
            if not event["asset_id"] or not event["date"] or key in seen:
                continue
            if not any(event.get(k) for k in ("failure_mode", "corrective_action", "preventive_action")):
                continue
            added.append(self.repo.add(event))
            seen.add(key)
        return added

    def _write_markdown(self, path: Path, doc_no: str, title: str, asset_id: str,
                        source: str, body: str) -> None:
        # Preserve prior generated knowledge and append only genuinely new
        # source material. This makes repeated uploads additive rather than a
        # lossy rewrite of the asset's existing history.
        if path.exists():
            try:
                old = path.read_text(encoding="utf-8")
                old_body = old.split("---", 2)[2].strip() if old.startswith("---") else old.strip()
                if f"[{source}]" in old_body:
                    body = old_body
                else:
                    body = old_body + f"\n\n---\n\n## Merged update from {source}\n\n" + body
            except OSError:
                pass
        content = ("---\n"
                   f"doc_no: {doc_no}\n"
                   f"title: {title}\n"
                   "revision: generated\n"
                   f"governs: [{asset_id}]\n"
                   f"source_document: {source}\n"
                   "generated: true\n"
                   "---\n\n" + body.strip() + "\n")
        path.write_text(content, encoding="utf-8")

    def _write_artifacts(self, asset_id: str, metadata: dict, source: str,
                         events: list[dict]) -> None:
        folder = self._folder(asset_id)
        folder.mkdir(parents=True, exist_ok=True)
        prefix = f"ASSET-{_safe_id(asset_id)}"
        citation = f"Source: [{source}] (uploaded document; extraction confidence {int(float(metadata.get('confidence_score') or 0) * 100)}%)."
        # All generated files are concise, evidence-oriented views of the
        # source.  Missing source fields are marked as not stated, never made up.
        manual = f"""# Company Manual — {asset_id}

## Asset Description
{_md_escape(metadata.get('equipment_name'))} is recorded as a {_md_escape(metadata.get('equipment_type'))}.

## Equipment Overview
Manufacturer: {_md_escape(metadata.get('manufacturer'))}  
Model: {_md_escape(metadata.get('model_number'))}  
Serial number: {_md_escape(metadata.get('serial_number'))}

## Operating Conditions and Limits
Pressure rating: {_md_escape(metadata.get('pressure_rating'))}  
Temperature rating: {_md_escape(metadata.get('temperature_rating'))}  
Capacity: {_md_escape(metadata.get('capacity'))}  
Flow rate: {_md_escape(metadata.get('flow_rate'))}  
Motor power: {_md_escape(metadata.get('motor_power'))}  
Operating conditions: {_md_escape(metadata.get('operating_conditions'))}

## Safety, Startup, Shutdown and Maintenance
Use the uploaded source and plant SOPs for approved procedures. The source does not establish a generated operating procedure where details are absent.

## Source Citation
{citation}"""
        history_lines = [f"- **{e.get('date')}** — {e.get('event_type') or 'Maintenance'}: "
                         f"{e.get('failure_mode') or e.get('corrective_action') or 'recorded event'} "
                         f"(WO {e.get('work_order') or 'not stated'}; source page {e.get('page_number') or 'not stated'})"
                         for e in events]
        history = "# Maintenance History — " + asset_id + "\n\n" + ("\n".join(history_lines) if history_lines else
                    "No dated maintenance event was extracted from this source.") + f"\n\n## Source Citation\n{citation}"
        service = f"""# Service Report — {asset_id}

Technician: {_md_escape(metadata.get('engineer'))}  
Service date: {_md_escape(metadata.get('service_date'))}  
Health score: {_md_escape(metadata.get('health_score'))}

## Checklist
Oil checked, bearing checked, alignment checked, seal checked, motor checked, current checked, temperature checked, vibration checked and leakage checked: not stated unless evidenced in the uploaded report.

## Remarks and Recommendation
{_md_escape(next((e.get('preventive_action') for e in events if e.get('preventive_action')), 'Not stated'))}

## Source Citation
{citation}"""
        inspection = f"""# Inspection Report — {asset_id}

Inspector: {_md_escape(metadata.get('engineer'))}  
Inspection date: {_md_escape(metadata.get('inspection_date'))}

Visual inspection, leakage, noise, vibration, temperature, pressure and safety status are retained as **not stated** unless present in the uploaded document. Overall health: {_md_escape(metadata.get('health_score'))}.

## Recommendations
{_md_escape(next((e.get('preventive_action') for e in events if e.get('preventive_action')), 'Not stated'))}

## Source Citation
{citation}"""
        timeline_events = [(metadata.get("installation_date"), "Installed"),
                           (metadata.get("service_date"), "Service recorded"),
                           (metadata.get("inspection_date"), "Inspection recorded")]
        timeline_events += [(e.get("date"), e.get("event_type") or "Maintenance event") for e in events]
        timeline = "# Timeline — " + asset_id + "\n\n" + "\n".join(
            f"- **{date or 'Date not stated'}** — {label}" for date, label in timeline_events if date or label) + f"\n\n## Source Citation\n{citation}"
        docs = {
            "company_manual.md": (f"{prefix}-MANUAL", f"Company Manual — {asset_id}", manual),
            "maintenance_report.md": (f"{prefix}-MAINTENANCE", f"Maintenance History — {asset_id}", history),
            "service_report.md": (f"{prefix}-SERVICE", f"Service Report — {asset_id}", service),
            "inspection_report.md": (f"{prefix}-INSPECTION", f"Inspection Report — {asset_id}", inspection),
            "timeline.md": (f"{prefix}-TIMELINE", f"Timeline — {asset_id}", timeline),
        }
        for filename, (doc_no, title, body) in docs.items():
            self._write_markdown(folder / filename, doc_no, title, asset_id, source, body)
        metadata_path = folder / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        history_json = [e for e in self.repo.all() if e.get("asset_id") == asset_id]
        (folder / "history.json").write_text(json.dumps(history_json, indent=2, ensure_ascii=False), encoding="utf-8")
        sensor = {"asset_id": asset_id, "available": asset_id == "P-101",
                  "summary": "Live P-101 telemetry is available through Asset360." if asset_id == "P-101" else "No telemetry source is linked yet."}
        (folder / "sensor_summary.json").write_text(json.dumps(sensor, indent=2), encoding="utf-8")
        # Keep a compact manifest rather than copying source text into artifacts.
        (folder / "citations.json").write_text(json.dumps({"asset_id": asset_id, "sources": [source]}, indent=2), encoding="utf-8")

    def generate(self, corpus, graph, doc_no: str, reingest: Callable[[], object]) -> dict:
        if doc_no not in corpus.docs:
            raise KeyError(doc_no)
        title = corpus.docs[doc_no].get("title", doc_no)
        text = self._document_text(corpus, doc_no)
        classification = classifier.classify_with_llm_verification(text, title=title)
        assets = detect_assets(text)
        if not assets:
            return {"ok": False, "asset_ids": [], "classification": classification,
                    "message": "No engineering asset tag was detected; the document remains indexed and searchable."}

        known = {node_id for node_id, node in graph.nodes.items()
                 if node.get("type") in ASSET_TYPES} | {a["asset_id"] for a in assets}
        extracted = extractor.extract(corpus, doc_no, known_assets=known,
                                      classification=classification,
                                      asset_ids=[a["asset_id"] for a in assets], use_llm=True)
        stored = self._store_events(extracted, title)
        all_events = self.repo.all()
        created: list[str] = []
        for asset in assets:
            old = self._read_metadata(asset["asset_id"])
            if not old:
                created.append(asset["asset_id"])
            incoming = extract_metadata(text, asset, classification, title)
            incoming.update({"source_documents": [title], "classifications": [classification.get("type")],
                             "created_at": old.get("created_at") or _now(), "last_modified": _now()})
            metadata = _merge(old, incoming)
            asset_events = [e for e in all_events if e.get("asset_id") == asset["asset_id"]]
            metadata.update({"document_count": len(metadata.get("source_documents", [])),
                             "history_events": len(asset_events),
                             "last_service": max((e.get("date", "") for e in asset_events), default=""),
                             "status": "Healthy" if not asset_events else "History available"})
            self._write_artifacts(asset["asset_id"], metadata, title, asset_events)

        # The generated markdown is now part of the corpus and the asset
        # registry is rebuilt in-process, so no server restart is needed.
        reingest()
        return {"ok": True, "asset_ids": [a["asset_id"] for a in assets],
                "created": created,
                "classification": classification, "events_created": len(stored),
                "message": "Asset360 digital history generated and indexed."}

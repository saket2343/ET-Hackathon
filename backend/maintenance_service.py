"""Orchestrator for the Asset360 maintenance ingestion workflow.

Ties the pieces together behind a small API the FastAPI layer calls:

    analyze(doc_no)                 -> classification + maintenance detection
    preview(doc_no, asset_ids?)     -> extracted events + validation findings
    commit(events)                  -> persist + update Asset360 live

The service never mutates Asset360 on its own; it only prepares candidates.
Mutation happens exclusively in ``commit``, after explicit user confirmation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import document_classifier as dc
import maintenance_extractor as extractor
import maintenance_validator as validator
from asset360_updater import Asset360Updater
from history_repository import MaintenanceEventRepository, normalize_event

_ASSET_TYPES = {"Pump", "Motor", "Valve", "Tank", "Exchanger", "Instrument"}


class MaintenanceService:
    def __init__(self, data_dir: Path):
        data_dir = Path(data_dir)
        self.repo = MaintenanceEventRepository(
            events_path=data_dir / "maintenance_events.json",
            csv_path=data_dir / "maintenance_log.csv",
        )
        self.updater = Asset360Updater(self.repo, data_dir / "spares.csv")

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _known_assets(corpus, graph) -> set[str]:
        assets = {nid for nid, d in graph.nodes.items()
                  if d.get("type") in _ASSET_TYPES}
        assets |= {s["tag"] for s in corpus.pid.get("symbols", [])}
        return assets

    # ------------------------------------------------------------ analyze
    def analyze(self, corpus, graph, doc_no: str) -> dict:
        """Classify a document and decide whether to offer an Asset360 update."""
        classification = dc.classify_corpus_doc(corpus, doc_no)
        assets = self._known_assets(corpus, graph)
        candidates = extractor.candidate_assets(corpus, doc_no, assets)
        return {
            "doc_no": doc_no,
            "title": classification.get("title", doc_no),
            "classification": classification,
            "maintenance_detected": classification["maintenance_related"],
            "candidate_assets": candidates,
        }

    # ------------------------------------------------------------ preview
    def preview(self, corpus, graph, doc_no: str,
                asset_ids: Optional[list[str]] = None,
                use_llm: bool = True) -> dict:
        """Extract structured events and attach validation findings. No writes."""
        classification = dc.classify_corpus_doc(corpus, doc_no)
        assets = self._known_assets(corpus, graph)
        events = extractor.extract(
            corpus, doc_no, known_assets=assets,
            classification=classification, asset_ids=asset_ids, use_llm=use_llm)
        annotated = validator.validate_all(events, corpus, graph)
        return {
            "doc_no": doc_no,
            "title": classification.get("title", doc_no),
            "classification": classification,
            "candidate_assets": extractor.candidate_assets(corpus, doc_no, assets),
            "events": annotated,
        }

    # ------------------------------------------------------------- commit
    def commit(self, corpus, graph, events: list[dict],
               reingest: Callable[[], None]) -> dict:
        """Validate, persist and apply confirmed events to Asset360 live.

        Returns a summary including any residual validation findings so the UI
        can report exactly what happened.
        """
        clean = [normalize_event(e) for e in events]
        findings = validator.validate_all(clean, corpus, graph)
        blocking = [f for ev in findings for f in ev["validation"]["findings"]
                    if f["level"] == "error"]
        if blocking:
            return {"ok": False, "committed": 0, "findings": blocking,
                    "message": "Resolve the highlighted problems before committing."}
        summary = self.updater.commit(clean, reingest)
        summary["ok"] = True
        return summary

    # -------------------------------------------------------------- reads
    def events(self) -> list[dict]:
        return self.repo.all()

    def get_event(self, event_id: str) -> Optional[dict]:
        return self.repo.get(event_id)

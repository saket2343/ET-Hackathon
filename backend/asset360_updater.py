"""Applies confirmed maintenance events to Asset360 — live, without a restart.

Once the user confirms extracted events, this module:
  1. persists them through the maintenance_events repository (system of record),
  2. draws down spares inventory for the parts used (updates data/spares.csv),
  3. triggers an in-process re-ingest so every Asset360 surface — failure
     history, maintenance timeline, spares, knowledge graph, documents and
     asset statistics — reflects the new events immediately.

The knowledge-graph relationships (HAS_FAILURE / CAUSED_BY / USED_PART /
RECOMMENDED) are derived by ``kg.build_graph`` from the persisted events, so the
re-ingest in step 3 is what refreshes the graph; this module owns persistence
and inventory, and orchestrates the refresh.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Callable

from history_repository import MaintenanceEventRepository


def _part_base(part: str) -> str:
    return re.split(r"\s*x\d+$", str(part).strip(), flags=re.I)[0].upper()


def _part_qty(part: str) -> int:
    m = re.search(r"x\s*(\d+)\s*$", str(part), re.I)
    return int(m.group(1)) if m else 1


class Asset360Updater:
    def __init__(self, repo: MaintenanceEventRepository, spares_path: Path):
        self.repo = repo
        self.spares_path = Path(spares_path)

    # ------------------------------------------------------------- spares
    def _draw_down_spares(self, events: list[dict]) -> list[dict]:
        """Subtract used quantities from spares on hand. Returns per-part
        deltas. Unknown parts are ignored here (they are surfaced as
        validation warnings upstream)."""
        if not self.spares_path.exists():
            return []
        with open(self.spares_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            rows = list(reader)
        by_part = {r["part_number"].upper(): r for r in rows}

        deltas: list[dict] = []
        for ev in events:
            for part in ev.get("parts_used") or []:
                base, used = _part_base(part), _part_qty(part)
                row = by_part.get(base)
                if row is None:
                    continue
                before = int(float(row.get("qty_on_hand", 0) or 0))
                after = max(0, before - used)
                row["qty_on_hand"] = str(after)
                deltas.append({"part_number": base, "used": used,
                               "before": before, "after": after,
                               "min_stock": int(float(row.get("min_stock", 0) or 0)),
                               "low_stock": after <= int(float(row.get("min_stock", 0) or 0))})
        if deltas:
            tmp = self.spares_path.with_suffix(".csv.tmp")
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            tmp.replace(self.spares_path)
        return deltas

    # ------------------------------------------------------------- commit
    def commit(self, events: list[dict], reingest: Callable[[], None]) -> dict:
        """Persist events, draw down spares, and rebuild the live corpus.

        ``reingest`` is the app's in-process rebuild callback (main._reingest);
        calling it is what makes Asset360 update without a backend restart.
        Returns a summary for the UI.
        """
        stored = [self.repo.add(ev) for ev in events]
        spares_deltas = self._draw_down_spares(stored)
        reingest()  # rebuild corpus + graph + index + agents, in process
        return {
            "committed": len(stored),
            "event_ids": [e["event_id"] for e in stored],
            "work_orders": [e["work_order"] for e in stored],
            "asset_ids": sorted({e["asset_id"] for e in stored if e["asset_id"]}),
            "spares_updated": spares_deltas,
        }

"""Document classifier for the Asset360 ingestion workflow.

Given the text of an uploaded document, decide what KIND of document it is and
whether it carries maintenance information worth offering to fold into
Asset360. Deterministic and dependency-free (keyword/structure scoring) so it
works with or without an LLM — the classification gate must never depend on a
network call.

Only a subset of document types should trigger the "update Asset360?" prompt
(design doc): Maintenance Report, Inspection Report, Work Order, Breakdown
Report, Root Cause Analysis, Sensor Report. Everything else is indexed into RAG
only, exactly as today.
"""
from __future__ import annotations

import os
import re

# Document types the classifier can assign.
DOC_TYPES = (
    "OEM Manual", "Company Manual", "Maintenance Report", "Inspection Report",
    "Work Order", "Commissioning Report", "Safety SOP", "Operating Procedure",
    "Calibration Report", "Failure Report", "Root Cause Analysis", "Warranty",
    "Datasheet", "P&ID", "Maintenance Checklist", "Sensor Logs",
    "Vendor Report", "Service Report", "General Document",
)

# Only these types offer to update Asset360.
MAINTENANCE_TYPES = frozenset({
    "Maintenance Report", "Inspection Report", "Work Order",
    "Failure Report", "Root Cause Analysis", "Sensor Logs", "Service Report",
    "Maintenance Checklist", "Calibration Report",
})

# Weighted keyword signatures per type. Scoring is document-frequency style:
# each signature that appears anywhere in the text contributes its weight once.
_SIGNATURES: dict[str, list[tuple[str, float]]] = {
    "Work Order": [
        (r"\bwork\s*order\b", 3), (r"\bwo[-\s]?\d", 3), (r"\bjob\s*card\b", 2),
        (r"\bpermit\s*to\s*work\b", 1), (r"\bassigned\s*to\b", 1),
        (r"\blabou?r\s*hours\b", 1),
    ],
    "Maintenance Report": [
        (r"\bmaintenance\s*(report|record|log)\b", 3),
        (r"\bcorrective\s*maintenance\b", 2), (r"\bpreventive\s*maintenance\b", 2),
        (r"\brepair(ed|s)?\b", 1), (r"\breplaced?\b", 1),
        (r"\bdowntime\b", 2), (r"\bparts?\s*used\b", 2),
    ],
    "Inspection Report": [
        (r"\binspection\s*(report|checklist|record)\b", 3),
        (r"\bvisual\s*inspection\b", 2), (r"\bthickness\s*measurement\b", 2),
        (r"\brecommended?\b", 1), (r"\bfindings?\b", 1), (r"\bcondition\b", 1),
        (r"\bntd\b|\bnon[-\s]?destructive\b", 2),
    ],
    "Failure Report": [
        (r"\bbreakdown\b", 3), (r"\bfailure\s*report\b", 3),
        (r"\bunplanned\s*(stop|shutdown|outage)\b", 2),
        (r"\btripped?\b", 1), (r"\bunexpected\s*(stop|failure)\b", 2),
    ],
    "Root Cause Analysis": [
        (r"\broot\s*cause\b", 3), (r"\brca\b", 3), (r"\bfishbone\b", 2),
        (r"\b5\s*whys?\b", 2), (r"\bcontributing\s*factors?\b", 1),
        (r"\bcorrective\s*action\b", 1), (r"\bpreventive\s*action\b", 1),
    ],
    "Sensor Logs": [
        (r"\bvibration\b", 2), (r"\bmm/s\b", 2), (r"\bsensor\b", 2),
        (r"\btelemetry\b", 2), (r"\bcondition\s*monitoring\b", 3),
        (r"\biso\s*10816\b", 2), (r"\btrend\b", 1), (r"\bbearing\s*temp\b", 2),
        (r"\banomaly\b", 1),
    ],
    "OEM Manual": [
        (r"\b(operating|installation|service|user)\s*manual\b", 3),
        (r"\b(vendor|oem|equipment|pump|compressor)\s*manual\b", 5),
        (r"\bdatasheet\b", 2), (r"\bspecifications?\b", 1),
        (r"\btorque\s*settings?\b", 1), (r"\bmodel\s*(no|number)\b", 1),
        (r"\bmanufacturer\b", 1),
    ],
    "P&ID": [
        (r"\bp\s*&\s*id\b", 3), (r"\bpiping\s*and\s*instrumentation\b", 3),
        (r"\bloop\s*diagram\b", 2), (r"\bdrawing\s*(no|number)\b", 1),
        (r"\bline\s*list\b", 1),
    ],
    "Operating Procedure": [
        (r"\bstandard\s*operating\s*procedure\b", 3), (r"\bsop[-\s]?\d", 3),
        (r"\bstep\s*\d+\b", 1), (r"\bprocedure\b", 1), (r"\bshall\b", 1),
    ],
    "Safety SOP": [
        (r"\bsafety\s*(procedure|instruction|data\s*sheet)\b", 3),
        (r"\blockout[-/\s]?tagout\b|\bloto\b", 2), (r"\bhazard\b", 2),
        (r"\bppe\b", 1), (r"\bpermit\b", 1), (r"\bmsds\b|\bsds\b", 2),
    ],
    "Company Manual": [
        (r"\bcompany\s*manual\b", 3), (r"\basset\s*manual\b", 2),
    ],
    "Commissioning Report": [
        (r"\bcommissioning\b", 3), (r"\bpre-commissioning\b", 2),
        (r"\bacceptance test\b", 2),
    ],
    "Calibration Report": [
        (r"\bcalibration\b", 3), (r"\bas[- ]found\b", 2), (r"\bas[- ]left\b", 2),
    ],
    "Warranty": [
        (r"\bwarranty\b", 3), (r"\bguarantee\b", 2),
    ],
    "Datasheet": [
        (r"\bdatasheet\b", 3), (r"\btechnical data\b", 2),
        (r"\bdesign conditions\b", 2),
    ],
    "Maintenance Checklist": [
        (r"\bmaintenance checklist\b", 3), (r"\bchecklist\b", 2),
    ],
    "Vendor Report": [
        (r"\bvendor report\b", 3), (r"\bmanufacturer'?s report\b", 2),
    ],
    "Service Report": [
        (r"\bservice report\b", 3), (r"\bservice engineer\b", 2),
    ],
}

# Equipment-tag pattern (mirrors ingest.TAG_RE for asset-like tags).
_ASSET_TAG = re.compile(r"\b[A-Z]{1,3}-\d{2,3}\b")
# Prefixes that look like equipment tags but are parts/documents/permits.
_NON_ASSET_PREFIXES = frozenset({
    "BRG", "MS", "CE", "VG", "SEAL", "GKT", "BLT", "SOP", "SAF", "WO",
    "HS", "MCC", "PM", "RCA",
})


def _is_asset_tag(tag: str) -> bool:
    return tag.split("-", 1)[0].upper() not in _NON_ASSET_PREFIXES


def _score(text: str) -> dict[str, float]:
    lowered = text.lower()
    scores: dict[str, float] = {}
    for dtype, sigs in _SIGNATURES.items():
        total = 0.0
        for pattern, weight in sigs:
            if re.search(pattern, lowered):
                total += weight
        if total:
            scores[dtype] = total
    return scores


def classify(text: str, *, title: str = "") -> dict:
    """Classify a document from its text.

    Returns a dict with the winning ``type``, a ``confidence`` in [0, 1],
    ``maintenance_related`` (whether the type triggers the Asset360 prompt),
    the full ``scores`` map and the ranked ``ranking`` for transparency.
    """
    haystack = f"{title}\n{text}"
    scores = _score(haystack)

    if not scores:
        return {
            "type": "General Document", "confidence": 0.4,
            "maintenance_related": False, "scores": {},
            "ranking": [], "asset_tags": _asset_tags(haystack),
        }

    ranking = sorted(scores.items(), key=lambda kv: -kv[1])
    top_type, top_score = ranking[0]
    runner_up = ranking[1][1] if len(ranking) > 1 else 0.0

    # Confidence: how dominant the winner is over the field, capped to [0.4, 0.98].
    total = sum(scores.values()) or 1.0
    dominance = top_score / total
    separation = (top_score - runner_up) / (top_score or 1.0)
    confidence = round(min(0.98, 0.45 + 0.35 * dominance + 0.20 * separation), 2)

    return {
        "type": top_type,
        "confidence": confidence,
        "maintenance_related": top_type in MAINTENANCE_TYPES,
        "scores": {k: round(v, 1) for k, v in scores.items()},
        "ranking": [{"type": t, "score": round(s, 1)} for t, s in ranking],
        "asset_tags": _asset_tags(haystack),
    }


def classify_with_llm_verification(text: str, *, title: str = "") -> dict:
    """Use the configured LLM to verify the rule classification when one is
    explicitly configured.  The deterministic result remains the safe fallback
    so uploads never block on a provider outage."""
    result = classify(text, title=title)
    result["method"] = "rule-based fallback"
    if not (os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN") or os.getenv("HF_TOKEN")
            or os.getenv("ANTHROPIC_API_KEY")):
        return result
    try:
        import llm
        allowed = ", ".join(DOC_TYPES)
        raw = llm.complete(
            f"Classify this engineering document as exactly one of: {allowed}. "
            "Reply with the type only.\n\n"
            f"Title: {title}\nText:\n{text[:7000]}",
            "You are a precise industrial document classifier. Do not infer facts.")
        choice = (raw or "").strip().splitlines()[0].strip(" .:-`*")
        match = next((dtype for dtype in DOC_TYPES if choice.lower() == dtype.lower()), None)
        if match:
            result["type"] = match
            result["maintenance_related"] = match in MAINTENANCE_TYPES
            result["method"] = "LLM verified"
    except Exception:
        pass
    return result


def _asset_tags(text: str) -> list[str]:
    """Distinct equipment-style tags mentioned, most frequent first."""
    from collections import Counter
    counts = Counter(t for t in _ASSET_TAG.findall(text) if _is_asset_tag(t))
    return [tag for tag, _ in counts.most_common()]


def classify_corpus_doc(corpus, doc_no: str) -> dict:
    """Classify an already-ingested document by its chunks. Returns the same
    shape as ``classify`` plus the ``doc_no`` and document ``title``."""
    chunks = [c for c in corpus.chunks if c.doc_no == doc_no]
    meta = corpus.docs.get(doc_no, {})
    title = meta.get("title", doc_no)
    text = "\n".join(c.text for c in chunks)
    result = classify(text, title=title)
    result["doc_no"] = doc_no
    result["title"] = title
    return result

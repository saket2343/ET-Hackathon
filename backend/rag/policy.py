"""Confidence and retry policies — thresholds from config, none in code."""
from __future__ import annotations

from .interfaces import PolicyDecision


class ThresholdConfidencePolicy:
    """BaseConfidencePolicy:
        coverage >  confident            -> ANSWER
        caveat  <= coverage <= confident -> ANSWER_WITH_NOTE
        coverage <  caveat, retries left -> RETRY
        coverage <  caveat, exhausted    -> INSUFFICIENT (never hallucinate)
    """

    def __init__(self, cfg: dict, max_iterations: int):
        self.confident = float(cfg.get("confident", 0.90))
        self.caveat = float(cfg.get("caveat", 0.70))
        self.note = cfg.get("note", "")
        self.insufficient = cfg.get(
            "insufficient_message",
            "The available evidence is insufficient.")
        self.max_iterations = max_iterations

    def decide(self, coverage: float, iteration: int) -> PolicyDecision:
        if coverage > self.confident:
            return PolicyDecision("ANSWER")
        if coverage >= self.caveat:
            return PolicyDecision("ANSWER_WITH_NOTE", note=self.note)
        if iteration < self.max_iterations:
            return PolicyDecision(
                "RETRY",
                note=f"coverage {coverage:.2f} < {self.caveat} — retrieving again")
        return PolicyDecision("INSUFFICIENT", note=self.insufficient)


class BoundedRetryPolicy:
    """BaseRetryPolicy: stop when iterations are exhausted or retrieval
    stops finding NEW evidence (retrying with the same evidence is waste)."""

    def __init__(self, cfg: dict):
        self.max_iterations = int(cfg.get("max_iterations", 2))
        self.min_new_evidence = int(cfg.get("min_new_evidence", 1))

    def should_retry(self, iteration: int, new_evidence_count: int) -> bool:
        return (iteration < self.max_iterations
                and new_evidence_count >= self.min_new_evidence)

"""Config-driven query classification.

The taxonomy and its trigger patterns live in config.yaml — extending the
category set is a YAML edit. Rules are ordered: the first matching rule is
the primary label; every matching rule contributes to all_labels so the
retrieval planner can see mixed intent (e.g. comparison + coding).
"""
from __future__ import annotations

import re

from .interfaces import Classification


class RuleQueryClassifier:
    """BaseQueryClassifier: ordered high-precision regex rules."""

    def __init__(self, cfg: dict):
        self.default = cfg.get("default_label", "explanation")
        self.rules: list[tuple[str, list[re.Pattern]]] = [
            (rule["label"],
             [re.compile(p, re.IGNORECASE) for p in rule["patterns"]])
            for rule in (cfg.get("rules") or [])
        ]

    def classify(self, query: str) -> Classification:
        q = " ".join(query.split())
        matches: list[tuple[str, float]] = []
        for label, patterns in self.rules:
            n = sum(1 for rx in patterns if rx.search(q))
            if n:
                # more distinct pattern hits -> higher confidence, capped
                matches.append((label, min(0.95, 0.75 + 0.1 * (n - 1))))
        if not matches:
            return Classification(label=self.default, confidence=0.4,
                                  all_labels=[(self.default, 0.4)],
                                  method="default")
        return Classification(label=matches[0][0],
                              confidence=matches[0][1],
                              all_labels=matches,
                              method="rules")

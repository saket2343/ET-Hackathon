"""Query Understanding Pipeline.

    raw -> unicode/text normalization -> spell correction -> entity
    recognition -> classification -> decomposition -> evidence planning
    -> retrieval planning -> (ambiguity check)

Each stage is an injected component behind a Protocol; the pipeline only
sequences them and records a per-stage trace. Any stage may be a no-op
implementation — the pipeline degrades to identity, never blocks.
"""
from __future__ import annotations

import re
import time
import unicodedata

from .interfaces import (BaseEvidencePlanner, BaseQueryClassifier,
                         BaseRetrievalPlanner, BaseSpellCorrector,
                         QueryUnderstanding)


class UnicodeTextNormalizer:
    """BaseNormalizer: NFKC, smart-quote/dash folding, whitespace collapse.
    Intra-token punctuation (node.js, P-101, __init__) is preserved."""

    _FOLD = {"‘": "'", "’": "'", "“": '"', "”": '"',
             "–": "-", "—": "-", " ": " "}

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        for bad, good in self._FOLD.items():
            text = text.replace(bad, good)
        return " ".join(text.split()).strip()


class RegexEntityRecognizer:
    """Gazetteer entity recognition over corpus concepts + tag IDs."""

    _TAG = re.compile(r"\b[A-Z]{1,4}-\d+\b")

    def __init__(self, concepts: list[str]):
        # longest-first so "multi-head attention" wins over "attention"
        self._concepts = sorted(set(concepts), key=len, reverse=True)
        self._rx = [
            (c, re.compile(r"(?<![A-Za-z0-9])" + re.escape(c) +
                           r"(?![A-Za-z0-9])", re.IGNORECASE))
            for c in self._concepts
        ]

    def extract(self, text: str) -> list[str]:
        found: list[str] = []
        taken = " " * len(text)
        for concept, rx in self._rx:
            m = rx.search(text)
            if m and taken[m.start():m.end()].strip() == "":
                found.append(concept)
                taken = taken[:m.start()] + "#" * (m.end() - m.start()) \
                    + taken[m.end():]
        found.extend(self._TAG.findall(text))
        return list(dict.fromkeys(found))


class PatternDecomposer:
    """Comparison/conjunction decomposition into subqueries."""

    _BETWEEN = re.compile(
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$)", re.IGNORECASE)
    _VS = re.compile(r"^(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)",
                     re.IGNORECASE)

    def decompose(self, query: str, entities: list[str]) -> list[str]:
        m = self._BETWEEN.search(query) or self._VS.search(query)
        sides = None
        if m:
            sides = (m.group(1).strip(), m.group(2).strip())
        elif len(entities) >= 2:
            sides = (entities[0], entities[1])
        if not sides:
            return []
        a, b = sides
        return [f"What is {a}?", f"What is {b}?",
                f"{a} {b} difference comparison"]


class QueryUnderstandingPipeline:
    def __init__(
        self,
        normalizer: UnicodeTextNormalizer,
        spell: BaseSpellCorrector,
        entities: RegexEntityRecognizer,
        classifier: BaseQueryClassifier,
        decomposer: PatternDecomposer,
        evidence_planner: BaseEvidencePlanner,
        retrieval_planner: BaseRetrievalPlanner,
        ambiguity=None,
    ):
        self.normalizer = normalizer
        self.spell = spell
        self.entities = entities
        self.classifier = classifier
        self.decomposer = decomposer
        self.evidence_planner = evidence_planner
        self.retrieval_planner = retrieval_planner
        self.ambiguity = ambiguity

    def process(self, raw: str) -> QueryUnderstanding:
        trace: list[dict] = []

        def step(stage: str, **info):
            trace.append({"stage": stage, **info})

        t0 = time.monotonic()
        normalized = self.normalizer.normalize(raw)
        step("normalize", output=normalized)

        spell = self.spell.correct(normalized)
        corrected = spell.corrected
        step("spell", corrections=[
            f"{c.original} → {c.corrected} ({c.confidence})"
            for c in spell.corrections],
            confidence=spell.confidence, protected=spell.protected)

        ents = self.entities.extract(corrected)
        step("entities", entities=ents)

        cls = self.classifier.classify(corrected)
        step("classify", label=cls.label, confidence=cls.confidence,
             all=cls.all_labels)

        clar = self.ambiguity.detect(corrected) if self.ambiguity else None
        if clar:
            step("ambiguity", term=clar.term, options=clar.options)

        plan = self.retrieval_planner.plan(cls, corrected)
        step("retrieval_plan", legs=plan.legs, k=plan.k,
             strategy=plan.strategy, reason=plan.reason)

        subqueries = (self.decomposer.decompose(corrected, ents)
                      if plan.decompose else [])
        if subqueries:
            step("decompose", subqueries=subqueries)

        facets = self.evidence_planner.plan(corrected, cls, ents)
        step("evidence_plan", facets=[f.name for f in facets])

        step("done", total_ms=round((time.monotonic() - t0) * 1000, 1))
        return QueryUnderstanding(
            original=raw, normalized=normalized, corrected=corrected,
            spell=spell, entities=ents, classification=cls,
            subqueries=subqueries, facets=facets, plan=plan,
            clarification=clar, trace=trace,
        )

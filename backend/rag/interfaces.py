"""Component interfaces and data contracts for the modular RAG pipeline.

Every pipeline component is a typing.Protocol: implementations are injected,
never imported by the orchestrator directly (dependency inversion). All
tunables live in rag/config.yaml — components receive their config section
at construction and hold no hard-coded thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable


# ------------------------------------------------------------------ data

@dataclass
class Correction:
    original: str
    corrected: str
    distance: int
    confidence: float
    reason: str            # e.g. "edit-distance-1, corpus freq 214"


@dataclass
class SpellResult:
    original: str
    corrected: str
    corrections: list[Correction] = field(default_factory=list)
    confidence: float = 1.0      # min correction confidence, 1.0 if none
    protected: list[str] = field(default_factory=list)  # spans left alone


@dataclass
class Classification:
    label: str                   # primary category
    confidence: float
    all_labels: list[tuple[str, float]] = field(default_factory=list)
    method: str = "rules"


@dataclass
class EvidenceFacet:
    name: str                    # "limitations of prior approaches"
    probe_query: str             # query to retrieve this facet
    terms: list[str]             # terms whose presence marks it covered
    covered: bool | None = None
    matched_terms: list[str] = field(default_factory=list)


@dataclass
class RetrievalPlan:
    legs: list[str]              # e.g. ["dense", "bm25", "graph"]
    k: int
    leg_weights: dict[str, float] = field(default_factory=dict)
    decompose: bool = False
    strategy: str = "hybrid"
    reason: str = ""


@dataclass
class Clarification:
    term: str
    options: list[str]
    question: str


@dataclass
class QueryUnderstanding:
    original: str
    normalized: str
    corrected: str
    spell: SpellResult | None = None
    entities: list[str] = field(default_factory=list)
    classification: Classification | None = None
    subqueries: list[str] = field(default_factory=list)
    facets: list[EvidenceFacet] = field(default_factory=list)
    plan: RetrievalPlan | None = None
    clarification: Clarification | None = None
    trace: list[dict] = field(default_factory=list)


@dataclass
class PolicyDecision:
    action: str                  # ANSWER | ANSWER_WITH_NOTE | RETRY | INSUFFICIENT
    note: str = ""


# ------------------------------------------------------------- protocols

@runtime_checkable
class BaseNormalizer(Protocol):
    def normalize(self, text: str) -> str: ...


@runtime_checkable
class BaseSpellCorrector(Protocol):
    def correct(self, text: str) -> SpellResult: ...


@runtime_checkable
class BaseQueryClassifier(Protocol):
    def classify(self, query: str) -> Classification: ...


@runtime_checkable
class BaseEvidencePlanner(Protocol):
    def plan(self, query: str, classification: Classification,
             entities: list[str]) -> list[EvidenceFacet]: ...
    def verify(self, facets: list[EvidenceFacet],
               evidence_texts: list[str]) -> float: ...


@runtime_checkable
class BaseRetrievalPlanner(Protocol):
    def plan(self, classification: Classification,
             query: str) -> RetrievalPlan: ...


@runtime_checkable
class BaseRetriever(Protocol):
    def retrieve(self, query: str, plan: RetrievalPlan) -> list[dict]: ...


@runtime_checkable
class BaseFusionStrategy(Protocol):
    def fuse(self, legs: dict[str, list[tuple[int, float]]]) -> dict[int, float]: ...


@runtime_checkable
class BaseDiversifier(Protocol):
    def diversify(self, ranked: list, k: int) -> list: ...


@runtime_checkable
class BaseReranker(Protocol):
    def rerank(self, query: str, candidates: list) -> list: ...


@runtime_checkable
class BaseEvidenceClusterer(Protocol):
    def cluster(self, evidence: list[dict]) -> list[dict]: ...


@runtime_checkable
class BaseAnswerPlanner(Protocol):
    def outline(self, query: str, clusters: list[dict]) -> dict: ...


@runtime_checkable
class BaseClaimVerifier(Protocol):
    def verify(self, answer: str, evidence: list[dict]) -> dict: ...


@runtime_checkable
class BaseConfidencePolicy(Protocol):
    def decide(self, coverage: float, iteration: int) -> PolicyDecision: ...


@runtime_checkable
class BaseRetryPolicy(Protocol):
    def should_retry(self, iteration: int, new_evidence_count: int) -> bool: ...


# Generation is injected as a plain callable so tests can stub it and the
# production wiring can pass llm.synthesize without an adapter class.
Generator = Callable[[str, str], Optional[str]]   # (query, case_file) -> answer

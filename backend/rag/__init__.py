"""Modular RAG pipeline package.

build_pipeline() wires the whole system from config + the existing corpus:
components are constructed once, injected everywhere, and each is
replaceable behind its Protocol (see interfaces.py).
"""
from __future__ import annotations

from collections import Counter

from .classify import RuleQueryClassifier
from .config import load_config
from .interfaces import *  # noqa: F401,F403 — re-export contracts
from .orchestrate import (HybridIndexRetriever, IterativeController,
                          ValidatorVerifier)
from .plan import (ConceptAmbiguityDetector, TableRetrievalPlanner,
                   TemplateEvidencePlanner)
from .policy import BoundedRetryPolicy, ThresholdConfidencePolicy
from .spell import CorpusSpellCorrector
from .understand import (PatternDecomposer, QueryUnderstandingPipeline,
                         RegexEntityRecognizer, UnicodeTextNormalizer)


def corpus_vocabulary(chunks) -> Counter:
    """Token frequencies over the indexed corpus — the spell corrector's
    source of truth for what counts as a valid word."""
    import re
    vocab: Counter = Counter()
    for ch in chunks:
        vocab.update(
            t.lower() for t in re.findall(r"[A-Za-z]{3,}", ch.text))
    return vocab


def corpus_concepts(corpus) -> list[str]:
    out: set[str] = set()
    for meta in corpus.docs.values():
        out.update(meta.get("concepts") or [])
    for ch in corpus.chunks:
        out.update(ch.concepts or [])
    return sorted(out)


def build_understanding(corpus, config: dict | None = None):
    cfg = config or load_config()
    vocab = corpus_vocabulary(corpus.chunks)
    concepts = corpus_concepts(corpus)
    return QueryUnderstandingPipeline(
        normalizer=UnicodeTextNormalizer(),
        spell=CorpusSpellCorrector(vocab, cfg["spell"]),
        entities=RegexEntityRecognizer(concepts),
        classifier=RuleQueryClassifier(cfg["classifier"]),
        decomposer=PatternDecomposer(),
        evidence_planner=TemplateEvidencePlanner(cfg["evidence_facets"]),
        retrieval_planner=TableRetrievalPlanner(cfg["retrieval_plans"]),
        ambiguity=ConceptAmbiguityDetector(concepts, cfg["ambiguity"]),
    )


def build_pipeline(corpus, index, query_processor, validator,
                   generator, config: dict | None = None):
    cfg = config or load_config()
    retry = BoundedRetryPolicy(cfg["retry"])
    return IterativeController(
        understander=build_understanding(corpus, cfg),
        retriever=HybridIndexRetriever(index, query_processor),
        generator=generator,
        verifier=ValidatorVerifier(validator),
        policy=ThresholdConfidencePolicy(cfg["confidence_policy"],
                                         retry.max_iterations),
        retry=retry,
        evidence_planner=TemplateEvidencePlanner(cfg["evidence_facets"]),
    )

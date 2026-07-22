"""Pre-retrieval planning: evidence facets, retrieval strategy, ambiguity.

EvidencePlanner  — infers WHAT evidence the answer needs before retrieval,
                   then verifies after retrieval which facets were found.
RetrievalPlanner — maps query class -> retrieval strategy (legs, k,
                   decomposition), from the config table.
AmbiguityDetector— asks a clarifying question instead of retrieving when a
                   short query underspecifies among several corpus concepts
                   ("Explain attention" -> which attention?).
"""
from __future__ import annotations

import re
from collections import defaultdict

from .interfaces import (Clarification, Classification, EvidenceFacet,
                         RetrievalPlan)

_STOP = {
    "the", "a", "an", "of", "to", "for", "is", "are", "was", "were", "in",
    "on", "at", "and", "or", "what", "how", "why", "explain", "describe",
    "tell", "me", "about", "please", "need", "do", "does", "did", "which",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-]+", text.lower())
            if t not in _STOP]


def _present(term: str, blob: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(term.lower()) +
                     r"(?![a-z0-9])", blob) is not None


class TemplateEvidencePlanner:
    """BaseEvidencePlanner driven by per-class facet templates."""

    def __init__(self, cfg: dict):
        self.templates: dict[str, list[dict]] = cfg or {}

    def plan(self, query: str, classification: Classification,
             entities: list[str]) -> list[EvidenceFacet]:
        template = self.templates.get(
            classification.label, self.templates.get("default", []))
        a = entities[0] if entities else ""
        b = entities[1] if len(entities) > 1 else ""
        facets = []
        for spec in template:
            name = spec["name"].replace("{A}", a).replace("{B}", b)
            if "{A}" in spec["name"] and not a:
                continue
            if "{B}" in spec["name"] and not b:
                continue
            # facet terms = template markers + any entity bound into the name
            terms = list(spec.get("terms") or [])
            for ent in (a, b):
                if ent and ent.lower() in name.lower():
                    terms.append(ent)
            facets.append(EvidenceFacet(
                name=name,
                probe_query=f"{name} {a}".strip() if a else name,
                terms=terms,
            ))
        return facets

    def verify(self, facets: list[EvidenceFacet],
               evidence_texts: list[str]) -> float:
        """Mark each facet covered/missing; return facet coverage ratio."""
        if not facets:
            return 1.0
        blob = " ".join(evidence_texts).lower()
        covered = 0
        for facet in facets:
            if not facet.terms:          # unmarked facet: judge by name terms
                terms = _tokens(facet.name)
            else:
                terms = facet.terms
            matched = [t for t in terms if _present(t, blob)]
            facet.matched_terms = matched
            facet.covered = bool(matched)
            covered += facet.covered
        return round(covered / len(facets), 3)


class TableRetrievalPlanner:
    """BaseRetrievalPlanner: class -> plan lookup from config."""

    def __init__(self, cfg: dict):
        self.table = cfg or {}

    def plan(self, classification: Classification,
             query: str) -> RetrievalPlan:
        spec = self.table.get(classification.label,
                              self.table.get("default", {}))
        return RetrievalPlan(
            legs=list(spec.get("legs", ["dense", "bm25"])),
            k=int(spec.get("k", 6)),
            decompose=bool(spec.get("decompose", False)),
            strategy=str(spec.get("strategy", "hybrid")),
            reason=(f"class '{classification.label}' "
                    f"(conf {classification.confidence:.2f}) -> "
                    f"{spec.get('strategy', 'hybrid')}"),
        )


class ConceptAmbiguityDetector:
    """Clarify-instead-of-guess for underspecified queries.

    Built from the corpus concept list at startup: a token that appears
    inside several distinct multi-word concepts ("attention" in
    "self-attention", "multi-head attention", "cross-attention") is
    ambiguous when it arrives with almost no other context.
    """

    def __init__(self, concepts: list[str], cfg: dict):
        self.max_tokens = int(cfg.get("max_query_tokens", 3))
        self.min_options = int(cfg.get("min_options", 3))
        self.max_options = int(cfg.get("max_options", 6))
        self._by_token: dict[str, set[str]] = defaultdict(set)
        for concept in concepts:
            words = _tokens(concept)
            if len(words) < 2:
                continue                     # single words disambiguate nothing
            for w in words:
                self._by_token[w].add(concept)

    def detect(self, query: str) -> Clarification | None:
        toks = _tokens(query)
        if not toks or len(toks) > self.max_tokens:
            return None
        for tok in toks:
            options = sorted(self._by_token.get(tok, ()))
            # if the query already names one option fully, it's not ambiguous
            options = [o for o in options if o.lower() not in query.lower()]
            if len(options) >= self.min_options:
                shown = options[: self.max_options]
                return Clarification(
                    term=tok,
                    options=shown,
                    question=(f"Which {tok}? " +
                              " • ".join(shown)),
                )
        return None

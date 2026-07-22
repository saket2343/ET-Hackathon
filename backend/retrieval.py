"""AXON hybrid retrieval (MVP).

Per the design doc: dense (semantic) + sparse (BM25 exact-term) + graph
proximity, fused with Reciprocal Rank Fusion. The dense leg is TF-IDF cosine —
a dependency-free stand-in for the fine-tuned embedding model + Qdrant; the
sparse leg is BM25 — the stand-in for OpenSearch.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter, OrderedDict, defaultdict
try:
    from sentence_transformers import CrossEncoder
except ImportError:  # optional heavy dependency — degrade to RRF ordering
    CrossEncoder = None

from ingest import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\./·°]*")


def _singularize(w: str) -> str:
    """Conservative English plural -> singular so query and document tokens
    agree on number. Without it, a query for 'transformers' never matched a
    paper that only ever writes 'transformer' (28 vs 0 occurrences), so the
    Attention paper was invisible to a 'what are transformers?' query while a
    LangChain page that happened to use the plural won. Applied by _tokenize
    to BOTH sides, so even an imperfect stem still self-matches; the ss/us/is/
    ous guards protect class/status/analysis/famous, and the isalpha guard
    protects code-ish tokens (node.js, gpt-4, p-101)."""
    if not w.isalpha() or len(w) <= 3 or w.endswith(("ss", "us", "is", "ous")):
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if w.endswith("s"):
        return w[:-1]
    return w


def _tokenize(text: str) -> list[str]:
    """Shared tokenizer for index and queries. Hyphenated compounds also
    emit their alphabetic parts: the Attention paper writes 'self-attention'
    (one token, x23) and never bare 'self', so the query 'self attention'
    could not match it at all. Emitting ['self-attention', 'self',
    'attention'] on BOTH sides lets spaced queries meet hyphenated prose
    while exact compound matches still score highest. Non-alpha parts
    (P-101 -> 'p', '101') are skipped by the isalpha/length guards."""
    out: list[str] = []
    for t in _TOKEN_RE.findall(text.lower()):
        out.append(_singularize(t))
        if "-" in t:
            for part in t.split("-"):
                if len(part) >= 3 and part.isalpha():
                    out.append(_singularize(part))
    return out


# Validator cross-encoder: deliberately DECOUPLED from the retrieval
# reranker. The validator's SUPPORTED/PARTIAL thresholds (0.70/0.45) are
# calibrated to ms-marco MiniLM's wide sigmoid spread, and it scores
# claim x evidence pairs per answer (dozens of calls) where MiniLM's speed
# matters. Measured under bge-reranker-base: a supported claim scored 0.731
# vs 0.540 for a fabricated one — barely separated around those thresholds —
# and validation latency grew ~10x. Retrieval quality and validation
# calibration are different jobs; they get different models.
VALIDATOR_RERANKER_MODEL = os.getenv(
    "VALIDATOR_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
).strip()

_validation_reranker = None

# NLI entailment model for the validator's second stage. Relevance models
# cannot catch entity-swapped or plausible-but-wrong claims (measured:
# ms-marco scores "LangChain is a library for stateful workflows" 1.000
# against a chunk that says LANGGRAPH is — identical to the true claim).
# nli-deberta-v3-small separates the same five test cases perfectly
# (entailment 0.98/0.00/0.00/1.00/0.00) at ~190ms/pair CPU, so it runs
# only on the top evidence candidates of claims relevance would bless.
# Set env NLI_VALIDATOR_MODEL="" to disable.
NLI_VALIDATOR_MODEL = os.getenv(
    "NLI_VALIDATOR_MODEL", "cross-encoder/nli-deberta-v3-small"
).strip()

_nli_verifier = None
_nli_failed = False


def get_validation_reranker():
    """Cached cross-encoder for answer validation (claim-support scoring)."""
    global _validation_reranker
    if CrossEncoder is None:
        return None
    if _validation_reranker is None:
        print(f"Loading validation reranker: {VALIDATOR_RERANKER_MODEL}")
        _validation_reranker = CrossEncoder(VALIDATOR_RERANKER_MODEL)
    return _validation_reranker


def get_nli_verifier():
    """Cached NLI cross-encoder for entailment verification, or None when
    disabled/unavailable (the validator then behaves exactly as before)."""
    global _nli_verifier, _nli_failed
    if CrossEncoder is None or not NLI_VALIDATOR_MODEL or _nli_failed:
        return None
    if _nli_verifier is None:
        try:
            print(f"Loading NLI verifier: {NLI_VALIDATOR_MODEL}")
            _nli_verifier = CrossEncoder(NLI_VALIDATOR_MODEL)
        except Exception as exc:
            print(f"NLI verifier unavailable (non-fatal): {exc}")
            _nli_failed = True
            return None
    return _nli_verifier


def _sigmoid(x: float) -> float:
    """Squash an unbounded cross-encoder logit into (0, 1).

    CrossEncoder.predict() returns a raw logit (roughly -11..11 for
    ms-marco-MiniLM), not a similarity score. Every score derived from it
    (final ranking score, the UI's evidence 'relevance', and the diagnostics
    shown in the case file) must go through this before being displayed or
    divided by a max — otherwise negative logits stay deeply negative and a
    max-scaled 'relevance' comes out as -17.68 instead of a sane 0-1 value.
    validator.py already does this same conversion for its own semantic
    scoring; this makes retrieval.py consistent with it.
    """
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class HybridIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

        docs_tokens = [
            _tokenize(" ".join([
                chunk.summary,
                " ".join(chunk.keywords),
                " ".join(chunk.concepts),
                chunk.text,
            ]))
            for chunk in chunks
        ]

        # Per-document term frequencies and lengths, computed ONCE at index
        # build time and shared by TF-IDF construction and every BM25 query
        # (BM25 previously rebuilt a Counter per document on every query).
        self._doc_tf: list[Counter] = [Counter(toks) for toks in docs_tokens]
        self._doc_lens: list[int] = [len(toks) for toks in docs_tokens]

        self._doc_freq: Counter = Counter()
        for tf in self._doc_tf:
            self._doc_freq.update(tf.keys())

        self._n = len(chunks)
        self._avg_len = sum(self._doc_lens) / max(1, self._n)

        self._reranker = None
        self._tfidf: list[dict[str, float]] = [
            self._tfidf_vector(tf) for tf in self._doc_tf
        ]

        # Inverted indexes: term -> [(chunk_idx, tf)] for BM25 and
        # term -> [(chunk_idx, tfidf_weight)] for the dense leg, so both
        # retrievers walk only the postings of the query's terms instead of
        # scanning every chunk on every query.
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, tf in enumerate(self._doc_tf):
            for term, freq in tf.items():
                self._postings[term].append((i, freq))
        self._tfidf_postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, vec in enumerate(self._tfidf):
            for term, weight in vec.items():
                self._tfidf_postings[term].append((i, weight))

        # Per-chunk haystack for entity-coverage scoring, built once. Stored
        # in SINGULARIZED form (via _tokenize) so entity matching agrees on
        # number with the retrieval legs: a query entity 'transformers'
        # (singularized to 'transformer') must match a paper that only writes
        # 'transformer', otherwise entity fusion and document reranking demote
        # the very document the dense/BM25 legs just surfaced.
        self._entity_haystacks: list[str] = [
            " ".join(_tokenize(" ".join([
                chunk.text,
                chunk.summary or "",
                " ".join(chunk.keywords),
                " ".join(chunk.entities),
            ])))
            for chunk in chunks
        ]
        # Hyphen-stripped variant: a query entity 'multihead' must match
        # prose that writes 'multi-head' (and vice versa) — measured cost of
        # the miss was entity coverage 0.38 instead of 1.0 on on-topic chunks.
        self._entity_haystacks_dehyphenated: list[str] = [
            h.replace("-", "") for h in self._entity_haystacks
        ]

        # ------------------------------------------------------------------
        # Document-aware structures: per-document chunk ordering (used for
        # neighbor expansion / adjacent-chunk merging) and an LRU cache of
        # retrieval results (invalidated naturally on reingest, because a
        # fresh HybridIndex is built).
        # ------------------------------------------------------------------
        self._doc_positions: dict[str, list[int]] = defaultdict(list)
        for i, chunk in enumerate(chunks):
            self._doc_positions[chunk.doc_no].append(i)
        self._chunk_pos: dict[int, tuple[str, int]] = {}
        for doc_no, idxs in self._doc_positions.items():
            for pos, i in enumerate(idxs):
                self._chunk_pos[i] = (doc_no, pos)
        self._cache: OrderedDict = OrderedDict()
        self._cache_max = 128

    # Cap on the chunk-text portion of the reranker evidence string, so the
    # enriched (summary + keywords + concepts + text) input stays compact and
    # cross-encoder inference cost does not grow.
    RERANK_EVIDENCE_TEXT_LIMIT = 1500

    # ------------------------------------------------------------------
    # Entity-aware score fusion
    #
    # The cross-encoder alone over-weights generic terms ("difference",
    # "comparison") and under-weights the actual entities the user asked
    # about — e.g. a LangSmith chunk outranking a LangGraph overview on a
    # "LangChain vs LangGraph" question. The final ranking score is
    # therefore a fusion:
    #
    #     final = 0.6 × CrossEncoder + 0.2 × EntityCoverage + 0.2 × Metadata
    #
    # EntityCoverage = (matched query entities) / (total query entities),
    # so a chunk mentioning BOTH LangChain and LangGraph always outranks a
    # tangential chunk mentioning only one (or neither).
    # ------------------------------------------------------------------
    FUSION_CE_WEIGHT = 0.60
    FUSION_ENTITY_WEIGHT = 0.20
    FUSION_METADATA_WEIGHT = 0.20

    def _query_entity_terms(self, query_plan) -> list[str]:
        """Lower-cased, deduplicated entity terms from the query plan:
        extracted entities + both sides of a comparison (if any)."""
        if query_plan is None:
            return []
        terms: list[str] = []
        for e in (getattr(query_plan, "entities", None) or []):
            terms.append(str(e))
        for a in (getattr(query_plan, "comparison_aspects", None) or []):
            terms.append(str(a))
        out: list[str] = []
        seen: set[str] = set()
        for t in terms:
            # Singularize word-wise so terms match the singularized entity
            # haystack ('transformers' -> 'transformer'); proper nouns
            # without plurals ('langchain', 'langgraph') are unchanged.
            tl = " ".join(_singularize(w) for w in t.lower().strip().split())
            if len(tl) >= 3 and tl not in seen:
                seen.add(tl)
                out.append(tl)
        return out

    def _entity_coverage(
        self, idx: int, entity_terms: list[str],
    ) -> tuple[float, list[str]]:
        """IDF-weighted entity coverage score for one chunk:
        (weight of matched query entities) / (weight of all query entities),
        where an entity's weight is the corpus IDF of its rarest token.
        Matching considers chunk text, summary, keywords, and graph
        entities so an entity counted by ingest also counts here.

        IDF weighting matters on comparison questions: with equal weights,
        an entity that appears everywhere in the corpus (LangChain, in a
        LangChain book) counts the same as the discriminative one
        (LangGraph), so a chunk covering only the ubiquitous entity kept
        half credit. Now the rare entity dominates the denominator.

        Matches are token-boundary anchored, NOT substring: plain
        `t in haystack` credited "chain" against a LangChain chunk and
        "graph" against the word "paragraph", which handed tangential
        chunks (LangSmith, Vector Database) unearned entity coverage on
        comparison questions. re caches compiled patterns, so per-call
        re.search over the precomputed haystack stays cheap."""
        if not entity_terms:
            return 0.0, []
        haystack = self._entity_haystacks[idx]
        dehyph = self._entity_haystacks_dehyphenated[idx]
        matched = []
        matched_w = total_w = 0.0
        for t in entity_terms:
            w = self._entity_weight(t)
            total_w += w
            hit = re.search(
                r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])",
                haystack,
            ) or re.search(
                # hyphen-insensitive: 'multihead' <-> 'multi-head'
                r"(?<![a-z0-9])" + re.escape(t.replace("-", ""))
                + r"(?![a-z0-9])",
                dehyph,
            )
            if hit:
                matched.append(t)
                matched_w += w
        return (matched_w / total_w if total_w else 0.0), matched

    def _entity_weight(self, term: str) -> float:
        """Discriminativeness of an entity term: corpus IDF of its rarest
        token (the rare token carries the identity of a multi-word name)."""
        toks = _tokenize(term)
        if not toks:
            return 1.0
        return max(self._idf(t) for t in toks)

    def _fuse_scores(
        self,
        relevance: float,
        entity_score: float,
        metadata_boost: float,
    ) -> float:
        """Weighted fusion of cross-encoder relevance, entity coverage and
        (normalised) metadata boost into the final ranking score."""
        meta_n = max(0.0, min(1.0, (metadata_boost - 1.0)
                              / self.METADATA_BOOST_CAP))
        return (self.FUSION_CE_WEIGHT * max(0.0, min(1.0, relevance))
                + self.FUSION_ENTITY_WEIGHT * entity_score
                + self.FUSION_METADATA_WEIGHT * meta_n)

    def _rerank_evidence(self, chunk: Chunk) -> str:
        """Build the evidence string the cross-encoder scores against.

        Instead of raw chunk text alone, prepend the chunk's summary,
        keywords and concepts — signals ingest already computed — so the
        reranker can match a query against what the chunk is ABOUT, not just
        the exact words it happens to contain. Total context is kept compact
        by truncating the chunk-text portion."""
        parts = []
        summary = (chunk.summary or "").strip()
        if summary:
            parts.append(summary)
        if chunk.keywords:
            parts.append("Keywords: " + ", ".join(chunk.keywords[:10]))
        if chunk.concepts:
            parts.append("Concepts: " + ", ".join(chunk.concepts[:8]))
        parts.append(chunk.text[: self.RERANK_EVIDENCE_TEXT_LIMIT])
        return "\n".join(parts)

    # Retrieval cross-encoder checkpoint, overridable via env. Default is
    # bge-reranker-base: measured on this corpus, ms-marco MiniLM misjudged
    # in both directions (0.78 for a tangential LangSmith page on a
    # LangChain-vs-LangGraph question — ranked it #1; ~0.0 across the board
    # on academic prose — collapse). bge fixes both (LangSmith 1->3, paper
    # scores 0.5+) at ~1.6s warm per retrieve on CPU. Set RERANKER_MODEL to
    # the MiniLM checkpoint to trade quality back for speed.
    RERANKER_MODEL = os.getenv(
        "RERANKER_MODEL", "BAAI/bge-reranker-base"
    ).strip()

    def _get_reranker(self):
        """
        Lazy-load the cross-encoder only when reranking
        is required for the first time.
        """

        if CrossEncoder is None:
            return None

        if self._reranker is None:
            print(f"Loading reranker: {self.RERANKER_MODEL}")
            self._reranker = CrossEncoder(self.RERANKER_MODEL)

        return self._reranker

    def _idf(self, term: str) -> float:
        return math.log((self._n + 1) / (self._doc_freq.get(term, 0) + 1)) + 1

    def _tfidf_vector(self, tf: Counter) -> dict[str, float]:
        """L2-normalised log-TF/IDF vector — one formula shared by document
        vectors (built once in __init__) and query vectors (built per query
        in _dense), so the weighting scheme can never drift between them."""
        vec = {
            term: (1 + math.log(freq)) * self._idf(term)
            for term, freq in tf.items()
        }
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: value / norm for term, value in vec.items()}

    def _dense(self, query: str) -> list[tuple[int, float]]:
        # Postings walk: only chunks sharing at least one term with the
        # query are touched (the cosine of any other chunk is 0 anyway).
        qvec = self._tfidf_vector(Counter(_tokenize(query)))
        scores: dict[int, float] = defaultdict(float)
        for t, qw in qvec.items():
            for i, dw in self._tfidf_postings.get(t, ()):
                scores[i] += qw * dw
        return sorted(scores.items(), key=lambda x: -x[1])

    def _bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
        # Postings walk (term -> [(chunk, tf)]) instead of a full corpus scan.
        scores: dict[int, float] = defaultdict(float)
        for t, qtf in Counter(_tokenize(query)).items():
            postings = self._postings.get(t)
            if not postings:
                continue
            idf = math.log(1 + (self._n - self._doc_freq[t] + 0.5) / (self._doc_freq[t] + 0.5))
            for i, f in postings:
                dl = self._doc_lens[i]
                scores[i] += qtf * idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self._avg_len))
        return sorted(scores.items(), key=lambda x: -x[1])

    _STOPWORDS = {
        "what", "is", "are", "the", "a", "an", "of", "in", "on", "for", "to",
        "and", "or", "do", "does", "how", "i", "we", "me", "my", "it", "be",
        "can", "should", "need", "about", "with", "this", "that", "there",
        "use", "used", "why", "when", "which", "please", "tell",
    }




    def _semantic_rerank(
    self,
    query: str,
    candidates: list,
    top_n: int = 30,
) -> tuple[list, dict]:
        """
        Rerank candidate chunks using a cross-encoder.

        RRF is responsible for candidate generation.
        The cross-encoder is responsible for final
        query-to-chunk relevance ordering.

        Returns
        -------
        reranked:
            Same 3-item tuple structure used by the
            existing retrieval pipeline:

            (chunk_index, reranker_score, metadata_boost)

        reranker_scores:
            Mapping from chunk index to raw reranker score.
        """

        if not candidates:
            return [], {}

        # Limit expensive cross-encoder inference
        candidates = candidates[:top_n]

        reranker = self._get_reranker()

        if reranker is None:
            # No cross-encoder available: score by CONTENT, not by rank —
            # idf-weighted query-term coverage plus a prose factor, so an
            # outline/TOC page stuffed with query keywords no longer outranks
            # a passage that actually discusses them.
            reranked = [
                (idx, self._lexical_support(query, self.chunks[idx].text), boost)
                for idx, _score, boost in candidates
            ]
            reranked.sort(key=lambda item: -item[1])
            return reranked, {idx: s for idx, s, _ in reranked}

        # Richer evidence than raw chunk text: summary + keywords + concepts
        # + (truncated) text. The reranker sees what the chunk is about, not
        # just its surface wording; total context stays compact.
        pairs = [
            (
                query,
                self._rerank_evidence(self.chunks[idx]),
            )
            for idx, _, _ in candidates
        ]

        scores = reranker.predict(pairs)

        reranker_scores = {}
        reranked = []

        for (
            idx,
            _retrieval_score,
            metadata_boost,
        ), reranker_score in zip(
            candidates,
            scores,
        ):
            # Raw cross-encoder logit -> bounded (0, 1) relevance score.
            # Sorting/ranking is unaffected (sigmoid is monotonic), but every
            # downstream consumer (UI, diagnostics, case file) now sees a
            # sane, comparable number instead of an unbounded logit.
            score = _sigmoid(float(reranker_score))

            reranker_scores[idx] = score

            reranked.append(
                (
                    idx,
                    score,
                    metadata_boost,
                )
            )

        reranked.sort(
            key=lambda item: -item[1]
        )

        return reranked, reranker_scores

    def relevance(self, query: str) -> float:
        """Best dense similarity — used by the grounded-or-refuse gate."""
        hits = self._dense(query)
        return hits[0][1] if hits else 0.0

    @staticmethod
    def prose_factor(text: str) -> float:
        """How much of this chunk is actual prose vs scaffolding/boilerplate.
        1.0 = full sentences; ~0 = a contents page, a references list, or a
        paper title page. Dead giveaways, each forcing the factor to 0:
        dotted-leader TOC lines ('… 111'), bibliography entries ('[23] A.
        Author…'), and author/affiliation blocks (dense email lines) — words-
        per-line alone reads all three as prose, which let a title page and
        a references chunk outrank real content once the cross-encoder
        collapsed."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return 0.0
        toc_lines = sum(1 for l in lines if re.search(r"\.{3,}\s*\d+\s*$", l))
        if toc_lines / len(lines) > 0.3:
            return 0.0
        ref_lines = sum(1 for l in lines if re.match(r"\[\d+\]\s", l))
        if ref_lines / len(lines) > 0.25:
            return 0.0
        email_lines = sum(
            1 for l in lines if re.search(r"\S+@\S+\.\S+", l))
        if email_lines / len(lines) > 0.15:
            return 0.0
        return sum(1 for l in lines if len(l.split()) >= 8) / len(lines)

    def _lexical_support(self, query: str, text: str) -> float:
        """Content-grounded relevance in (0, 1): idf-weighted query-term
        coverage MULTIPLIED by the prose factor — a keyword-dense outline page
        can no longer ride coverage alone past a passage that actually
        discusses the terms."""
        qterms = [t for t in set(_tokenize(query)) if t not in self._STOPWORDS]
        if not qterms:
            return 0.5
        toks = set(_tokenize(text))
        total = sum(self._idf(t) for t in qterms)
        covered = sum(self._idf(t) for t in qterms if t in toks)
        coverage = covered / total if total else 0.0
        prose = self.prose_factor(text)
        return round(min(1.0, 0.08 + 0.92 * coverage * (0.30 + 0.70 * prose)), 4)

    def query_terms(self, query: str) -> list[str]:
        """The query's informative (non-stopword) vocabulary."""
        return [t for t in set(_tokenize(query)) if t not in self._STOPWORDS]

    @property
    def has_semantic_reranker(self) -> bool:
        """True when the cross-encoder is available — acceptance gating can
        then trust semantic scores and use lexical checks only as sanity."""
        return CrossEncoder is not None

    def dense_retrieve(self, query: str, k: int = 6) -> list[dict]:
        """Pure dense top-k across the whole corpus — the vanilla-RAG baseline:
        no BM25, no graph, no fusion, no document routing, no reranking."""
        results = []
        for i, score in self._dense(query)[:k]:
            c = self.chunks[i]
            results.append({"chunk_id": c.chunk_id, "doc_no": c.doc_no,
                            "doc_title": c.doc_title, "revision": c.revision,
                            "section": c.section, "text": c.text,
                            "score": round(score, 4)})
        return results

    def grounding(self, query: str) -> dict:
        """Term-coverage grounding signal: is the query's vocabulary — especially
        its most distinctive term — actually present in the corpus?"""
        toks = [t for t in set(_tokenize(query)) if t not in self._STOPWORDS]
        if not toks:
            return {"coverage": 0.0, "rarest_present": False}
        present = [t for t in toks if self._doc_freq.get(t, 0) > 0]
        rarest = min(toks, key=lambda t: self._doc_freq.get(t, 0))
        return {"coverage": len(present) / len(toks),
                "rarest_present": self._doc_freq.get(rarest, 0) > 0}

    # ------------------------------------------------------------------
    # Metadata Scoring
    # ------------------------------------------------------------------

    # Section-name fragments that are likely to answer each question intent.
    # Keys cover every intent QueryProcessor.detect_intent can emit (plus a
    # few forward-looking ones); fragments are matched as substrings of the
    # chunk's section name, and the boost stays bounded by METADATA_BOOST_CAP.
    _INTENT_SECTIONS = {
        "author": ("title", "author", "abstract"),
        "summary": ("abstract", "introduction", "overview", "conclusion"),
        "implementation": ("implementation", "example", "code", "tutorial",
                           "usage", "getting started"),
        "methodology": ("method", "approach", "architecture", "algorithm",
                        "pipeline", "workflow"),
        "comparison": ("comparison", "versus", "results", "evaluation",
                       "discussion", "trade-off"),
        "definition": ("abstract", "introduction", "overview", "glossary",
                       "definition", "terminology"),
        "reasoning": ("discussion", "analysis", "why", "rationale"),
        "results": ("results", "evaluation", "experiments", "benchmark"),
        "limitations": ("limitation", "discussion", "future work",
                        "drawback"),
        "conclusion": ("conclusion", "summary", "takeaway"),
        "equation": ("equation", "formula", "proof", "derivation", "math"),
        "figure": ("figure", "table", "diagram", "chart"),
        "debugging": ("troubleshoot", "debug", "error", "faq", "pitfall",
                      "common mistake"),
        "optimization": ("optimiz", "performance", "tuning", "efficiency",
                         "scaling"),
        "architecture": ("architecture", "design", "component", "overview",
                         "structure"),
    }

    def _section_matches_intent(self, chunk: Chunk, intent: str) -> bool:
        """True when the chunk's section is likely to answer the detected
        question intent (e.g. an 'Abstract' section for a summary question)."""
        section = (chunk.section or "").lower()
        return any(
            keyword in section
            for keyword in self._INTENT_SECTIONS.get(intent, ())
        )

    # Bounded-additive metadata boost configuration. Each signal contributes
    # an additive delta; the total is clamped so metadata can nudge ranking
    # but can never dominate the retrieval signal (the old multiplicative
    # scheme could compound to 2-3x and drown out dense/BM25 relevance).
    METADATA_BOOST_CAP = 0.50            # max total additive boost
    SECTION_INTENT_DELTA = 0.20          # section matches question intent
    EXPLICIT_SECTION_DELTA = 0.15        # query plan explicitly boosts section
    ENTITY_DELTA_PER_MATCH = 0.10        # per matched graph entity
    ENTITY_DELTA_MAX = 0.20
    KEYWORD_DELTA_PER_MATCH = 0.08       # per matched chunk keyword
    KEYWORD_DELTA_MAX = 0.16

    def _metadata_score(self, chunk: Chunk, query_plan, query: str) -> float:
        """
        Metadata-aware reranking (bounded additive).

        Combines intent-section, explicit-section, entity and keyword
        signals. Each contributes an additive delta and the sum is capped at
        METADATA_BOOST_CAP, so the returned multiplier is bounded in
        [1.0, 1.0 + METADATA_BOOST_CAP]. Metadata refines ranking; it can
        no longer override content relevance the way the old compounding
        multiplicative boosts (up to ~2.7x) could.
        """
        boost = 0.0

        # Section matches the detected question intent.
        if self._section_matches_intent(chunk, query_plan.intent):
            boost += self.SECTION_INTENT_DELTA

        # Entity overlap with the query plan's graph entities.
        if query_plan.boost_entities:
            entity_matches = len(
                set(chunk.entities).intersection(query_plan.boost_entities)
            )
            boost += min(
                self.ENTITY_DELTA_MAX,
                entity_matches * self.ENTITY_DELTA_PER_MATCH,
            )

        # Keyword overlap between the query and the chunk's keywords.
        keyword_matches = len(
            set(_tokenize(query)).intersection(chunk.keywords)
        )
        boost += min(
            self.KEYWORD_DELTA_MAX,
            keyword_matches * self.KEYWORD_DELTA_PER_MATCH,
        )

        # Sections the query plan explicitly asked to boost.
        if chunk.section:
            section = chunk.section.lower()
            if any(sec.lower() in section for sec in query_plan.boost_sections):
                boost += self.EXPLICIT_SECTION_DELTA

        return 1.0 + min(self.METADATA_BOOST_CAP, boost)


    # A single section may contribute at most this many chunks. Same-section
    # chunks are not discarded outright — they may hold complementary
    # evidence — but section diversity is preferred.
    MAX_PER_SECTION = 2

    # MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
    MMR_LAMBDA = 0.72

    # Evidence saturation: once at least MMR_MIN_SELECT chunks are chosen,
    # stop when even the best remaining candidate is a near-duplicate
    # (similarity to an already-selected chunk above MMR_SATURATION_SIM).
    # MMR always surfaces the LEAST redundant candidate first, so if that
    # one is a near-duplicate, everything left is too — filling k with
    # copies only wastes prompt budget. (A gain floor on the MMR score
    # itself cannot express this: with λ=0.72 the score stays positive even
    # for an exact duplicate.)
    MMR_SATURATION_SIM = 0.90
    MMR_MIN_SELECT = 3

    def _chunk_similarity(self, a: int, b: int) -> float:
        """TF-IDF cosine similarity between two indexed chunks — reuses the
        document vectors built at index time (no extra inference cost)."""
        va, vb = self._tfidf[a], self._tfidf[b]
        if len(vb) < len(va):
            va, vb = vb, va
        return sum(w * vb.get(t, 0.0) for t, w in va.items())

    def _diversity_filter(
        self,
        ranked: list[tuple[int, float, float]],
        k: int,
        max_per_doc: int | None = None,
    ) -> list[tuple[int, float, float]]:
        """Diversity-aware selection (MMR + per-doc/per-section caps).

        Greedy Maximal Marginal Relevance:

            MMR(c) = λ·relevance(c) − (1−λ)·max_sim(c, already selected)

        so a chunk that is nearly identical to one already chosen is pushed
        down in favour of complementary evidence, while hard caps still bound
        how many chunks one document (max_per_doc) or one section
        (MAX_PER_SECTION) may contribute."""
        if max_per_doc is None:
            max_per_doc = k

        # Normalise relevance to (0, 1] so the redundancy penalty operates
        # on a comparable scale regardless of the upstream score range.
        max_rel = max((item[1] for item in ranked), default=1.0) or 1.0

        pool = list(ranked)
        selected: list[tuple[int, float, float]] = []
        doc_counts: Counter = Counter()
        section_counts: Counter = Counter()

        while pool and len(selected) < k:
            best_item = None
            best_mmr = None
            best_redundancy = 0.0
            for item in pool:
                chunk = self.chunks[item[0]]
                section_key = (chunk.doc_no, chunk.section)
                if doc_counts[chunk.doc_no] >= max_per_doc:
                    continue
                if section_counts[section_key] >= self.MAX_PER_SECTION:
                    continue
                rel = item[1] / max_rel
                redundancy = max(
                    (self._chunk_similarity(item[0], s[0]) for s in selected),
                    default=0.0,
                )
                mmr = (self.MMR_LAMBDA * rel
                       - (1.0 - self.MMR_LAMBDA) * redundancy)
                if best_mmr is None or mmr > best_mmr:
                    best_mmr = mmr
                    best_item = item
                    best_redundancy = redundancy
            if best_item is None:
                break
            if (len(selected) >= min(self.MMR_MIN_SELECT, k)
                    and best_redundancy > self.MMR_SATURATION_SIM):
                break  # evidence saturated: only near-duplicates remain
            chunk = self.chunks[best_item[0]]
            selected.append(best_item)
            doc_counts[chunk.doc_no] += 1
            section_counts[(chunk.doc_no, chunk.section)] += 1
            pool.remove(best_item)

        return selected

    def _covers_term(self, idx: int, term: str) -> bool:
        """Token-boundary match of a (possibly multi-word) aspect against a
        chunk's haystack; falls back to requiring every informative token of
        the phrase when the exact phrase is absent."""
        hay = self._entity_haystacks[idx]
        if re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", hay):
            return True
        if re.search(
            r"(?<![a-z0-9])" + re.escape(term.replace("-", ""))
            + r"(?![a-z0-9])",
            self._entity_haystacks_dehyphenated[idx],
        ):
            return True
        toks = [t for t in _tokenize(term) if t not in self._STOPWORDS]
        return bool(toks) and all(
            re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", hay)
            for t in toks
        )

    def _ensure_aspect_coverage(
        self,
        selected: list[tuple[int, float, float]],
        pool: list[tuple[int, float, float]],
        aspects: list[str],
    ) -> list[tuple[int, float, float]]:
        """Coverage-aware selection repair: every comparison aspect must be
        represented in the final evidence. Relevance + MMR alone can fill k
        with chunks about one side only (three near-identical LangGraph
        chunks and nothing on LangChain). For each uncovered aspect, the
        best-ranked pool candidate covering it is swapped in for the weakest
        selected chunk whose own aspect coverage is redundant (every aspect
        it covers is covered by another selected chunk)."""
        if not aspects or not selected:
            return selected
        selected = list(selected)
        chosen = {item[0] for item in selected}

        def covered_by(item) -> set[str]:
            return {a for a in aspects if self._covers_term(item[0], a)}

        for aspect in aspects:
            if any(self._covers_term(item[0], aspect) for item in selected):
                continue
            candidate = next(
                (item for item in pool
                 if item[0] not in chosen
                 and self._covers_term(item[0], aspect)),
                None,
            )
            if candidate is None:
                continue  # corpus simply has nothing on this aspect
            # Victim: weakest selected chunk that uniquely covers nothing.
            victim = None
            for item in sorted(selected, key=lambda x: x[1]):
                mine = covered_by(item)
                others = set().union(*(
                    covered_by(o) for o in selected if o is not item
                )) if len(selected) > 1 else set()
                if mine <= others:
                    victim = item
                    break
            if victim is None:
                continue  # every selected chunk is load-bearing; keep as is
            selected[selected.index(victim)] = candidate
            chosen.discard(victim[0])
            chosen.add(candidate[0])
        selected.sort(key=lambda x: -x[1])
        return selected

    # ------------------------------------------------------------------
    # Adaptive top_k, dedup, and document-aware context expansion
    # ------------------------------------------------------------------

    NEIGHBOR_EXPAND_LIMIT = 1200   # only expand chunks smaller than this
    MERGED_TEXT_LIMIT = 2600       # cap on any merged evidence block

    # Phrases signalling a multi-part / comparative question. Shared by
    # adaptive_k() and _candidate_pool_size() so the two adaptive sizes can
    # never disagree about what counts as a complex query.
    _MULTIPART_MARKERS = (" and ", " compare", " versus ", " vs ",
                          " difference", " both ", " all ", " list ",
                          " steps", " explain", " summarize")

    def _query_complexity(self, query: str) -> tuple[int, bool]:
        """Complexity signals shared by adaptive_k and _candidate_pool_size:
        (informative token count, has a multi-part/comparative marker)."""
        toks = [t for t in _tokenize(query) if t not in self._STOPWORDS]
        ql = f" {query.lower()} "
        multipart = any(m in ql for m in self._MULTIPART_MARKERS)
        return len(toks), multipart

    def adaptive_k(self, query: str) -> int:
        """Choose top_k based on query complexity: short factoid questions
        need few chunks; multi-part / comparative questions need more."""
        n_toks, multipart = self._query_complexity(query)
        k = 4
        if n_toks > 6:
            k += 2
        if n_toks > 12:
            k += 2
        if multipart:
            k += 2
        if query.count("?") > 1:
            k += 1
        return max(3, min(10, k))

    def _dedup_results(self, results: list[dict]) -> list[dict]:
        """Drop near-duplicate chunks (>85% token containment) so the LLM
        context is not wasted on repeated passages."""
        kept: list[dict] = []
        seen: list[set] = []
        for r in results:
            toks = set(_tokenize(r["text"]))
            dup = False
            for prev in seen:
                inter = len(toks & prev)
                if inter and inter / max(1, min(len(toks), len(prev))) > 0.85:
                    dup = True
                    break
            if dup:
                continue
            kept.append(r)
            seen.append(toks)
        return kept

    def _merge_adjacent(self, results: list[dict]) -> list[dict]:
        """Merge selected chunks that are adjacent in the same document
        section into a single coherent evidence block."""
        def pos_key(r):
            return self._chunk_pos.get(r.get("chunk_index", -1), ("~", 10**9))

        by_position = sorted(results, key=pos_key)
        merged: list[dict] = []
        for r in by_position:
            idx = r.get("chunk_index")
            if merged and idx is not None:
                prev = merged[-1]
                pidx = prev.get("chunk_index")
                if pidx is not None:
                    d1, p1 = self._chunk_pos.get(pidx, (None, None))
                    d2, p2 = self._chunk_pos.get(idx, (None, None))
                    same_section = (self.chunks[pidx].section
                                    == self.chunks[idx].section)
                    if (d1 == d2 and p1 is not None and p2 == p1 + 1
                            and same_section
                            and len(prev["text"]) + len(r["text"])
                            < self.MERGED_TEXT_LIMIT):
                        prev["text"] = prev["text"].rstrip() + "\n\n" + r["text"].lstrip()
                        prev["merged_chunk_ids"] = (
                            prev.get("merged_chunk_ids", [prev["chunk_id"]])
                            + [r["chunk_id"]])
                        prev["score"] = max(prev["score"], r["score"])
                        prev["chunk_index"] = idx  # allow chains of merges
                        continue
            merged.append(r)
        merged.sort(key=lambda x: -x["score"])
        return merged

    def _expand_neighbors(self, results: list[dict]) -> list[dict]:
        """Document-aware expansion: prepend/append the neighboring chunks
        (chunk-1, chunk+1) from the same section so the LLM sees complete
        passages instead of fragments."""
        selected = {r.get("chunk_index") for r in results}
        for r in results:
            idx = r.get("chunk_index")
            if idx is None or len(r["text"]) > self.NEIGHBOR_EXPAND_LIMIT:
                continue
            doc_no, pos = self._chunk_pos.get(idx, (None, None))
            if doc_no is None:
                continue
            order = self._doc_positions[doc_no]
            section = self.chunks[idx].section
            before = after = ""
            if pos > 0 and order[pos - 1] not in selected:
                n = self.chunks[order[pos - 1]]
                if n.section == section:
                    before = n.text
            if pos < len(order) - 1 and order[pos + 1] not in selected:
                n = self.chunks[order[pos + 1]]
                if n.section == section:
                    after = n.text
            merged = "\n\n".join(p for p in (before, r["text"], after) if p)
            if len(merged) > self.MERGED_TEXT_LIMIT:
                merged = merged[: self.MERGED_TEXT_LIMIT]
            if merged != r["text"]:
                r["original_text"] = r["text"]
                r["text"] = merged
                r["context_expanded"] = True
        return results

    # ------------------------------------------------------------------
    # Retrieval pipeline stages
    #
    # retrieve() is an orchestration method; each stage below is a small,
    # independently readable step. None of these change retrieval logic —
    # they are extractions of the previous inline blocks.
    # ------------------------------------------------------------------

    def _graph_retrieval(
        self,
        graph_entities,
    ) -> list[tuple[int, int]] | None:
        """Stage: graph proximity leg — chunks scored by how many of the
        query's graph-neighborhood entities they mention. Returns None when
        no graph entities were supplied (leg is skipped, as before)."""
        if not graph_entities:
            return None
        entity_set = set(graph_entities)
        graph_hits = []
        for i, c in enumerate(self.chunks):
            score = sum(1 for entity in c.entities if entity in entity_set)
            if score > 0:
                graph_hits.append((i, score))
        graph_hits.sort(key=lambda x: -x[1])
        return graph_hits

    def _apply_doc_filter(
        self,
        rankings: list[list[tuple[int, float]]],
        doc_filter,
    ) -> list[list[tuple[int, float]]]:
        """Stage: restrict every ranking to chunks from allowed documents."""
        if doc_filter is None:
            return rankings
        allowed = {
            i for i, c in enumerate(self.chunks) if c.doc_no in doc_filter
        }
        return [
            [(i, s) for i, s in ranking if i in allowed]
            for ranking in rankings
        ]

    # Weighted RRF configuration. The graph leg scores by coarse integer
    # entity counts (many rank ties, tie order arbitrary), so it gets less
    # fusion credit than the content-grounded dense/BM25 legs. Fusion depth
    # covers the whole adaptive candidate pool (CANDIDATE_POOL_MAX) — the old
    # fixed depth of 20 silently starved pools larger than 20.
    RRF_K = 60
    RRF_DEPTH = 40
    RRF_LEG_WEIGHTS = {"dense": 1.0, "bm25": 1.0, "graph": 0.5}

    def _rrf_fusion(
        self,
        legs: dict[str, list[tuple[int, float]]],
    ) -> dict[int, float]:
        """Stage: weighted Reciprocal Rank Fusion across named retrieval legs."""
        rrf: dict[int, float] = defaultdict(float)
        for name, ranking in legs.items():
            weight = self.RRF_LEG_WEIGHTS.get(name, 1.0)
            for rank, (idx, _) in enumerate(ranking[: self.RRF_DEPTH]):
                rrf[idx] += weight / (self.RRF_K + rank)
        return rrf

    def _apply_metadata_boost(
        self,
        rrf: dict[int, float],
        query_plan,
        query: str,
    ) -> list[tuple[int, float, float]]:
        """Stage: apply the bounded metadata boost to fused scores and rank.
        Returns (chunk_index, boosted_score, metadata_boost) tuples sorted by
        boosted score."""
        if query_plan is None:
            # No query plan (retrieve() default): metadata boosting needs
            # intent/boost hints, so it degrades to a neutral 1.0 multiplier
            # instead of crashing on query_plan.intent.
            boosted = [(idx, score, 1.0) for idx, score in rrf.items()]
            boosted.sort(key=lambda x: -x[1])
            return boosted
        boosted = []
        for idx, score in rrf.items():
            chunk = self.chunks[idx]
            metadata_boost = self._metadata_score(chunk, query_plan, query)
            boosted.append((idx, score * metadata_boost, metadata_boost))
        boosted.sort(key=lambda x: -x[1])
        return boosted

    # Adaptive candidate pool bounds. The old pool was a fixed 30; now it
    # shrinks for simple factoid queries (less cross-encoder inference) and
    # grows modestly for complex multi-part queries (better recall). The cap
    # keeps worst-case reranking cost close to the old fixed pool.
    CANDIDATE_POOL_MIN = 16
    CANDIDATE_POOL_MAX = 40

    def _candidate_pool_size(self, query: str, k: int) -> int:
        """Stage: adaptive candidate pool sizing from query complexity,
        using the same complexity signals as adaptive_k()."""
        n_toks, multipart = self._query_complexity(query)
        pool = 20
        if n_toks > 6:
            pool += 6
        if n_toks > 12:
            pool += 6
        if multipart:
            pool += 8
        pool = max(pool, 3 * k)  # never starve a large adaptive k
        return max(self.CANDIDATE_POOL_MIN, min(self.CANDIDATE_POOL_MAX, pool))

    def _document_rerank(
        self,
        reranked: list[tuple[int, float, float]],
        k: int,
    ) -> list[tuple[int, float, float]]:
        """Stage: document-level reranking — keep only chunks from the
        documents with the strongest evidence. A document is scored by its
        two best chunks, not the sum over all of them: summing let a large
        document full of mediocre matches evict a small document holding the
        single best answer (fatal for comparisons whose second entity lives
        in a short document)."""
        doc_chunk_scores: dict[str, list[float]] = defaultdict(list)
        for idx, score, _ in reranked:
            doc_chunk_scores[self.chunks[idx].doc_no].append(score)
        doc_scores = {
            doc: sum(sorted(scores, reverse=True)[:2])
            for doc, scores in doc_chunk_scores.items()
        }
        ranked_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
        max_docs = min(3, max(1, k // 2))
        allowed_docs = {doc for doc, _ in ranked_docs[:max_docs]}
        return [
            item for item in reranked
            if self.chunks[item[0]].doc_no in allowed_docs
        ]

    # ------------------------------------------------------------------
    # Evidence Acceptance Gate
    #
    # Ranking PROPOSES candidates; this gate DECIDES whether each one is
    # strong enough to be sent to the LLM. It does not replace ranking —
    # it evaluates every candidate AFTER reranking, combining the signals
    # the pipeline already computed. Thresholds are configurable class
    # attributes; the gate is fully deterministic.
    # ------------------------------------------------------------------

    ACCEPTANCE_THRESHOLD = 0.15          # min combined score to accept
    ACCEPTANCE_WEIGHTS = {
        "reranker": 0.45,                # cross-encoder relevance (0..1)
        "dense": 0.15,                   # TF-IDF cosine (0..1)
        "bm25": 0.15,                    # BM25, saturated to (0..1)
        "graph": 0.10,                   # graph proximity, saturated
        "metadata": 0.05,                # bounded metadata boost, normalised
        "entity": 0.10,                  # entity overlap with graph context
    }
    BM25_SATURATION = 4.0                # bm25/(bm25+SAT) squash
    GRAPH_SATURATION = 3.0               # graph hits needed for full credit

    # Reranker-collapse floor: when even the BEST candidate's cross-encoder
    # score is below this, the CE is out-of-domain for this query/corpus
    # (ms-marco MiniLM scores dense academic prose near 0 across the board).
    # Its 0.45 acceptance weight then zeroes out and every candidate but the
    # top-1 anchor is rejected — the "only 1 evidence chunk" failure. In
    # that regime the gate redistributes the reranker's weight over the
    # remaining signals instead of letting a silent model mismatch veto
    # perfectly good lexical/dense/entity evidence.
    CE_COLLAPSE_FLOOR = 0.20

    def _acceptance_score(
        self,
        idx: int,
        rerank_score: float,
        metadata_boost: float,
        dense_scores: dict[int, float],
        bm25_scores: dict[int, float],
        graph_scores: dict[int, float],
        graph_entities,
        weights: dict | None = None,
    ) -> float:
        """Combined evidence-strength score in (0, 1) built exclusively from
        signals the pipeline already computed — no extra inference cost."""
        dense = min(1.0, dense_scores.get(idx, 0.0))
        bm25 = bm25_scores.get(idx, 0.0)
        bm25_n = bm25 / (bm25 + self.BM25_SATURATION)
        graph = graph_scores.get(idx, 0)
        graph_n = min(1.0, graph / self.GRAPH_SATURATION)
        meta_n = max(0.0, min(1.0, (metadata_boost - 1.0)
                              / self.METADATA_BOOST_CAP))
        entity_n = 0.0
        if graph_entities:
            overlap = len(set(self.chunks[idx].entities) & set(graph_entities))
            entity_n = min(1.0, overlap / self.GRAPH_SATURATION)
        w = weights or self.ACCEPTANCE_WEIGHTS
        return (w["reranker"] * max(0.0, min(1.0, rerank_score))
                + w["dense"] * dense
                + w["bm25"] * bm25_n
                + w["graph"] * graph_n
                + w["metadata"] * meta_n
                + w["entity"] * entity_n)

    def _apply_ce_collapse_fallback(
        self,
        query: str,
        reranked: list[tuple[int, float, float]],
        reranker_scores: dict[int, float],
    ) -> tuple[list[tuple[int, float, float]], dict[int, float], bool]:
        """Stage 5b: reranker-collapse fallback. When even the best candidate
        scores below CE_COLLAPSE_FLOOR the CE is out-of-domain for this
        query/corpus — SUBSTITUTE the content-grounded lexical-support signal
        (idf-weighted coverage × prose factor) for the collapsed CE, in both
        ranking and the acceptance gate. Substitution beats weight
        redistribution: redistribution let a paper's TITLE PAGE (authors,
        emails, copyright) through purely on entity coverage, while the prose
        factor inside lexical support correctly starves boilerplate."""
        best_ce = max(reranker_scores.values(), default=0.0)
        if not reranker_scores or best_ce >= self.CE_COLLAPSE_FLOOR:
            return reranked, reranker_scores, False
        print("⚠️ Reranker collapse: best cross-encoder score "
              f"{best_ce:.2f} < {self.CE_COLLAPSE_FLOOR} — substituting "
              "lexical-support (coverage × prose) for ranking and gating")
        lex = {
            idx: self._lexical_support(query, self.chunks[idx].text)
            for idx, _, _ in reranked
        }
        substituted = sorted(
            ((idx, lex[idx], boost) for idx, _, boost in reranked),
            key=lambda item: -item[1],
        )
        return substituted, lex, True

    def _evidence_acceptance_gate(
        self,
        ranked: list[tuple[int, float, float]],
        dense_scores: dict[int, float],
        bm25_scores: dict[int, float],
        graph_scores: dict[int, float],
        graph_entities,
        reranker_scores: dict[int, float] | None = None,
        relaxed: bool = False,
    ) -> tuple[list[tuple[int, float, float]], dict[int, dict]]:
        """Stage: evaluate every reranked candidate and reject obviously weak
        evidence before it reaches the LLM. The top-ranked candidate always
        passes (it anchors the answer — if even it is weak, the downstream
        grounded-or-refuse gate fires, exactly as before). Returns the
        accepted candidates plus per-chunk acceptance diagnostics.

        relaxed=True (retrieval scoped to ONE small document): completeness
        beats precision — the user named the document, so rejecting its
        chunks for weak query-term overlap produces incomplete answers.
        Measured failure: 'tell me the projects in the resume' kept 3 of 6
        resume chunks because the parts listing three of the five projects
        never use the word 'project'; the answer then omitted them."""
        accepted: list[tuple[int, float, float]] = []
        acceptance: dict[int, dict] = {}
        thr = 0.0 if relaxed else self.ACCEPTANCE_THRESHOLD
        reranker_scores = reranker_scores or {}
        for rank, candidate in enumerate(ranked):
            idx, score, metadata_boost = candidate
            # Use the PURE cross-encoder relevance for the gate's "reranker"
            # component (or the lexical-support substitute when the CE has
            # collapsed — stage 5b already swapped reranker_scores). After
            # entity-aware fusion the candidate's score already blends entity
            # coverage and metadata — feeding that in here would count both
            # signals twice.
            ce_score = reranker_scores.get(idx, score)
            a = self._acceptance_score(
                idx, ce_score, metadata_boost,
                dense_scores, bm25_scores, graph_scores, graph_entities,
            )
            if rank == 0:
                reason = "top-ranked anchor evidence"
            elif a >= thr:
                reason = (f"combined evidence signal {a:.2f} ≥ "
                          f"acceptance threshold {thr:.2f}")
            else:
                reason = None
            is_accepted = reason is not None
            acceptance[idx] = {
                "acceptance_score": round(a, 4),
                "accepted": is_accepted,
                "acceptance_reason": reason,
                "rejected_reason": None if is_accepted else (
                    f"combined evidence signal {a:.2f} below "
                    f"acceptance threshold {thr:.2f} — weak across "
                    f"reranker/dense/BM25/graph signals"
                ),
            }
            if is_accepted:
                accepted.append(candidate)
        return accepted, acceptance

    def _build_results(
        self,
        top: list[tuple[int, float, float]],
        rrf: dict[int, float],
        dense_scores: dict[int, float],
        bm25_scores: dict[int, float],
        graph_scores: dict[int, float],
        reranker_scores: dict[int, float],
        acceptance: dict[int, dict],
        entity_scores: dict[int, float] | None = None,
        matched_entities: dict[int, list[str]] | None = None,
    ) -> list[dict]:
        """Stage: turn selected (idx, score, boost) tuples into result dicts
        with full retrieval + acceptance diagnostics for explainability."""

        entity_scores = entity_scores or {}
        matched_entities = matched_entities or {}
        results = []

        for idx, final_score, metadata_boost in top:
            chunk = self.chunks[idx]
            methods = []
            reasons = []

            dense = dense_scores.get(idx, 0.0)
            bm25 = bm25_scores.get(idx, 0.0)
            graph = graph_scores.get(idx, 0)
            escore = entity_scores.get(idx, 0.0)
            ematched = matched_entities.get(idx, [])

            if dense > 0:
                methods.append("Dense")
                if dense > 0.60:
                    reasons.append("High semantic similarity")
            if bm25 > 0:
                methods.append("BM25")
                if bm25 > 3:
                    reasons.append("Strong keyword overlap")
            if graph > 0:
                methods.append("Graph")
                reasons.append("Matched graph neighborhood")
            if not methods:
                methods.append("RRF")
            if ematched:
                reasons.append(
                    "Covers query entities: " + ", ".join(ematched[:4])
                )

            gate = acceptance.get(idx, {})

            selected_because = (
                f"cross-encoder {reranker_scores.get(idx, 0.0):.2f}"
                + (f", entity coverage {escore:.2f}"
                   f" ({len(ematched)} matched)" if entity_scores else "")
                + f", metadata boost {metadata_boost:.2f}"
                + f" → final {final_score:.4f}"
            )

            results.append({
                "chunk_index": idx,
                "chunk_id": chunk.chunk_id,
                "doc_no": chunk.doc_no,
                "doc_title": chunk.doc_title,
                "revision": chunk.revision,
                "section": chunk.section,
                "text": chunk.text,
                "summary": chunk.summary,
                "keywords": chunk.keywords,
                "concepts": chunk.concepts,
                "entities": chunk.entities,
                "diversity_selected": True,
                # Retrieval diagnostics — full per-leg score breakdown.
                "dense_score": round(dense, 4),
                "bm25_score": round(bm25, 4),
                "graph_score": graph,
                "rrf_score": round(rrf.get(idx, 0.0), 4),
                "reranker_score": round(reranker_scores.get(idx, 0.0), 4),
                "entity_score": round(escore, 4),
                "matched_entities": ematched,
                "score": round(final_score, 4),  # final fused ranking score
                "retrieval_method": " + ".join(methods),
                "metadata_score": round(metadata_boost, 2),
                "retrieval_reason": "; ".join(reasons) if reasons
                else "elected by RRF fusion and cross-encoder reranking",
                "selected_because": selected_because,
                # Acceptance-gate diagnostics (explainability only).
                "acceptance_score": gate.get("acceptance_score", 0.0),
                "accepted": gate.get("accepted", True),
                "acceptance_reason": gate.get("acceptance_reason"),
                "rejected_reason": gate.get("rejected_reason"),
            })

        return results

    def _postprocess_results(self, results: list[dict]) -> list[dict]:
        """Stage: document-aware post-processing:
          1. drop near-duplicate chunks,
          2. merge adjacent chunks from the same section,
          3. expand remaining fragments with neighboring chunks."""
        results = self._dedup_results(results)
        results = self._merge_adjacent(results)
        results = self._expand_neighbors(results)
        return results

    def retrieve(
        self,
        query,
        query_plan=None,
        graph_entities=None,
        k=6,
        doc_filter=None,
    ):
        """
        Hybrid Retrieval — pipeline orchestration.

        Dense Semantic Search
            +
        BM25 Exact Match
            +
        Graph Expansion
            ↓
        Reciprocal Rank Fusion (RRF)
            ↓
        Metadata boost (bounded additive)
            ↓
        Cross-encoder reranking (adaptive candidate pool)
            ↓
        Evidence Acceptance Gate
            ↓
        Document rerank → diversity filter → results → post-processing

        Returns retrieval + acceptance diagnostics for explainability.
        """

        # Adaptive top_k: never shrink below what the caller asked for, but
        # expand for complex / multi-part queries.
        k = max(k or 0, self.adaptive_k(query))

        # Query-level LRU cache (retrieval is deterministic per index build).
        # The query plan participates in scoring (metadata boost, entity
        # fusion), so its identifying fields are part of the key — otherwise
        # two calls sharing a retrieval query but carrying different plans
        # (different intent / entities / boost sections) would collide.
        plan_key = None
        if query_plan is not None:
            plan_key = (
                getattr(query_plan, "intent", None),
                tuple(getattr(query_plan, "entities", None) or ()),
                tuple(getattr(query_plan, "comparison_aspects", None) or ()),
                tuple(getattr(query_plan, "boost_sections", None) or ()),
                tuple(getattr(query_plan, "boost_entities", None) or ()),
            )
        cache_key = (
            query,
            k,
            tuple(sorted(doc_filter)) if doc_filter else None,
            tuple(sorted(graph_entities)) if graph_entities else None,
            plan_key,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return [dict(r) for r in cached]

        # ---- Stage 1: retrieval legs + document filter -------------------
        # Legs are kept as NAMED variables (not positions in a list) so the
        # per-leg diagnostic scores below can never be mis-assigned when a
        # leg is absent.
        dense_hits, bm25_hits, graph_hits = self._apply_doc_filter(
            [
                self._dense(query),
                self._bm25(query),
                self._graph_retrieval(graph_entities) or [],
            ],
            doc_filter,
        )

        # ---- Stage 2: preserve per-leg scores for diagnostics ------------
        dense_scores = dict(dense_hits)
        bm25_scores = dict(bm25_hits)
        graph_scores = dict(graph_hits)

        # ---- Stage 3: weighted reciprocal rank fusion ----------------------
        legs = {"dense": dense_hits, "bm25": bm25_hits}
        if graph_hits:
            legs["graph"] = graph_hits
        rrf = self._rrf_fusion(legs)

        # The CLEAN question — used for judging RELEVANCE (cross-encoder,
        # metadata keywords). `query` is the rewritten/expanded string whose
        # appended scaffolding ("definition overview comparison") exists to
        # widen LEXICAL recall; feeding it to the relevance scorers actively
        # misranks. Measured on "key difference between LangChain and
        # LangGraph" (cross-encoder, LangGraph-definition vs LangSmith):
        #     raw query       -> 0.703 vs 0.516  (gap +0.187)
        #     rewritten query -> 0.630 vs 0.505  (gap +0.125, -33%)
        # and the appended word "comparison" matched LangSmith's ingest
        # keywords, handing it metadata boost 1.36 vs 1.26 — enough to
        # overturn the cross-encoder and rank a tangential LangSmith page
        # above the actual LangGraph definition.
        judge_query = getattr(query_plan, "original_query", None) or query

        # ---- Stage 4: bounded metadata boost ------------------------------
        boosted = self._apply_metadata_boost(rrf, query_plan, judge_query)

        # ---- Stage 5: adaptive candidate pool + cross-encoder rerank ------
        pool_size = self._candidate_pool_size(query, k)
        reranked, reranker_scores = self._semantic_rerank(
            judge_query, boosted, top_n=pool_size,
        )

        # ---- Stage 5b: reranker-collapse fallback --------------------------
        reranked, reranker_scores, _ce_collapsed = (
            self._apply_ce_collapse_fallback(
                judge_query, reranked, reranker_scores)
        )

        # ---- Stage 6: entity-aware score fusion ---------------------------
        # final = 0.6×CrossEncoder + 0.2×EntityCoverage + 0.2×MetadataBoost.
        # A chunk that mentions BOTH comparison entities now outranks a
        # tangential chunk that merely matches generic query wording.
        entity_terms = self._query_entity_terms(query_plan)
        entity_scores: dict[int, float] = {}
        matched_entities: dict[int, list[str]] = {}
        if entity_terms:
            fused = []
            for idx, rel, metadata_boost in reranked:
                escore, ematched = self._entity_coverage(idx, entity_terms)
                entity_scores[idx] = round(escore, 4)
                matched_entities[idx] = ematched
                fused.append(
                    (idx, self._fuse_scores(rel, escore, metadata_boost),
                     metadata_boost)
                )
            fused.sort(key=lambda item: -item[1])
            reranked = fused

        # ---- Stage 7: evidence acceptance gate ----------------------------
        # Ranking proposed; the gate disposes. Weak candidates are rejected
        # here (with recorded reasons) before document reranking so they can
        # never be sent to the LLM.
        single_small_doc = (
            doc_filter is not None and len(doc_filter) == 1
            and len(self._doc_positions.get(next(iter(doc_filter)), [])) <= 12
        )
        reranked, acceptance = self._evidence_acceptance_gate(
            reranked,
            dense_scores, bm25_scores, graph_scores, graph_entities,
            reranker_scores=reranker_scores,
            relaxed=single_small_doc,
        )

        # ---- Stage 8: document-level reranking ----------------------------
        reranked = self._document_rerank(reranked, k)

        # ---- Stage 9: adaptive document diversity -------------------------
        # If retrieval is explicitly focused on one document, allow all
        # selected evidence to come from that document.
        if doc_filter and len(doc_filter) == 1:
            max_per_doc = k
        else:
            max_per_doc = min(4, k)
        top = self._diversity_filter(reranked, k=k, max_per_doc=max_per_doc)

        # ---- Stage 9b: coverage-aware selection repair ---------------------
        # Comparison questions must end up with evidence for BOTH sides (and
        # ideally a chunk covering both); relevance + MMR alone cannot
        # guarantee that.
        aspects = [
            a.lower().strip()
            for a in (getattr(query_plan, "comparison_aspects", None) or [])
            if a and len(a.strip()) >= 3
        ]
        if aspects:
            top = self._ensure_aspect_coverage(top, reranked, aspects)

        # ---- Stage 10: build result dicts with diagnostics ----------------
        results = self._build_results(
            top, rrf,
            dense_scores, bm25_scores, graph_scores,
            reranker_scores, acceptance,
            entity_scores, matched_entities,
        )

        # ---- Stage 11: document-aware post-processing ---------------------
        results = self._postprocess_results(results)

        self._cache[cache_key] = [dict(r) for r in results]
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

        return results
    

    

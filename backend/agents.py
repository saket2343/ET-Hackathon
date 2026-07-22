"""AXON multi-agent system (MVP).

Supervisor -> Planner -> [Predictive, RootCause, Knowledge/Engineering,
Maintenance, Safety] -> Risk/Decision -> Critic, per the design doc's
supervisor-worker graph. Each worker returns structured findings plus the
evidence that grounds them; the Critic computes a groundedness/confidence
score and gates the release.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import llm
import memory as convmem
import predictive as predictive_mod
import response_engine
from ingest import Corpus, DATA_DIR, TAG_RE
from kg import KnowledgeGraph
from retrieval import HybridIndex
from query_processor import QueryProcessor
from validator import AnswerValidator
from conversation import ConversationContext
from reasoning_trace import ReasoningTrace

_FOLLOWUP_HEADING = re.compile(
    r"#{1,4}\s*Suggested\s+Follow-?up\s+Questions\s*\n(.*)$",
    re.IGNORECASE | re.DOTALL)


def _extract_followups(answer: str,
                       entities: list[str] | None = None,
                       anchor: str | None = None) -> list[str]:
    """Structured follow-up suggestions for the UI (Module 02, req. 9).

    Prefers the model's own "Suggested Follow-up Questions" section; if the
    model omitted it, synthesizes a few grounded next questions from the
    query's entities/anchor so every answer still ends with useful next steps.
    """
    out: list[str] = []
    if answer:
        m = _FOLLOWUP_HEADING.search(answer)
        if m:
            for line in m.group(1).splitlines():
                line = line.strip().lstrip("-*•").strip()
                line = re.sub(r"^\d+[.)]\s*", "", line)
                if line.endswith("?") or (line and len(line.split()) >= 3):
                    out.append(line)
                if len(out) >= 4:
                    break
    if out:
        return out
    # Deterministic fallback — never leave the user without next steps.
    subject = anchor or (entities[0] if entities else None)
    if subject:
        out = [f"What causes issues with {subject}?",
               f"Show the maintenance history for {subject}.",
               f"Which documents reference {subject}?"]
    return out


class EvidenceOrganizer:
    """
    Organize retrieved evidence before it is passed to the LLM.

    Goals
    -----
    1. Remove duplicate chunks
    2. Group by document
    3. Group by section
    4. Sort by retrieval score
    5. Keep only the strongest evidence
    """

    MAX_CHUNKS_PER_SECTION = 2

    def organize(self, evidence: list[dict]) -> list[dict]:

        documents = defaultdict(lambda: defaultdict(list))

        seen = set()

        for chunk in evidence:

            key = (
                chunk["doc_no"],
                chunk["section"],
                chunk["text"][:200],
            )

            if key in seen:
                continue

            seen.add(key)

            documents[
                chunk["doc_no"]
            ][
                chunk["section"]
            ].append(chunk)

        organized = []

        for doc_name, sections in documents.items():

            doc_sections = []

            for section_name, chunks in sections.items():

                # Order by the FINAL fused ranking score (cross-encoder +
                # entity coverage + metadata). rrf_score is a pre-rerank
                # candidate-generation signal; sorting the prompt by it
                # undid the reranking layer's work.
                chunks.sort(
                    key=lambda x: x.get("score", 0),
                    reverse=True,
                )

                doc_sections.append({

                    "section": section_name,

                    "chunks": chunks[: self.MAX_CHUNKS_PER_SECTION]

                })

            doc_sections.sort(
                key=lambda s: max(
                    (c.get("score", 0) for c in s["chunks"]),
                    default=0,
                ),
                reverse=True,
            )

            organized.append({

                "document": doc_name,

                "sections": doc_sections,

            })

        # Strongest document first — evidence order in the prompt should
        # follow final relevance, not corpus iteration order.
        organized.sort(
            key=lambda d: max(
                (c.get("score", 0)
                 for s in d["sections"] for c in s["chunks"]),
                default=0,
            ),
            reverse=True,
        )

        return organized
    




class AnswerPlanner:
    """
    Build a response outline before asking the LLM to write.

    The planner never generates text.
    It only decides WHAT should be covered.
    """

    DEFAULT_STRUCTURE = [
        "Executive Summary",
        "Direct Answer",
        "Detailed Explanation",
        "Real-World Example",
        "Evidence from Uploaded Documents",
        "Limitations",
        "Sources",
        "Suggested Follow-up Questions",
    ]

    def build(
        self,
        query: str,
        query_plan,
        evidence: list[dict],
    ) -> dict:

        structure = list(self.DEFAULT_STRUCTURE)

        # Extract terms from the actual user query
        query_terms = {
            token.lower()
            for token in re.findall(
                r"\b[a-zA-Z0-9_.-]+\b",
                query,
            )
        }

        must_include = set()

        for chunk in evidence:

            candidates = (
                chunk.get("entities", [])
                + chunk.get("concepts", [])
                + chunk.get("keywords", [])
            )

            for item in candidates:

                item_lower = str(item).lower()

                # Keep only evidence terms related to the user's query
                if any(
                    term in item_lower
                    or item_lower in term
                    for term in query_terms
                ):
                    must_include.add(item)

        # Intent-specific additions

        if query_plan.intent == "comparison":

            structure.insert(
                2,
                "Comparison Table",
            )

        elif query_plan.intent == "author":

            structure.insert(
                1,
                "Authors",
            )

        elif query_plan.intent == "results":

            structure.insert(
                2,
                "Experimental Results",
            )

        elif query_plan.intent == "limitations":

            structure.insert(
                3,
                "Limitations",
            )

        return {

            "structure": structure,

            "must_include": sorted(must_include),

            "avoid": [
                "Hallucination",
                "Unsupported claims",
                "Speculation",
            ],

        }


class AgentSystem:
    def __init__(
        self,
        corpus: Corpus,
        graph: KnowledgeGraph,
        index: HybridIndex,
    ):
        self.corpus = corpus
        self.graph = graph
        self.index = index
        # Validation uses its own calibrated/fast cross-encoder — NOT the
        # retrieval reranker (bge), whose score distribution doesn't match
        # the validator's SUPPORTED/PARTIAL thresholds and whose per-pair
        # latency is too high for claim-level scoring.
        from retrieval import get_nli_verifier, get_validation_reranker
        self.validator = AnswerValidator(
            semantic_model=get_validation_reranker(),
            nli_model=get_nli_verifier(),
        )

        self.query_processor = QueryProcessor()
        self._retrieval_focus: dict | None = None
        self._retrieval_diagnostics: dict | None = None
        self.context = ConversationContext()
        # Corpus-grounded spell correction (rag package). Optional: a
        # failure to build it must never block boot — queries then flow
        # uncorrected, exactly as before.
        try:
            from rag import corpus_vocabulary
            from rag.config import load_config
            from rag.spell import CorpusSpellCorrector
            self.spell = CorpusSpellCorrector(
                corpus_vocabulary(corpus.chunks), load_config()["spell"])
        except Exception as exc:
            print(f"Spell corrector unavailable (non-fatal): {exc}")
            self.spell = None
    # ---------------------------------------------------------------- workers

    def _supervisor(self, query: str) -> dict:
        entities = sorted(set(TAG_RE.findall(query.upper())))
        anchor = next((e for e in entities if e in self.graph.nodes), None)
        # default demo asset if the query names none
        if anchor is None and re.search(r"pump|vibrat|bearing|p-?101", query, re.I):
            anchor = "P-101"
        return {"entities": entities, "anchor": anchor,
                "target_docs": self._target_documents(query, entities)}

    # A BM25 hit at/above this score is a confident lexical match; below it
    # the query is too vague to overturn conversation-based routing.
    FOLLOWUP_SHIFT_MIN_SCORE = 4.0
    # The best inherited-doc hit must fall below this fraction of the
    # query's best hit for the topic to count as changed. Measured shift
    # cases sit near 0.61-0.62 ("what are transformers" 4.74/7.83;
    # "HashRing KV" 7.3/11.8); a genuine follow-up scores 1.0 because its
    # top hit IS inside the inherited docs.
    FOLLOWUP_SHIFT_MARGIN = 0.8

    def _followup_topic_shift(self, query: str,
                              inherited_docs: set[str]) -> bool:
        """True when the query's own corpus evidence CONSENSUS lives outside
        the documents inherited from the conversation — the user changed
        topics, so follow-up routing must not fire.

        Without this guard, 'what is transformer' asked after LangChain
        questions inherited the LangChain PDFs as target_docs, retrieval was
        scoped to them, and the Attention paper (which holds the answer) was
        excluded entirely — the same query worked in a fresh chat.

        The test is consensus over the top-k hits, NOT a best-score ratio:
        measured on the failing query, the paper held ALL top-6 BM25 slots
        yet a ratio test failed (common words like 'what'/'are' inflate a
        LangChain chunk to 0.61 of the best score). If not one of the top-k
        confident hits falls inside the inherited docs, the query is about
        something else. A genuine follow-up ('explain more about memory')
        keeps inheritance — its top hits include the inherited docs — and a
        vague one ('explain more') keeps it via the score floor."""
        if not inherited_docs:
            return False
        hits = self.index._bm25(query)
        if not hits or hits[0][1] < self.FOLLOWUP_SHIFT_MIN_SCORE:
            return False                      # vague query — trust context
        best_score = hits[0][1]
        top_doc = self.index.chunks[hits[0][0]].doc_no
        best_inside = max(
            (s for i, s in hits
             if self.index.chunks[i].doc_no in inherited_docs),
            default=0.0,
        )
        # The query's OWN best evidence decides. A shift means: the single
        # strongest hit lies outside the inherited documents AND clearly
        # beats anything inside them.
        #
        # The previous rule — "no inherited-doc hit anywhere in the top-5" —
        # could essentially never fire: this corpus is ~90% LangChain, so
        # almost every query has LangChain chunks near the top. Measured on
        # "explain about the project HashRing KV" after a LangChain turn:
        # the resume led at 11.8 with four LangChain chunks behind it, so
        # `inside == 0` was False, the LangChain docs were inherited, and
        # the resume — the only document that mentions HashRing — was
        # excluded from retrieval entirely.
        return (top_doc not in inherited_docs
                and best_inside < self.FOLLOWUP_SHIFT_MARGIN * best_score)

    def _graph_reasoning(self, query: str, entities: list[str]) -> dict:
        """Graph-first step: detect which graph entities/concepts the query names,
        then TRAVERSE the graph to expand them into a connected context subgraph.
        The expanded nodes seed the retrieval graph-leg and light up the UI —
        this is the knowledge graph acting as the system of record, with vectors
        indexing into it (design §4.1)."""
        ql = query.lower()
        seeds: set[str] = set()
        for e in entities:                                  # named equipment tags
            if e in self.graph.nodes:
                seeds.add(e)
        qnorm = re.sub(r"[^a-z0-9]", "", ql)
        qtokens5 = {w for w in re.findall(r"[a-z][a-z0-9]{4,}", ql)}  # words >= 5 chars
        qtokens3 = {t for t in re.findall(r"[a-z0-9]{3,}", ql)}
        for nid, nd in self.graph.nodes.items():            # named concepts
            if nd.get("type") != "Concept" or len(nid) < 4:
                continue
            nl = nid.lower()
            # exact/plural/separator-tolerant match, or a query word contained in
            # the concept name (so "Runnable" seeds RunnableLambda / …)
            if (self._concept_in_query(nid, ql, qnorm, qtokens3)
                    or any(w in nl for w in qtokens5)):
                seeds.add(nid)
        if not seeds:
            return {"seeds": [], "nodes": set(), "edges": [], "paths": []}
        # Relevance-weighted expansion: causally-linked nodes outrank loose
        # associations, and the neighborhood is bounded so retrieval focus
        # doesn't dilute on large graphs.
        node_scores, edges = self.graph.weighted_expand(seeds, hops=2)
        nodes = set(node_scores)
        # human-readable multi-hop paths from each seed (for the reasoning trace)
        paths = []
        for s in sorted(seeds)[:4]:
            hops = [f"{r}→{m}" for m, r in self.graph.neighbors(s)][:4]
            if hops:
                paths.append(f"{s}: " + ", ".join(hops))
        # plain-language summary (Issue 7 — judges shouldn't read RELATED_TO)
        related: list[str] = []
        for s in sorted(seeds):
            for m, r in self.graph.neighbors(s, "RELATED_TO"):
                if m not in seeds and m not in related:
                    related.append(m)
        summary = f"Detected {', '.join(sorted(seeds)[:3])}"
        if related:
            summary += f"; expanded via the graph to related concepts: {', '.join(related[:5])}"
        summary += f". Used {len(nodes)} connected nodes to focus retrieval."
        return {"seeds": sorted(seeds), "nodes": nodes, "edges": edges,
                "paths": paths, "summary": summary}

    def _target_documents(self, query: str, entities: list[str]) -> set[str]:
        """Intent routing: which document(s) is the query explicitly about?
        Resolved from named equipment tags, concept names, and doc ids — a
        confident, deterministic signal that beats vote-based dominance."""
        ql = query.lower()
        qnorm = re.sub(r"[^a-z0-9]", "", ql)               # separator-insensitive
        qtokens = {t for t in re.findall(r"[a-z0-9]{3,}", ql)}
        doc_stop = {
            "the", "and", "for", "with", "about", "explain", "detail",
            "overview", "definition", "what", "why", "how", "need",
        }

        def doc_terms(value: str) -> set[str]:
            return {
                t
                for t in re.findall(r"[a-z0-9]{2,}", value.lower())
                if t not in doc_stop and not t.isdigit()
            }

        # Strongest signal: the query names a document (by tag it contains, or by
        # its id/filename ignoring separators — so "1bit llm" -> the "1_bit" doc).
        # When present this WINS: scope to that doc only, don't dilute with the
        # concepts it happens to share with larger documents.
        strong: set[str] = set()
        if entities:
            ent = set(entities)
            for ch in self.corpus.chunks:
                if ent & set(ch.entities):
                    strong.add(ch.doc_no)
        for doc_no, meta in self.corpus.docs.items():
            dnorm = re.sub(r"[^a-z0-9]", "", doc_no.lower())
            tnorm = re.sub(r"[^a-z0-9]", "", (meta.get("title") or "").lower().replace(".pdf", ""))
            title_terms = doc_terms(doc_no) | doc_terms((meta.get("title") or "").replace(".pdf", ""))
            if title_terms:
                overlap = title_terms & qtokens
                # Title-token routing handles natural variants such as
                # "attention all you need" for "Attention Is All You Need".
                if len(overlap) >= min(2, len(title_terms)):
                    strong.add(doc_no)
                    continue
            if (len(dnorm) >= 4 and dnorm in qnorm) or (len(tnorm) >= 4 and tnorm in qnorm):
                strong.add(doc_no)
        if strong:
            return strong

        # Fallback: concept names in the query -> all their source documents.
        targets: set[str] = set()
        for nid, nd in self.graph.nodes.items():
            if nd.get("type") == "Concept" and len(nid) >= 4 and self._concept_in_query(nid, ql, qnorm, qtokens):
                for sd in nd.get("source_docs", [nd.get("source_doc")]):
                    if sd:
                        targets.add(sd)
        if targets:
            # Concept routing is a heuristic, and a single generic token can
            # hijack it: the Attention paper's 'GPUs' concept routed
            # 'tell me about the gpu-graph accelerator' — a RESUME project —
            # to the paper, locking retrieval onto the wrong document while
            # BM25 scored the resume chunk at double the runner-up. Sanity
            # check: if NONE of the query's top confident lexical hits fall
            # inside the concept-routed documents, the corpus disagrees with
            # the concept association — drop it and let the dominant-
            # document vote decide from actual retrieval.
            hits = self.index._bm25(query)[:5]
            if hits and hits[0][1] >= 4.0:
                inside = sum(1 for i, _ in hits
                             if self.index.chunks[i].doc_no in targets)
                if inside == 0:
                    return set()
        return targets

    @staticmethod
    def _concept_in_query(concept: str, ql: str, qnorm: str, qtokens: set[str]) -> bool:
        """Match a concept against the query tolerantly: exact phrase, the whole
        concept ignoring separators, or a singular/plural token match (so 'llm'
        matches the 'LLMs' concept)."""
        nl = concept.lower()
        if re.search(r"\b" + re.escape(nl) + r"\b", ql):
            return True
        nnorm = re.sub(r"[^a-z0-9]", "", nl)
        if len(nnorm) >= 5 and nnorm in qnorm:
            return True
        for w in re.findall(r"[a-z0-9]{3,}", nl):          # per concept word
            for t in qtokens:
                if t == w or t + "s" == w or w + "s" == t:  # exact or plural
                    return True
        return False

    def _planner(self, query: str, anchor: str | None) -> dict:
        """Decompose the question into a sub-task list and a retrieval strategy.
        The Planner OWNS query decomposition; workers only execute the plan."""
        tasks = []
        if anchor:
            tasks.append(f"Assess condition and remaining useful life of {anchor} (Predictive)")
            tasks.append(f"Traverse the graph around {anchor} and match failure history (RootCause)")
        tasks.append("Retrieve governing procedures, specs and limits with citations (Knowledge)")
        if anchor:
            tasks.append("Check spares and maintenance history; draft a work order (Maintenance)")
            tasks.append("Check permit / isolation requirements — hard gate (Safety)")
            tasks.append("Aggregate into a ranked recommendation (Risk)")
        tasks.append("Verify every answer claim against its sources (Critic)")
        return {"tasks": tasks,
                "strategy": "graph+document" if anchor else "document",
                "subqueries": self._plan_subqueries(query, anchor)}

    def _plan_subqueries(self, query: str, anchor: str | None) -> list[str]:
        """Query decomposition. For an anchored asset the sub-queries are derived
        from the asset's graph context (its governing documents) plus generic
        condition/procedure/safety aspects — parameterised by the asset, not
        hardcoded to any one machine, so it generalises to C-301, E-201, etc."""
        subs = [query]
        if anchor:
            for doc, _ in self.graph.neighbors(anchor, "GOVERNS"):
                title = self.graph.nodes.get(doc, {}).get("label", doc)
                subs.append(f"{anchor} {title}")
            subs.append(f"{anchor} operating limits vibration temperature alarm response")
            subs.append(f"{anchor} maintenance repair procedure torque replacement spares")
            subs.append(f"{anchor} permit isolation lockout safety requirement")
        seen: set[str] = set()
        out: list[str] = []
        for s in subs:                                     # dedupe, preserve order
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out[:6]

    def _predictive_agent(self) -> dict:
        return predictive_mod.analyze(self.corpus.sensors)

    def _rootcause_agent(self, anchor: str) -> dict:
        nodes, edges = self.graph.subgraph(anchor, hops=2)
        connected = [n for n, r in self.graph.neighbors(anchor, "CONNECTED_TO")]
        measured_by = [n for n, r in self.graph.neighbors(anchor, "MEASURED_BY")]
        history = [wo for wo in self.corpus.maintenance if wo["equipment"] == anchor]
        similar = [wo for wo in history if wo["failure_mode"] == "bearing failure"]
        candidate = None
        if similar:
            candidate = {
                "failure_mode": "bearing wear (BRG-204)",
                "likely_cause": "shaft misalignment",
                "evidence_chain": [
                    f'{anchor} -CONNECTED_TO-> {", ".join(connected)} (from P&ID {self.corpus.pid["drawing"]})',
                    f'{len(similar)} past work orders on {anchor} with the same rising-vibration signature: '
                    + ", ".join(w["wo_number"] for w in similar),
                    "All past occurrences were resolved by BRG-204 replacement + laser alignment",
                ],
            }
        return {"anchor": anchor, "subgraph_nodes": sorted(nodes), "subgraph_edges": edges,
                "connected_to": connected, "measured_by": measured_by,
                "history": history, "similar_count": len(similar), "candidate_cause": candidate}

    @staticmethod
    def _synthetic_evidence(*, doc_no: str, doc_title: str, section: str,
                            revision: str, text: str,
                            matched: list[str]) -> dict:
        """Build an evidence item from structured system-of-record data, in the
        exact shape retrieval produces so it is citable, verifiable and
        renderable like any retrieved chunk."""
        return {
            "chunk_id": f"{doc_no}::data",
            "doc_no": doc_no, "doc_title": doc_title, "section": section,
            "revision": revision, "page": 1, "text": text,
            "summary": text[:200], "score": 0.9, "rrf_score": 0.0,
            "reranker_score": 0.9,
            "matched_entities": [m for m in matched if m],
            "selected_because": ("structured system-of-record data "
                                 "(live telemetry / maintenance log)"),
            "entities": [], "keywords": [], "concepts": [], "synthetic": True,
        }

    def _asset_data_evidence(self, anchor: str, findings: dict) -> list[dict]:
        """The monitored asset's OWN data as first-class, citable evidence.

        Live telemetry (the sensor stream) and maintenance work-order history
        (the maintenance log) are facts in the system of record, not in any
        retrievable text chunk. Without promoting them here they live only in
        the case-file findings — with no evidence id — so the validator can
        neither see nor verify claims grounded in them ("vibration 5.45 mm/s",
        "WO-8811"), flags every one as ungrounded, and the confidence floor
        suppresses an otherwise-correct answer. Promoting them lets the LLM
        cite the data and the validator confirm it.
        """
        items: list[dict] = []
        p = findings.get("predictive") or {}
        if anchor == getattr(predictive_mod, "ANCHOR_ASSET", None) and p:
            zone = ("above the danger limit"
                    if p.get("latest_vibration", 0) >= p.get("danger_limit", 1e9)
                    else "in the alert zone" if p.get("in_alert_zone")
                    else "within normal limits")
            text = (
                f"Live condition monitoring for {anchor} (vibration transmitter "
                f"VT-101 on the drive-end bearing). Current "
                f"{p.get('signal', 'vibration')} is {p.get('latest_vibration')} "
                f"mm/s velocity RMS, {zone}. Baseline "
                f"{p.get('baseline_vibration')} mm/s; trend rising at "
                f"+{p.get('trend_mm_s_per_day')} mm/s per day. Alert limit "
                f"{p.get('alert_limit')} mm/s and danger limit "
                f"{p.get('danger_limit')} mm/s per SOP-315 (ISO 10816). "
                f"Estimated remaining useful life {p.get('rul_days')} days. "
                f"Bearing temperature {p.get('bearing_temp_recent_c')} °C "
                f"(normal). Anomaly detected: "
                f"{'yes' if p.get('anomaly') else 'no'}."
            )
            items.append(self._synthetic_evidence(
                doc_no="VT-101",
                doc_title=f"Live Condition Monitoring — {anchor} (VT-101)",
                section="Vibration telemetry", revision="live", text=text,
                matched=[anchor.lower(), "vibration", "vt-101"]))

        history = (findings.get("rootcause") or {}).get("history") or []
        for wo in sorted(history, key=lambda w: w.get("date", ""),
                         reverse=True)[:6]:
            fm = (wo.get("failure_mode") or "").strip()
            cause = (wo.get("cause") or wo.get("root_cause") or "").strip()
            action = (wo.get("action") or wo.get("corrective_action") or "").strip()
            parts = [f"Work order {wo.get('wo_number', '')} dated "
                     f"{wo.get('date', '')} on {wo.get('equipment', anchor)}"]
            if wo.get("type"):
                parts.append(f"{wo['type']} maintenance")
            if fm and fm.lower() not in ("none", ""):
                parts.append(f"failure mode {fm}")
            if cause and cause.lower() not in ("none", ""):
                parts.append(f"root cause {cause}")
            if action:
                parts.append(f"action taken: {action}")
            pu = str(wo.get("parts_used") or "").strip()
            if pu and pu.lower() not in ("none", ""):
                parts.append(f"parts used {pu}")
            if wo.get("downtime_hours"):
                parts.append(f"downtime {wo['downtime_hours']} hours")
            items.append(self._synthetic_evidence(
                doc_no=wo.get("wo_number", "WO"),
                doc_title=(f"Maintenance Work Order {wo.get('wo_number', '')} "
                           f"— {wo.get('equipment', anchor)}"),
                section="Maintenance history",
                revision=str(wo.get("date", "")),
                text="; ".join(parts) + ".",
                matched=[anchor.lower(), "maintenance", fm.lower()]))
        return items

    def _accept_evidence(self, query: str, query_plan, hits: list[dict],
                         k: int = 6) -> tuple[list[dict], dict[str, str]]:
        """The retrieval → acceptance gate. Ranking PROPOSES candidates; this
        DECIDES which become evidence. A chunk is used only if it clears:
          - a relative score bar (≥ 45% of the top hit's score),
          - an absolute content bar (covers the query's informative terms,
            with real lexical support — not a keyword-stuffed outline),
          - for comparison questions: mentions at least one comparison side.
        Everything rejected gets a recorded, human-readable reason so the
        diagnostics panel shows WHY, not just 'lower relevance'."""
        if not hits:
            return [], {}
        # Gate against the PROCESSED query vocabulary — the query processor
        # already normalised typos/casing ("langraph" → "LangGraph"); gating
        # on the raw string falsely rejects evidence that spells terms right.
        qtext = (getattr(query_plan, "rewritten_query", None)
                 or getattr(query_plan, "expanded_query", None) or query)
        qterms = [t for t in self.index.query_terms(qtext) if len(t) >= 3]
        # With a real cross-encoder the score IS a semantic relevance signal —
        # lexical checks become a sanity floor, not a second gate. Without it,
        # the lexical bars carry the acceptance decision.
        semantic = self.index.has_semantic_reranker
        need_terms = 1 if semantic else min(2, len(qterms))
        support_bar = 0.12 if semantic else 0.35
        aspects = [a.lower() for a in (getattr(query_plan, "comparison_aspects", None) or [])]
        used: list[dict] = []
        reasons: dict[str, str] = {}
        # Hard filter first: outline/contents pages are navigation, not
        # evidence — they must not even anchor the answer.
        ranked = []
        for h in sorted(hits, key=lambda x: -x.get("score", 0.0)):
            text = h.get("text", "") or ""
            lines = [l for l in text.splitlines() if l.strip()]
            if len(lines) > 3 and self.index.prose_factor(text) < 0.15:
                reasons[h["chunk_id"]] = "outline / table-of-contents page, not substantive evidence"
                continue
            ranked.append(h)
        if not ranked:
            return [], reasons
        top = ranked[0].get("score", 0.0) or 1.0
        single_doc_focus = len({h.get("doc_no") for h in ranked if h.get("doc_no")}) == 1
        # Document-scoped question on a SMALL document: the user named the
        # document, so completeness beats precision — every substantive
        # chunk (the outline hard-filter above already ran) becomes
        # evidence. Without this, 'tell me the projects in the resume' kept
        # 3 of 6 resume chunks because the parts listing three of the five
        # projects never use the word 'project', and the answer omitted
        # them. Mirrors the same relaxation in the retrieval-level gate.
        focus_doc = ranked[0].get("doc_no") if single_doc_focus else None
        small_doc_focus = bool(
            focus_doc
            and len(self.index._doc_positions.get(focus_doc, [])) <= 12
        )
        for rank, h in enumerate(ranked):
            cid = h["chunk_id"]
            if len(used) >= k:
                reasons[cid] = "evidence budget reached"
                continue
            if rank == 0:                      # best hit anchors the answer;
                used.append(h)                 # if even it is weak, the
                continue                       # grounded-or-refuse gate fires
            if small_doc_focus:
                used.append(h)
                continue
            score = h.get("score", 0.0)
            if score < 0.45 * top:
                reasons[cid] = f"below relevance bar ({score:.2f} < 45% of top hit)"
                continue
            text = h.get("text", "") or ""
            tl = text.lower()
            present = [t for t in qterms if t in tl]
            if len(present) < need_terms:
                reasons[cid] = (f"insufficient query-term support "
                                f"({len(present)}/{len(qterms)} informative terms)")
                continue
            if aspects and not any(a in tl for a in aspects):
                reasons[cid] = "mentions neither side of the comparison"
                continue
            support = self.index._lexical_support(qtext, text)
            if support < support_bar:
                if single_doc_focus and self.index.prose_factor(text) >= 0.20:
                    used.append(h)
                    continue
                reasons[cid] = f"weak content support ({support:.2f} — outline/keyword-only match)"
                continue
            used.append(h)
        return used, reasons

    @staticmethod
    def _diagnostics(candidates: list[dict], used: list[dict],
                     focus: set[str] | None = None, query: str = "",
                     graph_entities: set[str] | None = None,
                     reasons: dict[str, str] | None = None) -> dict:
        """Evidence diagnostics: every chunk the retriever surfaced, its relevance
        (fused score normalised to 0–1), the terms it matched, and whether it was
        used or rejected and why. The 'show your work' surface for enterprise trust."""
        used_ids = {e["chunk_id"] for e in used}
        pool = {c["chunk_id"]: c for c in candidates}
        for e in used:                                   # include doc-scoped extras
            pool.setdefault(e["chunk_id"], e)
        mx = max((c.get("score", 0.0) for c in pool.values()), default=1.0) or 1.0
        ge = graph_entities or set()
        qterms = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9.\-]{2,}", query)
                  if len(t) >= 4 or any(x.isdigit() for x in t)]
        items = []
        reasons = reasons or {}
        for c in sorted(pool.values(), key=lambda x: -x.get("score", 0.0)):
            is_used = c["chunk_id"] in used_ids
            reason = ("" if is_used else
                      reasons.get(c["chunk_id"])
                      or ("outside focused document" if focus and c["doc_no"] not in focus
                          else "lower relevance"))
            ent = set(c.get("entities", []))
            text_low = (c.get("text", "") or "").lower()
            matched = [e for e in ent if e in ge]        # graph concepts/entities present
            for t in qterms:                              # distinctive query terms present
                if t.lower() in text_low and t not in matched:
                    matched.append(t)
            items.append({"doc_no": c["doc_no"], "section": c["section"],
                          "relevance": round(c.get("score", 0.0) / mx, 2),
                          "used": is_used, "reason": reason,
                          "method": "Graph+Dense" if matched and (set(matched) & ge) else "Hybrid",
                          "matched_terms": matched[:6]})
        used_n = sum(1 for i in items if i["used"])
        return {"retrieved": len(items), "used": used_n,
                "rejected": len(items) - used_n, "items": items}

    # ---------------------------------------------------------------- M03
    # Whole-document / page-wise requests ("summarize the entire paper",
    # "explain page 5") are DOCUMENT requests, not search queries: top-k
    # retrieval structurally cannot answer them, whatever k is. They get a
    # direct document-map evidence path instead.

    _DOC_REQ_VERB = re.compile(
        r"\b(?:summari[sz]e|summary|explain|describe|overview|walk\s*through|"
        r"walkthrough|go\s+through|review)\b", re.IGNORECASE)
    _DOC_REQ_WHOLE = re.compile(
        r"\b(?:whole|entire|complete|full|every\s+page|all\s+pages|"
        r"page\s*(?:-|\s)?by\s*(?:-|\s)?page|page[- ]wise|start\s+to\s+"
        r"(?:finish|end))\b", re.IGNORECASE)
    _DOC_REQ_PAGE = re.compile(r"\bpage\s+(\d{1,4})\b", re.IGNORECASE)
    _DOC_WORDS = {"document", "doc", "pdf", "paper", "resume", "book",
                  "manual", "file", "report", "contents", "image", "it"}

    def _doc_request(self, query: str) -> dict | None:
        """Classify a query as a document-level request, or None."""
        m = self._DOC_REQ_PAGE.search(query)
        if m and self._DOC_REQ_VERB.search(query):
            return {"mode": "page", "page": int(m.group(1))}
        if not self._DOC_REQ_VERB.search(query):
            return None
        if self._DOC_REQ_WHOLE.search(query):
            return {"mode": "whole"}
        # bare "summarize the <doc>": after removing the verb and generic
        # document words, nothing content-bearing remains — the request is
        # about the document itself, not a topic inside it.
        toks = [t for t in re.findall(r"[a-z0-9][a-z0-9_.-]+", query.lower())
                if t not in convmem._STOP]
        leftovers = [t for t in toks
                     if not self._DOC_REQ_VERB.fullmatch(t)
                     and t not in self._DOC_WORDS
                     and t not in {"summarize", "summarise", "summary",
                                   "overview", "walkthrough"}]
        content = [t for t in leftovers
                   if t not in {d.lower() for d in self.corpus.docs}
                   and not any(t in d.lower() or t in
                               str(m.get("title", "")).lower()
                               for d, m in self.corpus.docs.items())]
        if not content:
            return {"mode": "whole"}
        return None

    # Above this many chunks, whole-document evidence uses the document MAP
    # (per-chunk ingest summaries bucketed into ranges) instead of full text.
    DOC_MODE_FULL_TEXT_LIMIT = 12
    DOC_MODE_BUCKETS = 15

    def _document_mode_evidence(self, doc_no: str, req: dict) -> list[dict]:
        idxs = self.index._doc_positions.get(doc_no, [])
        chunks = [self.index.chunks[i] for i in idxs]
        if not chunks:
            return []

        def item(cid, section, text, page):
            return {
                "chunk_id": cid, "doc_no": doc_no,
                "doc_title": chunks[0].doc_title,
                "revision": chunks[0].revision, "section": section,
                "text": text, "summary": "", "keywords": [], "concepts": [],
                "entities": [], "page": page, "score": 1.0,
                "dense_score": 0.0, "bm25_score": 0.0, "graph_score": 0,
                "rrf_score": 0.0, "reranker_score": 1.0, "entity_score": 0.0,
                "matched_entities": [],
                "retrieval_method": "document-mode",
                "retrieval_reason": f"document-level request ({req['mode']})",
                "selected_because": f"document {req['mode']} request — "
                                    "bypasses top-k retrieval",
                "accepted": True, "acceptance_score": 1.0,
            }

        if req["mode"] == "page":
            sel = [c for c in chunks if c.page == req["page"]][:8]
            return [item(c.chunk_id, c.section, c.text, c.page) for c in sel]

        if len(chunks) <= self.DOC_MODE_FULL_TEXT_LIMIT:
            return [item(c.chunk_id, c.section, c.text, c.page)
                    for c in chunks]

        # Large document: hierarchical map from the per-chunk summaries the
        # ingest pipeline already computed — full structural coverage at a
        # fraction of the tokens, no extra LLM cost.
        size = max(1, math.ceil(len(chunks) / self.DOC_MODE_BUCKETS))
        out = []
        for start in range(0, len(chunks), size):
            group = chunks[start:start + size]
            pages = sorted({c.page for c in group})
            label = (f"Pages {pages[0]}–{pages[-1]}"
                     if pages[-1] != pages[0] else f"Page {pages[0]}")
            digest = "\n".join(
                f"- ({c.section}) {c.summary or c.text[:120]}"
                for c in group)[:1500]
            out.append(item(f"{doc_no}::map{start}", label,
                            f"{label} overview:\n{digest}", pages[0]))
        return out

    def _knowledge_agent(
    self,
    query: str,
    query_plan,
    graph_entities: set[str],
    plan: dict,
    target_docs: set[str] | None = None,
):
        """Execute the Planner's retrieval strategy. Document questions use
        document-focus retrieval (routed by named entity/concept, else a
        vote-based dominant doc); asset questions execute the Planner's
        sub-queries and round-robin merge their evidence."""
        self._retrieval_focus = None
        self._retrieval_diagnostics = None

        # plan_query = self.query_processor.process(query)

        retrieval_queries = query_plan.retrieval_queries

        # M03: document-level requests bypass top-k retrieval entirely.
        doc_req = self._doc_request(query)
        if doc_req and target_docs and len(target_docs) == 1:
            doc = next(iter(target_docs))
            evidence = self._document_mode_evidence(doc, doc_req)
            if evidence:
                self._retrieval_focus = {
                    "document": doc, "rejected_documents": [],
                    "routed_by": f"document-{doc_req['mode']} request",
                }
                self._retrieval_diagnostics = {
                    "retrieved": len(evidence), "used": len(evidence),
                    "rejected": 0, "mode": f"document-{doc_req['mode']}",
                    "items": [],
                }
                print(f"📄 Document-mode evidence: {doc_req['mode']} of "
                      f"{doc} → {len(evidence)} item(s)")
                return evidence

        if plan["strategy"] == "document":

            first = self.index.retrieve(

                retrieval_queries[0],

                query_plan=query_plan,

                graph_entities=graph_entities,

                k=12,

            )
            if not first:
                return []
            focus: set[str] | None = None
            routed_by = None
            rejected_docs: list[str] = []
            if target_docs:
                focus, routed_by = set(target_docs), "named entity/concept"
            else:
                from collections import Counter
                weight: Counter = Counter()
                for rank, h in enumerate(first):
                    weight[h["doc_no"]] += 1.0 / (1 + rank)  # rank-weighted vote
                dominant, dweight = weight.most_common(1)[0]
                share = dweight / sum(weight.values())
                n_dominant = sum(1 for h in first if h["doc_no"] == dominant)
                if share >= 0.4 and n_dominant >= 2:
                    focus, routed_by = {dominant}, f"dominant document ({int(share * 100)}%)"
                    rejected_docs = sorted({h["doc_no"] for h in first if h["doc_no"] != dominant})

            if focus:
                scoped = self.index.retrieve(

                                retrieval_queries[0],

                                query_plan=query_plan,

                                graph_entities=graph_entities,

                                k=6,

                                doc_filter=focus,

                            )
                used, reasons = self._accept_evidence(query, query_plan, scoped or first, k=6)
                self._retrieval_focus = {"document": ", ".join(sorted(focus)),
                                         "rejected_documents": rejected_docs,
                                         "routed_by": routed_by}
            else:
                used, reasons = self._accept_evidence(query, query_plan, first, k=6)
            self._retrieval_diagnostics = self._diagnostics(first, used, focus, query,
                                                            graph_entities, reasons)
            return used

        # Asset case: execute the Planner's sub-queries. Graph boost applies to
        # the user's own query; the derived sub-queries are targeted lookups.
        sub_queries = plan["subqueries"]
        processed_sub = self.query_processor.process(sub_queries[0])

        rankings = [
            self.index.retrieve(

                processed_sub.retrieval_queries[0],

                query_plan=processed_sub,

                graph_entities=graph_entities,

                k=4,

            )
        ]
        
        for sq in sub_queries[1:]:

            processed = self.query_processor.process(sq)

            rankings.append(

                self.index.retrieve(

                    processed.retrieval_queries[0],

                    query_plan=processed,

                    k=4,

                )

            )
        
        merged: dict[str, dict] = {}
        for rank in range(4):                            # round-robin interleave
            for ranking in rankings:
                if rank < len(ranking) and ranking[rank]["chunk_id"] not in merged:
                    merged[ranking[rank]["chunk_id"]] = ranking[rank]
        used, gate_reasons = self._accept_evidence(query, query_plan,
                                                   list(merged.values()), k=8)
        candidates: dict[str, dict] = {}                 # union of all sub-query hits
        for ranking in rankings:
            for c in ranking:
                if c["chunk_id"] not in candidates:

                    candidates[c["chunk_id"]] = c.copy()

                    candidates[c["chunk_id"]]["hits"] = 1

                else:

                    candidates[c["chunk_id"]]["score"] += c["score"]

                    candidates[c["chunk_id"]]["rrf_score"] += c.get("rrf_score", 0)

                    candidates[c["chunk_id"]]["hits"] += 1
        self._retrieval_diagnostics = self._diagnostics(list(candidates.values()), used, None,
                                                        query, graph_entities, gate_reasons)
        return used

    def _maintenance_agent(self, anchor: str, pred: dict) -> dict:
        needed = ["BRG-204"]
        spares = [s for s in self.corpus.spares if s["part_number"] in needed]
        in_stock = all(int(s["qty_on_hand"]) >= 2 for s in spares) if spares else False
        last_pm = max((wo for wo in self.corpus.maintenance
                       if wo["equipment"] == anchor and wo["type"] == "preventive"),
                      key=lambda w: w["date"], default=None)
        window_days = max(1, int((pred.get("rul_days") or 7) - 2))
        draft_wo = {
            "status": "DRAFT — requires human approval",
            "equipment": anchor,
            "title": f"Replace drive-end and non-drive-end bearings (BRG-204) on {anchor}",
            "procedure": "SOP-207 rev 3",
            "permit": "LOTO per SAF-12",
            "parts": "2 x BRG-204 (Store A / Bin 14)",
            "schedule_within_days": window_days,
            "estimated_downtime_hours": 5,
        }
        return {"spares_needed": needed, "spares": spares, "spares_in_stock": in_stock,
                "last_preventive": last_pm, "draft_work_order": draft_wo}

    def _safety_agent(self, anchor: str) -> dict:
        """Hard gate: find procedures governing the asset that REQUIRE a permit."""
        gates = []
        for doc, rel in self.graph.neighbors(anchor, "GOVERNS"):
            for permit, rel2 in self.graph.neighbors(doc, "REQUIRES"):
                if self.graph.nodes[permit]["type"] == "Permit":
                    gates.append({"procedure": doc,
                                  "procedure_title": self.graph.nodes[doc]["label"],
                                  "permit": self.graph.nodes[permit]["label"]})
        return {"hard_gate": bool(gates), "requirements": gates,
                "statement": ("HARD GATE: energy isolation permit required before any work — "
                              "no recommendation proceeds without it.") if gates else
                             "No permit requirement found for this asset."}

    def _risk_agent(self, pred: dict, root: dict, maint: dict, safety: dict) -> dict:
        rul = pred.get("rul_days")
        action_days = maint["draft_work_order"]["schedule_within_days"]
        return {
            "recommendation": (
                f"Schedule bearing replacement on {root['anchor']} within {action_days} days "
                f"(predicted danger-limit crossing in ~{rul} days). Isolate per SAF-12 (LOTO), "
                f"replace 2 x BRG-204 per SOP-207, laser-align, and verify vibration < 2.8 mm/s."
            ),
            "alternatives": [
                {"option": "Run to failure", "assessment":
                    "Rejected — history shows seizure risk and a full line stop (past downtime 5–6 h planned vs unbounded unplanned)."},
                {"option": "Immediate shutdown", "assessment":
                    f"Not required yet — vibration {pred['latest_vibration']} mm/s is in the alert zone "
                    f"({pred['alert_limit']}–{pred['danger_limit']}), which allows planned intervention."},
            ],
            "estimated_downtime_hours": maint["draft_work_order"]["estimated_downtime_hours"],
        }

    # Terms too generic to prove a claim is grounded.
    _VERIFY_STOP = {
        "the", "and", "for", "with", "this", "that", "from", "which", "into",
        "your", "their", "there", "here", "these", "those", "should", "would",
        "could", "about", "above", "below", "before", "after", "while", "when",
        "where", "what", "have", "has", "been", "being", "are", "was", "were",
        "will", "must", "may", "not", "any", "all", "each", "per", "via", "over",
        "under", "within", "also", "such", "more", "most", "than", "then",
        "requires", "required", "require", "using", "used", "based", "value",
        "values", "check", "checked", "provide", "provides", "including",
    }

    def _salient_terms(self, text: str) -> list[str]:
        """Distinctive terms (numbers, tags, technical/long words) that carry the
        checkable content of a claim or query — stopwords and filler removed."""
        terms = re.findall(r"[A-Za-z][A-Za-z0-9.\-]{2,}|\d+(?:\.\d+)?", text)
        return [t for t in terms
                if (any(c.isdigit() for c in t) or t[:1].isupper() or len(t) >= 6)
                and t.lower() not in self._VERIFY_STOP]

    def _verify_answer(
    self,
    query: str,
    answer: str,
    corpus_text: str,
    evidence: list[dict],
    graph_seeds: list[str],
) -> dict:
        """
        Multi-factor deterministic critic.

        Evaluates:
            • Grounding
            • Completeness
            • Evidence coverage
            • Citation support
            • Graph agreement

        Produces an explainable confidence score.
        """

        corpus_low = corpus_text.lower()
        ans_low = answer.lower()

        # ============================================================
        # 1. Grounding
        # ============================================================

        checks = []

        # Sections whose content is NOT graded for grounding: General Knowledge
        # is explicitly model knowledge (not from the evidence), and Sources is a
        # citation list. Everything else (Direct Answer, Explanation, Important
        # Details, Limitations, Summary) is verified paragraph-by-paragraph.
        skip_sections = ("general knowledge", "sources")

        # Split the answer into sections by their ## headings, drop the ungraded
        # sections (General Knowledge = model knowledge; Sources = citations),
        # then break each remaining section into claim units (sentences / bullets).
        parts = re.split(r"\n#{1,4}\s*([^\n]+)\n", "\n" + answer)
        section_bodies = []
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                title = parts[i].strip().lower()
                if not any(sk in title for sk in skip_sections):
                    section_bodies.append(parts[i + 1] if i + 1 < len(parts) else "")
        else:
            section_bodies = [answer]

        claim_units = []
        for body in section_bodies:
            for raw in re.split(r"(?<=[.!?])\s+|\n[-*]\s+|\n{2,}", body):
                unit = re.sub(r"[*_#>`]+", " ", raw).strip()
                if len(unit.split()) >= 8:
                    claim_units.append(unit)

        for claim in claim_units:

            salient = self._salient_terms(claim)

            if len(salient) < 3:
                continue

            present = sum(
                1
                for t in salient
                if t.lower() in corpus_low
            )

            ratio = present / len(salient)

            if ratio >= 0.80:
                level = "Strong"
            elif ratio >= 0.60:
                level = "Moderate"
            elif ratio >= 0.40:
                level = "Weak"
            else:
                level = "Unsupported"

            checks.append(
                {
                    "claim": claim[:180],
                    "grounding_score": round(ratio, 2),
                    "grounding_level": level,
                    "grounded_terms": present,
                    "total_terms": len(salient),
                }
            )

        # ============================================================
        # Remove duplicate claims
        # ============================================================

        unique = {}

        for check in checks:
            unique.setdefault(check["claim"], check)

        checks = list(unique.values())

        grounded = (
            sum(c["grounding_score"] for c in checks) / len(checks)
            if checks
            else 0.70
        )

        # ============================================================
        # Grounding statistics
        # ============================================================

        strong = sum(
            1
            for c in checks
            if c["grounding_level"] == "Strong"
        )

        moderate = sum(
            1
            for c in checks
            if c["grounding_level"] == "Moderate"
        )

        weak = sum(
            1
            for c in checks
            if c["grounding_level"] == "Weak"
        )

        unsupported = sum(
            1
            for c in checks
            if c["grounding_level"] == "Unsupported"
        )

        # ============================================================
        # 2. Completeness
        # ============================================================

        q_terms = [
            t.lower()
            for t in self._salient_terms(query)
        ]

        completeness = (
            sum(
                1
                for t in set(q_terms)
                if t in ans_low
            )
            / len(set(q_terms))
            if q_terms
            else 0.80
        )

        # ============================================================
        # 3. Evidence Coverage
        # ============================================================

        used_ev = 0

        for e in evidence:

            terms = [
                t.lower()
                for t in self._salient_terms(
                    e["text"][:400]
                )
            ]

            if not terms:
                continue

            overlap = (
                sum(
                    1
                    for t in set(terms)
                    if t in ans_low
                )
                / len(set(terms))
            )

            if overlap >= 0.12:
                used_ev += 1

        evidence_cov = (
            used_ev / len(evidence)
            if evidence
            else 0.0
        )

        # ============================================================
        # 4. Graph Agreement
        # ============================================================

        graph_agree = None

        if graph_seeds:

            hits = sum(
                1
                for seed in graph_seeds
                if seed.lower() in ans_low
                or seed.lower() in corpus_low
            )

            graph_agree = hits / len(graph_seeds)

        # ============================================================
        # 5. Citation Support
        # ============================================================

        citation = min(
            1.0,
            len(evidence) / 3.0,
        )

        # ============================================================
        # Confidence Factors
        # ============================================================

        factors = {

            "grounding": (
                grounded,
                0.30,
                f"{strong} strong, "
                f"{moderate} moderate, "
                f"{weak} weak, "
                f"{unsupported} unsupported claim(s)",
            ),

            "completeness": (
                completeness,
                0.25,
                "Question coverage",
            ),

            "evidence_coverage": (
                evidence_cov,
                0.20,
                f"{used_ev}/{len(evidence)} retrieved passages used",
            ),

            "citation_support": (
                citation,
                0.15,
                f"{len(evidence)} retrieved sources",
            ),
        }

        if graph_agree is not None:

            factors["graph_agreement"] = (
                graph_agree,
                0.10,
                "Graph-expanded concepts confirmed",
            )

        # ============================================================
        # Final Confidence
        # ============================================================

        total_weight = sum(
            weight
            for _, weight, _ in factors.values()
        )

        confidence = round(
            sum(
                score * weight
                for score, weight, _ in factors.values()
            )
            / total_weight,
            2,
        )

        # ============================================================
        # Verdict
        # ============================================================

        if confidence >= 0.75:
            verdict = "RELEASE"

        elif confidence >= 0.55:
            verdict = "RELEASE WITH CAVEATS"

        else:
            verdict = "ESCALATE TO HUMAN EXPERT"

        # ============================================================
        # Return
        # ============================================================

        return {

            "checks": checks,

            "confidence": confidence,

            "verdict": verdict,

            "unsupported": unsupported,

            "factors": {
                name: {
                    "score": round(score, 2),
                    "detail": detail,
                }
                for name, (score, _, detail) in factors.items()
            },

            "policy": (
                "Calibrated multi-factor verification "
                "(grounding, completeness, evidence coverage, "
                "citation support, graph agreement)"
            ),
        }
    @staticmethod
    def _refusal_message() -> str:
        return ("I don't have grounded evidence to answer that, so I won't guess. "
                "Per the grounded-or-refuse policy this is escalated to a human expert — "
                "and logged as a **knowledge gap**: if this question matters to operations, "
                "the procedure should be captured before the expertise walks out the door.")

    # ------------------------------------------------------------------ case

    def _support_graph(self, query: str, answer: str, evidence: list[dict],
                       anchor: str | None, findings: dict) -> list[str]:
        """The FINAL support graph: only nodes that contributed to accepted
        evidence — source documents of used chunks, plus their entities/
        concepts that actually surface in the answer or the query. This is
        deliberately narrower than the retrieval-expanded traversal, which is
        an intermediate artifact, not what the answer stands on."""
        al = (answer or "").lower()
        ql = (query or "").lower()
        keep: set[str] = set()
        for ev in evidence:
            doc = ev.get("doc_no")
            if doc in self.graph.nodes:
                keep.add(doc)
            for nid in list(ev.get("entities") or []) + list(ev.get("concepts") or []):
                if nid in self.graph.nodes:
                    n = nid.lower()
                    if n in al or n in ql:
                        keep.add(nid)
        if anchor and anchor in self.graph.nodes:
            keep.add(anchor)
            rc = findings.get("rootcause", {})
            keep.update(n for n in rc.get("connected_to", []) if n in self.graph.nodes)
            for wo in rc.get("history", []):
                if wo.get("wo_number") in self.graph.nodes:
                    keep.add(wo["wo_number"])
        return sorted(keep)

    def _conversation_case(self, query: str, history: list, trace: ReasoningTrace) -> dict:
        """Answer a question about the conversation itself from the transcript.
        Grounding here is the chat log, not the document corpus."""
        n_user = sum(1 for t in history if t.get("role") == "user")
        trace.add(
            stage="conversation",
            title="Conversation Memory",
            summary=f"Question is about the conversation itself — answering from "
                    f"the transcript ({len(history)} turns, {n_user} user questions)",
        )
        case_file = (
            "CONVERSATION TRANSCRIPT (the user's question is about this chat)\n"
            + "=" * 70 + "\n"
            + convmem.transcript(history)
            + "\n" + "=" * 70 + "\n"
            "Answer the user's question strictly from the transcript above.\n"
            "Be specific: quote or paraphrase the actual questions/answers.\n"
            "If the user asks how topics relate, connect them explicitly.\n"
            "Do not invent turns that are not in the transcript.\n"
        )
        answer = llm.synthesize(query, case_file, None)
        # last_generation() is thread-local (per-request); the module-global
        # ACTIVE_PROVIDER can be overwritten by a concurrent request.
        answer_source = (llm.last_generation().get("provider")
                         or llm.ACTIVE_PROVIDER) if answer else "conversation-memory"
        if not answer:
            answer = convmem.conversation_answer(query, history)
        # Light the graph with what the conversation has actually touched.
        touched = (self.context.entities | self.context.concepts)
        highlight = sorted(t for t in touched if t in self.graph.nodes)[:24]
        trace.add(
            stage="critic",
            title="Grounding Verification",
            summary="RELEASE — answer grounded in the conversation transcript",
            confidence=0.9,
        )
        trace.persist(
            DATA_DIR / "traces.jsonl",
            query=query,
            answer_source=answer_source,
            verdict="RELEASE — CONVERSATION MEMORY",
            confidence=0.9,
            generation=llm.last_generation(),
        )
        return {
            "request_id": trace.request_id,
            "query": query, "anchor": None, "plan": [], "trace": trace.export(),
            "findings": {}, "citations": [],
            "confidence": 0.9, "verdict": "RELEASE — CONVERSATION MEMORY",
            "confidence_factors": None,
            "answer": answer, "answer_source": answer_source,
            "evidence_diagnostics": None,
            "graph_highlight": highlight,
            "memory_answer": True,
            "response_profile": response_engine.analyze(query).as_dict(),
            "followups": _extract_followups(answer),
        }

    def run_case(self, query: str, history: list | None = None) -> dict:
        trace = ReasoningTrace()
        history = history or []
        original_query = query
        # Questions ABOUT the conversation itself ("what did we discuss?",
        # "summarize this chat") are answered from the transcript — the corpus
        # can never ground them, so the refusal gate would wrongly fire.
        if history and convmem.is_meta_query(query):
            return self._conversation_case(query, history, trace)
        query = self.context.rewrite(query)
        # Adaptive Response Engine (Module 02): detect HOW the user wants the
        # answer — output format, reading level, length, persona — FIRST, then
        # strip the styling phrases out of the retrieval query. This must run
        # BEFORE spell correction: styling words are the user's literal request
        # and must not be "corrected" (e.g. spell-check rewrote "bullet" →
        # "built", which both lost the bullet-format request and corrupted
        # retrieval). Stripping first means spell-check only ever sees the real
        # question.
        response_profile = response_engine.analyze(query)
        style_directives = response_profile.directives
        if not response_profile.is_empty:
            stripped = response_profile.stripped_query
            detected = {k: v for k, v in {
                "persona": response_profile.persona,
                "reading_level": response_profile.reading_level,
                "length": response_profile.length,
                "formats": ", ".join(response_profile.formats) or None,
            }.items() if v}
            print(f"Adaptive response profile: {detected}; "
                  f"retrieval query: {stripped!r}")
            trace.add(
                stage="style",
                title="Adaptive Response Engine",
                summary=("Response adapted to "
                         + (", ".join(f"{k}={v}" for k, v in detected.items())
                            or "default style")),
                directives=style_directives,
                retrieval_query=stripped,
                **detected,
            )
            if stripped != query:
                query = stripped
        # Spell correction BEFORE any intent/entity/aspect logic. One typo
        # ("differece") otherwise defeats the "difference between" intent
        # pattern, so the query is misrouted as definition, no comparison
        # aspects are extracted, and every downstream stage degrades.
        if self.spell is not None:
            sp = self.spell.correct(query)
            if sp.corrections:
                query = sp.corrected
                fixes = "; ".join(f"{c.original} → {c.corrected}"
                                  for c in sp.corrections)
                print(f"Spell corrections: {fixes} "
                      f"(confidence {sp.confidence})")
                trace.add(
                    stage="spell_correction",
                    title="Spell Correction",
                    summary=fixes,
                    confidence=sp.confidence,
                    corrected_query=query,
                )
        query_plan = self.query_processor.process(query)
        print("\n=== QUERY PLAN ===")
        print("Original query:", query_plan.original_query)
        print("Rewritten query:", query_plan.rewritten_query)
        print("Expanded query:", query_plan.expanded_query)
        print("Intent:", query_plan.intent)
        print("Entities:", query_plan.entities)
        print("Retrieval queries:", query_plan.retrieval_queries)
        # findings["query_plan"] = query_plan
        if original_query != query:

            trace.add(
                    stage="conversation",
                    title="Conversation Context",
                    summary="Follow-up query rewritten",
                    original_query=original_query,
                    rewritten_query=query,
                )

        sup = self._supervisor(query)
        anchor = sup["anchor"]
        # Conversational continuity: if this reads like a follow-up (no document
        # or asset detected on its own) but there's prior context, inherit routing
        # from the recent user turns so "explain more" stays on the same document.
        followup = False
        if (original_query == query and not anchor and not sup["target_docs"] and history):
            prev = " ".join(h["content"] for h in history[-4:] if h.get("role") == "user")
            if prev:
                ctx = self._supervisor(prev + " " + query)
                if ((ctx["target_docs"] or ctx["anchor"])
                        and not self._followup_topic_shift(
                            query, ctx["target_docs"])):
                    sup["target_docs"] = ctx["target_docs"]
                    sup["entities"] = sup["entities"] or ctx["entities"]
                    anchor = ctx["anchor"]
                    followup = True
        known_entities = {e for c in self.corpus.chunks for e in c.entities}
        entity_grounded = any(e in known_entities for e in sup["entities"])
        trace.add(
                    stage="supervisor",
                    title="Query Understanding",
                    summary=f"Intent classified (anchor={anchor or 'None'})",
                    followup=followup,
                    entities=sup["entities"],
                    target_documents=list(sup["target_docs"]),
                )
        # Graph-first: detect entities/concepts and traverse the graph BEFORE
        # retrieval, so vectors index into a graph-expanded context.
        gr = self._graph_reasoning(query, sup["entities"])
        if gr["seeds"]:
            trace.add(
                        stage="graph",
                        title="Knowledge Graph",
                        summary=gr["summary"],
                        seeds=gr["seeds"],
                        nodes=list(gr["nodes"]),
                        edges=len(gr["edges"]),
                        paths=gr["paths"],
                    )
        plan = self._planner(query, anchor)
        trace.add(
                    stage="planner",
                    title="Retrieval Planning",
                    summary=f"{len(plan['subqueries'])} retrieval queries generated",
                    strategy=plan["strategy"],
                    tasks=plan["tasks"],
                    subqueries=plan["subqueries"],
                )
        findings: dict = {
                    "query_plan": query_plan,
                }
        if anchor:
            findings["predictive"] = self._predictive_agent()
            p = findings["predictive"]
            trace.add(
                        stage="predictive",
                        title="Predictive Analysis",
                        summary=(
                            f"Vibration {p['latest_vibration']} mm/s "
                            f"(baseline {p['baseline_vibration']}), "
                            f"trend +{p['trend_mm_s_per_day']} mm/s/day, "
                            f"RUL ≈ {p['rul_days']} days"
                        ),
                    )
            findings["rootcause"] = self._rootcause_agent(anchor)
            rc = findings["rootcause"]
            cause = rc["candidate_cause"]["likely_cause"] if rc["candidate_cause"] else "unknown"
            trace.add(
                    stage="rootcause",
                    title="Root Cause Analysis",
                    summary=(
                        f"Graph traversal: {anchor} connects to "
                        f"{', '.join(rc['connected_to'])}; "
                        f"{len(rc['history'])} past work orders; "
                        f"likely cause: {cause}"
                    ),
                )

        # Retrieval graph-leg is seeded by BOTH the asset subgraph (asset cases)
        # and the GraphReasoner's traversal (concept/document cases).
        graph_entities = set(findings.get("rootcause", {}).get("subgraph_nodes", [])) | gr["nodes"]
        evidence = self._knowledge_agent(query=query, query_plan=query_plan, graph_entities=graph_entities,plan=plan,target_docs=sup["target_docs"],)

        print("\n=== FINAL RETRIEVED EVIDENCE ===")

        for i, item in enumerate(evidence, 1):
            print(f"\n--- Evidence {i} ---")
            print("Document:", item.get("doc_title"))
            print("Section:", item.get("section"))
            print("Score:", item.get("score"))
            print("RRF score:", item.get("rrf_score"))
            print("Reranker score:", item.get("reranker_score"))
            print("Matched entities:",
                  ", ".join(item.get("matched_entities") or []) or "—")
            print("Selected because:", item.get("selected_because"))
            print("Text:", item.get("text", "")[:500])
        findings["evidence"] = evidence
        # Promote the monitored asset's own structured data (live telemetry +
        # maintenance work-order history) into the evidence set so it is
        # citable by the LLM and verifiable by the validator. Appended after
        # the retrieved chunks, so their evidence ids are unaffected.
        if anchor:
            data_ev = self._asset_data_evidence(anchor, findings)
            if data_ev:
                evidence.extend(data_ev)
                findings["evidence"] = evidence
                print(f"Promoted {len(data_ev)} structured data evidence "
                      f"item(s) for {anchor} (telemetry + maintenance history)")
        findings["evidence_sufficiency"] = (
                    self._evidence_sufficiency(
                        evidence=evidence,
                        comparison_aspects=getattr(
                            query_plan,
                            "comparison_aspects",
                            [],
                        ),
                    )
                )
        if evidence:
            self.context.update(
                evidence=evidence,
                intent=query_plan.intent,
                query=original_query,
            )
        focus = self._retrieval_focus
        diag = self._retrieval_diagnostics
        note = "hybrid retrieval (dense+BM25+graph, RRF): "
        if diag:
            note += f"retrieved {diag['retrieved']}, used {diag['used']}, rejected {diag['rejected']}"
        else:
            note += f"{len(evidence)} passages"
        note += " from " + ", ".join(sorted({e["doc_no"] for e in evidence}))
        if focus:
            note += f"; document-focused on {focus['document']}"
            if focus["rejected_documents"]:
                note += f"; set aside {', '.join(focus['rejected_documents'])}"
        trace.add(
                    stage="retrieval",
                    title="Hybrid Retrieval",
                    summary=note,
                    retrieval_focus=focus,
                    diagnostics=diag,
                    retrieved_documents=sorted({e["doc_no"] for e in evidence}),
                )

        if anchor:
            findings["maintenance"] = self._maintenance_agent(anchor, findings["predictive"])
            m = findings["maintenance"]
            trace.add(
                        stage="maintenance",
                        title="Maintenance Planning",
                        summary=(
                            f"Spares in stock: {m['spares_in_stock']}; "
                            f"draft work order prepared "
                            f"(schedule within "
                            f"{m['draft_work_order']['schedule_within_days']} days)"
                        ),
                    )

            findings["safety"] = self._safety_agent(anchor)
            trace.add(
                        stage="safety",
                        title="Safety Assessment",
                        summary=findings["safety"]["statement"],
                        hard_gate=findings["safety"]["hard_gate"],
                    )

            findings["risk"] = self._risk_agent(findings["predictive"], findings["rootcause"],
                                                findings["maintenance"], findings["safety"])
            trace.add(
                        stage="risk",
                        title="Risk Decision",
                        summary=findings["risk"]["recommendation"],
                    )

        # Grounding gate (grounded-or-refuse): decide whether to attempt an answer
        # at all. Asset cases always have telemetry + graph findings to ground on;
        # document questions must clear a relevance/coverage/target bar.
        if anchor:
            grounded = True
        else:
            g = self.index.grounding(query)
            term_grounded = g["rarest_present"] and g["coverage"] >= 0.6
            grounded = bool(evidence) and (entity_grounded or bool(sup["target_docs"])
                                           or term_grounded or self.index.relevance(query) >= 0.2)

        if not grounded:
            findings["evidence"] = evidence = []
            # findings["evidence_sufficiency"] = (
            #                     self._evidence_sufficiency(evidence)
            #                 )
            answer, answer_source = self._refusal_message(), "grounded-or-refuse"
            critic = {"checks": [], "confidence": 0.1,
                      "verdict": "REFUSE — NO GROUNDED EVIDENCE", "policy": "grounded-or-refuse"}
        else:
            # REASON -> synthesise the answer, then VERIFY it (Chain-of-Verification).
            planner = AnswerPlanner()

            answer_plan = planner.build(
                query=query,
                query_plan=query_plan,
                evidence=evidence,
            )

            findings["answer_plan"] = answer_plan

            answer, answer_source = self._synthesize(
                query,
                anchor,
                findings,
                # Curated memory: summary of older turns + older turns relevant
                # to THIS query (recall) + the recent turns verbatim. This is
                # what lets a long chat connect back to earlier discussion.
                convmem.curate(query, history),
                style_directives=style_directives,
                profile=response_profile,
            )

            # =========================================================
            # STEP 1 — Initial deterministic validation
            # =========================================================

            answer, validation = (
                        self.validator.validate_and_repair(
                            answer=answer,
                            evidence=evidence,
                        )
                    )
                        
            print("\n=== ANSWER VALIDATION ===")

            for claim in validation.get("claims", []):
                if claim.get("status") != "SUPPORTED":
                    print("\nWEAK OR UNSUPPORTED CLAIM:")
                    print("Claim:", claim.get("claim"))
                    print("Status:", claim.get("status"))
                    print("Support score:", claim.get("support_score"))
                    print("Citation status:", claim.get("citation_status"))
                    print("Alignment score:", claim.get("alignment_score"))
                    print("Best evidence:", claim.get("best_evidence"))


            # =========================================================
            # STEP 2 — Validation is advisory only
            # =========================================================

            citation_repairs = validation.get(
                "citation_repairs",
                0,
            )

            repair_attempted = citation_repairs > 0
            repair_accepted = citation_repairs > 0

            initial_coverage = validation.get(
                    "initial_coverage",
                    validation.get("coverage", 0.0),
                )
            if repair_attempted:
                print(
                    f"\n🔧 Citation repair applied: "
                    f"{citation_repairs} citation(s) added."
                )

            # =========================================================
            # STEP 2b — LLM self-correction (previously dead code:
            # _repair_answer existed but was never called). One rewrite
            # attempt grounded in the same case file; the rewrite is
            # kept only if re-validation shows improved coverage.
            # =========================================================

            if not validation["valid"]:
                answer, validation = self._llm_self_correct(
                    query,
                    answer,
                    findings,
                    validation,
                    convmem.curate(query, history),
                )
                if validation.get("llm_repair"):
                    print(f"\n🔁 LLM self-correction: {validation['llm_repair']}")

            # =========================================================
            # STEP 2c — Confidence floor. When even after citation repair
            # and self-correction the answer's claims are predominantly
            # ungrounded, DO NOT release it with a warning banner: replace
            # it with an honest insufficient-evidence response. Measured
            # failure: 'tell me about the gpu-graph accelerator' produced
            # a fully fabricated hardware-accelerator definition (coverage
            # 0.10, every claim INSUFFICIENT) that shipped as RELEASE.
            # =========================================================

            # Two independent suppression conditions:
            #  (a) low average coverage with more ungrounded than grounded
            #      claims — catches wholly-fabricated answers (coverage ~0.1);
            #  (b) MAJORITY of factual claims ungrounded, at ANY average
            #      coverage — catches out-of-corpus questions where the topic
            #      IS in the corpus so the answer carries grounded background
            #      ("LangSmith is for tracing [1]") that lifts average
            #      coverage to ~0.55, while the claim that actually answers
            #      the question (the pricing, the CEO) is fabricated.
            #      Measured: leaked negatives sit at ungrounded-ratio 0.5;
            #      valid answers at <=0.4, so 0.5 separates them and keeps
            #      genuine partial answers.
            COVERAGE_FLOOR = 0.35
            UNGROUNDED_MAJORITY = 0.5
            used_asset_template = False
            total_claims = len(validation.get("claims", []))
            insufficient = validation.get("insufficient_evidence_claims", 0)
            supported = validation.get("supported_claims", 0)

            # Fabrication vs. grounded-synthesis. The floor exists to suppress
            # answers whose content is NOT in the corpus (fabrication), not
            # well-grounded answers a weak model merely under-cited or
            # mis-cited. A fabricated claim's invented specifics (a price, a
            # name, a made-up term) appear in no chunk, so its best evidence
            # has LOW lexical AND LOW semantic support; grounded synthesis has
            # HIGH both. NLI entailment against a single retrieved chunk is
            # near-zero for a claim synthesized across pages, which tanks the
            # support score — but that is not fabrication. So a claim counts as
            # fabricated only when its best evidence does NOT strongly cover it,
            # or the evidence actively contradicts it (entity swap).
            GROUNDED_LEXICAL = 0.5
            GROUNDED_SEMANTIC = 0.85

            def _corpus_grounded(claim: dict) -> bool:
                be = claim.get("best_evidence") or {}
                lex = be.get("lexical_score") or 0.0
                sem = be.get("semantic_score") or 0.0
                contra = claim.get("nli_contradiction") or 0.0
                return (lex >= GROUNDED_LEXICAL and sem >= GROUNDED_SEMANTIC
                        and contra < 0.5)

            claim_reports = validation.get("claims", [])
            fabricated = sum(
                1 for c in claim_reports
                if c.get("status") == "INSUFFICIENT_EVIDENCE"
                and not _corpus_grounded(c))
            grounded = total_claims - fabricated  # supported + grounded synthesis
            ungrounded_ratio = (fabricated / total_claims
                                if total_claims else 0.0)
            low_coverage = (validation.get("coverage", 1.0) < COVERAGE_FLOOR
                            and fabricated > grounded)
            majority_ungrounded = (total_claims >= 2
                                   and ungrounded_ratio >= UNGROUNDED_MAJORITY
                                   and fabricated >= grounded)
            if (not validation.get("valid")
                    and (low_coverage or majority_ungrounded)):
                # A monitored asset with a diagnosed root cause is NOT an
                # ungrounded case: its facts (telemetry + maintenance history)
                # are held in the system of record. The generic LLM draft was
                # suppressed only because a small model under-cited them. Rather
                # than refuse, answer from the deterministic, fully-cited asset
                # template — strictly better than both the draft and a refusal,
                # and grounded in the same data Asset360 shows.
                if (anchor and findings.get("predictive")
                        and findings.get("rootcause", {}).get("candidate_cause")):
                    answer = self._template_answer(query, anchor, findings,
                                                   response_profile)
                    answer_source = "asset-data"
                    used_asset_template = True
                    validation["confidence_floor"] = (
                        "draft suppressed; answered from the grounded asset-data "
                        "template (live telemetry + maintenance history)")
                    print("⛔→✓ Confidence floor: replaced under-cited draft "
                          "with the grounded asset-data template")
                else:
                    docs = sorted({e.get("doc_no", "?") for e in evidence})[:3]
                    answer = (
                        "## Insufficient Evidence\n\n"
                        "The retrieved documents do not contain enough grounded "
                        "information to answer this question reliably, so no "
                        "speculative answer is being generated.\n\n"
                        + (f"Closest material found: {', '.join(docs)}.\n\n"
                           if docs else "")
                        + "Try rephrasing the question, or name the document "
                          "you mean (e.g. \"in the person_resume, ...\").")
                    answer_source = "confidence-floor"
                    reason = ("majority of claims not grounded in the corpus "
                              f"({fabricated}/{total_claims})"
                              if majority_ungrounded else
                              f"coverage {validation.get('coverage', 0.0):.2f} "
                              f"< {COVERAGE_FLOOR}")
                    validation["confidence_floor"] = (
                        f"answer suppressed — {reason} "
                        f"({fabricated} fabricated claim(s))")
                    print(f"⛔ Confidence floor: {validation['confidence_floor']}")

            if not validation["valid"]:
                print(
                    "\n⚠️ Validation warning: "
                    "answer still contains weak or unsupported claims "
                    "after citation repair."
                )

                trace.add(
                        stage="answer_validation_warning",
                        title="Answer Validation Warning",
                        summary="Answer validation detected weak or unsupported evidence claims",
                        coverage=validation.get(
                            "coverage",
                            0.0,
                        ),
                        evidence_verdict=validation.get(
                            "evidence_verdict",
                            "UNKNOWN",
                        ),
                        issues=validation.get(
                            "issues",
                            [],
                        ),
                        insufficient_evidence_claims=validation.get(
                            "insufficient_evidence_claims",
                            0,
                        ),
                        mismatched_citations=validation.get(
                            "mismatched_citations",
                            0,
                        ),
                    )

            # =========================================================
            # STEP 3 — Store FINAL validation
            # =========================================================

            validation["repair_attempted"] = repair_attempted
            validation["repair_accepted"] = repair_accepted
            validation["initial_coverage"] = initial_coverage

            findings["validation"] = validation

            # =========================================================
            # STEP 4 — Final multi-factor critic
            # =========================================================

            critic = self._verify_answer(
                query,
                answer,
                self._case_file_text(findings),
                evidence,
                gr["seeds"],
            )

            # When the confidence floor already replaced the answer with an
            # insufficient-evidence response, the critic must report a
            # refusal, not grade the (now-removed) draft. Otherwise the UI
            # shows "RELEASE" over an "Insufficient Evidence" body.
            if answer_source == "confidence-floor":
                critic = {"checks": [], "confidence": 0.15,
                          "verdict": "REFUSE — INSUFFICIENT GROUNDED EVIDENCE",
                          "policy": "confidence-floor",
                          "unsupported": 0}
            # The asset-data template is hand-composed from the system of
            # record (telemetry + maintenance log) and cites the retrieved
            # procedures — it is grounded by construction, so release it
            # rather than grade the discarded LLM draft.
            elif used_asset_template:
                critic = {"checks": [], "confidence": 0.82,
                          "verdict": "RELEASE — GROUNDED IN ASSET DATA",
                          "policy": "asset-data-template",
                          "unsupported": 0}

            # =========================================================
            # STEP 5 — Add warning only to final answer
            # =========================================================

            if (
                answer_source != "confidence-floor"
                and critic.get("unsupported", 0) > 0
                and critic["verdict"] != "RELEASE"
            ):

                answer += (
                    "\n\n⚠️ Verification Notice\n"
                    f"{critic['unsupported']} claim(s) are weakly "
                    "supported by the retrieved evidence."
                )

        findings["critic"] = critic
        trace.add(
                stage="critic",
                title="Grounding Verification",
                summary=critic["verdict"],
                confidence=critic["confidence"],
                factors=critic.get("factors"),
                checks=critic["checks"],
            )
        
        validation = findings.get("validation")

        if validation:

            weak_claims = (validation.get("partially_supported_claims", 0)
                           + validation.get("insufficient_evidence_claims", 0))
            trace.add(
                        stage="validator",
                        title="Answer Validation",
                        summary=(
                            f"All {validation.get('supported_claims', 0)} claims supported "
                            f"(coverage {validation.get('coverage', 0.0):.2f})"
                            if weak_claims == 0 and validation.get("valid", False)
                            else f"Validation complete — {weak_claims} weak claim(s), "
                                 f"coverage {validation.get('coverage', 0.0):.2f}, "
                                 f"valid={validation.get('valid', False)}"
                        ),
                        status="success" if weak_claims == 0 else "warning",

                        coverage=validation["coverage"],
                        valid=validation["valid"],

                        evidence_verdict=validation["evidence_verdict"],

                        supported_claims=validation["supported_claims"],

                        partially_supported_claims=validation[
                            "partially_supported_claims"
                        ],

                        insufficient_evidence_claims=validation[
                            "insufficient_evidence_claims"
                        ],
                        repair_attempted=validation.get(
                            "repair_attempted",
                            False,
                        ),

                        repair_accepted=validation.get(
                            "repair_accepted",
                            False,
                        ),

                        initial_coverage=validation.get(
                            "initial_coverage",
                            validation["coverage"],
                        ),

                        final_coverage=validation["coverage"],

                        citation_alignment_score=validation.get(
                            "citation_alignment_score",
                            0.0,
                        ),

                        mismatched_citations=validation.get(
                            "mismatched_citations",
                            0,
                        ),
                    )

        # One structured JSON line per request: stages, latency breakdown,
        # and outcome summary — the substrate for offline error analysis,
        # latency bottleneck hunting, and regression triage.
        validation = findings.get("validation") or {}
        trace.persist(
            DATA_DIR / "traces.jsonl",
            query=original_query,
            intent=getattr(findings.get("query_plan"), "intent", None),
            answer_source=answer_source,
            verdict=critic["verdict"],
            confidence=critic["confidence"],
            coverage=validation.get("coverage"),
            supported_claims=validation.get("supported_claims"),
            insufficient_claims=validation.get("insufficient_evidence_claims"),
            llm_repair=validation.get("llm_repair"),
            evidence_count=len(evidence),
            generation=llm.last_generation(),
        )

        return {
            "request_id": trace.request_id,
            "query": query, "anchor": anchor, "plan": plan["tasks"], "trace": trace.export(),
            "findings": findings, "citations": self._citations(evidence),
            "confidence": critic["confidence"], "verdict": critic["verdict"],
            "confidence_factors": critic.get("factors"),
            "answer": answer, "answer_source": answer_source,
            "evidence_diagnostics": self._retrieval_diagnostics,
            "graph_highlight": self._support_graph(query, answer, evidence, anchor, findings),
            # Adaptive Response Engine: what style was detected, and structured
            # follow-ups so the UI never depends on parsing the answer text.
            "response_profile": response_profile.as_dict(),
            "followups": _extract_followups(answer, sup.get("entities"), anchor),
        }

    # ------------------------------------------------------ vanilla baseline

    def vanilla_rag(self, query: str) -> dict:
        """Naive RAG baseline: pure dense top-k across ALL documents, then stuff
        the passages into the LLM. No graph, no document routing, no hybrid
        fusion, no reranking, no verification — the thing AXON improves on."""
        hits = self.index.dense_retrieve(query, k=6)
        docs = sorted({h["doc_no"] for h in hits})
        context = "\n\n".join(f"[{i + 1}] ({h['doc_no']} — {h['section']}) {h['text'][:450]}"
                              for i, h in enumerate(hits))
        answer = llm.synthesize_vanilla(query, context) if hits else None
        if not answer:
            answer = ("Top passages (no synthesis available):\n" +
                      "\n".join(f"[{i + 1}] {h['doc_no']} — {h['section']}" for i, h in enumerate(hits[:4]))
                      if hits else "No passages found.")
        return {"method": "Vanilla RAG — dense top-k, no graph, no verification",
                "documents": docs, "documents_touched": len(docs),
                "graph_used": False, "verified": False,
                "citations": [{"n": i + 1, "doc_no": h["doc_no"], "section": h["section"]}
                              for i, h in enumerate(hits)],
                "answer": answer}

    def compare(self, query: str) -> dict:
        """Run the vanilla RAG baseline and the full AXON GraphRAG pipeline on the
        same query so the difference is measurable, not asserted."""
        axon = self.run_case(query)
        vanilla = self.vanilla_rag(query)
        gr = next((t for t in axon["trace"] if t["agent"] == "GraphReasoner"), None)
        axon_docs = sorted({c["doc_no"] for c in axon["citations"]})
        return {
            "query": query,
            "vanilla": vanilla,
            "axon": {
                "method": "AXON GraphRAG — graph traversal + document focus + verification",
                "documents": axon_docs, "documents_touched": len(axon_docs),
                "graph_used": bool(gr), "graph_nodes": len(gr["nodes"]) if gr and "nodes" in gr else len(axon.get("graph_highlight", [])),
                "verified": True, "confidence": axon["confidence"], "verdict": axon["verdict"],
                "citations": axon["citations"], "answer": axon["answer"],
                "trace": axon["trace"], "graph_highlight": axon["graph_highlight"],
            },
        }

    # -------------------------------------------------------------- synthesis

    @staticmethod
    def _citations(evidence: list[dict]) -> list[dict]:
        return [{"n": i + 1, "doc_no": e["doc_no"], "revision": e["revision"],
                 "section": e["section"], "doc_title": e["doc_title"]}
                for i, e in enumerate(evidence)]

    # ---------------------------------------------------------------- prompt
    # Evidence rendering budgets for the CASE FILE. Per-chunk limit keeps one
    # passage from monopolizing the prompt; the global budget bounds the whole
    # evidence block (~6k tokens at 4 chars/token) regardless of how many
    # chunks selection produced. Both cut at sentence boundaries.
    EVIDENCE_TEXT_LIMIT = 1800
    EVIDENCE_CHAR_BUDGET = 24000

    @staticmethod
    def _clip_sentences(text: str, limit: int) -> str:
        """Clip text to <= limit chars at a sentence boundary (falling back
        to a word boundary), so the LLM never sees a passage cut mid-word —
        truncated evidence reads as if the source itself stops there."""
        if len(text) <= limit:
            return text
        cut = text[:limit]
        # Prefer the last sentence end in the window; require it past 40% of
        # the limit so one early period doesn't discard most of the budget.
        best = max(cut.rfind(". "), cut.rfind(".\n"),
                   cut.rfind("! "), cut.rfind("? "))
        if best >= int(limit * 0.4):
            return cut[: best + 1]
        ws = cut.rfind(" ")
        return (cut[:ws] if ws > 0 else cut) + " …"

    def _case_file_text(
    self,
    findings: dict,
    query: str | None = None,
) -> str:
        """
        Build a structured evidence package for the LLM.

        The goal is to make evidence provenance explicit so the LLM can
        reason over it instead of treating every chunk equally.
        """

        lines = [] 


        # ============================================================
        # Query Understanding
        # ============================================================

        query_plan = findings.get("query_plan")
        sufficiency = findings.get("evidence_sufficiency", {})

        if query:

            lines.append("=" * 70)
            lines.append("USER QUESTION")
            lines.append("=" * 70)
            lines.append(query)

        if query_plan:

            lines.append("")
            lines.append("=" * 70)
            lines.append("QUESTION ANALYSIS")
            lines.append("=" * 70)

            lines.append(
                f"Intent: {getattr(query_plan, 'intent', 'Unknown')}"
            )

            aspects = getattr(
                query_plan,
                "comparison_aspects",
                [],
            )

            if aspects:
                lines.append(
                    "Comparison Aspects: "
                    + ", ".join(aspects)
                )

        # ============================================================
        # Evidence Coverage
        # ============================================================

        if sufficiency:

            lines.append("")
            lines.append("=" * 70)
            lines.append("EVIDENCE COVERAGE")
            lines.append("=" * 70)

            lines.append(
                f"Overall Sufficient: "
                f"{sufficiency.get('sufficient', False)}"
            )

            lines.append(
                f"Coverage Score: "
                f"{sufficiency.get('score', 0.0)}"
            )

            aspect_coverage = sufficiency.get(
                "aspect_coverage",
                {},
            )

            if aspect_coverage:

                lines.append("")

                for aspect, data in aspect_coverage.items():

                    lines.append(
                        f"- {aspect}: "
                        f"{data.get('level', 'UNKNOWN')} "
                        f"({data.get('substantive_chunks', 0)} "
                        f"substantive chunk(s))"
                    )

        # ============================================================
        # Predictive Intelligence
        # ============================================================

        if "predictive" in findings:
            p = findings["predictive"]

            lines.append("=" * 70)
            lines.append("PREDICTIVE ANALYSIS")
            lines.append("=" * 70)

            lines.append(
                f"Current vibration : {p['latest_vibration']} mm/s\n"
                f"Baseline vibration: {p['baseline_vibration']} mm/s\n"
                f"Trend             : +{p['trend_mm_s_per_day']} mm/s/day\n"
                f"Alert limit       : {p['alert_limit']} mm/s\n"
                f"Danger limit      : {p['danger_limit']} mm/s\n"
                f"Estimated RUL     : {p['rul_days']} days\n"
                f"Bearing temp      : {p['bearing_temp_recent_c']} °C\n"
                f"Anomaly           : {'YES' if p['anomaly'] else 'NO'}"
            )

        # ============================================================
        # Root Cause
        # ============================================================

        rc = findings.get("rootcause", {})

        if rc.get("candidate_cause"):

            lines.append("")
            lines.append("=" * 70)
            lines.append("ROOT CAUSE ANALYSIS")
            lines.append("=" * 70)

            for item in rc["candidate_cause"]["evidence_chain"]:
                lines.append(f"• {item}")

        # ============================================================
        # Maintenance
        # ============================================================

        if "maintenance" in findings:

            m = findings["maintenance"]

            lines.append("")
            lines.append("=" * 70)
            lines.append("MAINTENANCE RECOMMENDATION")
            lines.append("=" * 70)

            lines.append(f"Spares Available : {m['spares_in_stock']}")

            wo = m["draft_work_order"]

            lines.append(f"Draft WO         : {wo['title']}")
            lines.append(f"Schedule         : Within {wo['schedule_within_days']} days")

        # ============================================================
        # Safety
        # ============================================================

        safety = findings.get("safety", {})

        if safety.get("hard_gate"):

            lines.append("")
            lines.append("=" * 70)
            lines.append("SAFETY REQUIREMENTS")
            lines.append("=" * 70)

            for req in safety["requirements"]:

                lines.append(
                    f"• {req['procedure']} requires {req['permit']}"
                )

        # ============================================================
        # Risk Recommendation
        # ============================================================

        if "risk" in findings:

            lines.append("")
            lines.append("=" * 70)
            lines.append("FINAL RECOMMENDATION")
            lines.append("=" * 70)

            lines.append(findings["risk"]["recommendation"])

        # ============================================================
        # Evidence Summary
        # ============================================================

        evidence = findings.get("evidence", [])
        for i, chunk in enumerate(evidence, start=1):
            chunk.setdefault("evidence_id", i)

        lines.append("")
        lines.append("=" * 70)
        lines.append("EVIDENCE SUMMARY")
        lines.append("=" * 70)

        lines.append(f"Evidence Retrieved : {len(evidence)} chunks")
        lines.append("Retrieval Method   : Hybrid Dense + BM25 + Graph + RRF")

        # These become available after retrieval.py improvements
        graph_used = any(e.get("graph_score") for e in evidence)

        lines.append(f"Graph Expansion    : {'YES' if graph_used else 'NO'}")

        # ============================================================
        # Evidence Package
        # ============================================================

        # ============================================================
        # Organized Evidence Package
        # ============================================================

        organizer = EvidenceOrganizer()

        organized = organizer.organize(evidence)

        lines.append("")
        lines.append("=" * 70)
        lines.append("ORGANIZED EVIDENCE")
        lines.append("=" * 70)

        # Character budget for the evidence block. The system prompt tells
        # the model that retrieval scores are internal diagnostics it should
        # ignore — so raw Final/Reranker/RRF numbers, retrieval methods and
        # per-chunk Summary/Keywords/Concepts lines (the summary is the
        # chunk's own first paragraph, keywords/concepts are already inside
        # the text) were pure token waste. Each chunk now carries only what
        # the model can use: id, a qualitative relevance hint, entities,
        # content, and source. Content is clipped at a sentence boundary at
        # EVIDENCE_TEXT_LIMIT (the old 700-char mid-word cut threw away most
        # of the merged/neighbor-expanded passages retrieval built), and the
        # whole block stops at EVIDENCE_CHAR_BUDGET with an explicit note.
        rendered = omitted = 0
        used_chars = 0
        for document in organized:

            lines.append("")
            lines.append(f"DOCUMENT: {document['document']}")
            lines.append("-" * 70)

            for section in document["sections"]:

                lines.append("")
                lines.append(f"SECTION: {section['section']}")

                for chunk in section["chunks"]:

                    if used_chars >= self.EVIDENCE_CHAR_BUDGET:
                        chunk["prompt_included"] = False
                        omitted += 1
                        continue

                    entry = []
                    score = chunk.get("score", 0)
                    relevance = ("high" if score >= 0.55
                                 else "medium" if score >= 0.30 else "low")
                    entry.append("")
                    entry.append(f"[{chunk['evidence_id']}] Relevance: {relevance}")

                    if chunk.get("entities"):
                        entry.append(
                            "Entities: " + ", ".join(chunk["entities"][:10])
                        )

                    entry.append("Content:")
                    entry.append(self._clip_sentences(
                        chunk.get("text", "").strip(),
                        self.EVIDENCE_TEXT_LIMIT,
                    ))
                    entry.append(
                        f"Source: {chunk['doc_no']} | "
                        f"{chunk['section']} | "
                        f"rev {chunk['revision']}"
                    )

                    chunk["prompt_included"] = True
                    rendered += 1
                    used_chars += sum(len(e) + 1 for e in entry)
                    lines.extend(entry)

        if omitted:
            lines.append("")
            lines.append(
                f"NOTE: {omitted} additional evidence passage(s) were "
                f"omitted for context-budget reasons. Do not cite evidence "
                f"identifiers that do not appear above."
            )

        # ============================================================
        # Evidence Gaps
        # ============================================================

        aspect_coverage = sufficiency.get(
            "aspect_coverage",
            {},
        )

        missing_aspects = [
            aspect
            for aspect, data in aspect_coverage.items()
            if data.get("level") in {
                "MISSING",
                "MENTION_ONLY",
            }
        ]

        if missing_aspects:

            lines.append("")
            lines.append("=" * 70)
            lines.append("EVIDENCE GAPS")
            lines.append("=" * 70)

            for aspect in missing_aspects:
                lines.append(
                    f"- Insufficient substantive evidence for: {aspect}"
        )

        

        return "\n".join(lines)
    
    def _aspect_evidence_strength(
    self,
    aspect: str,
    evidence: list[dict],
) -> dict:
        """
        Measure whether an aspect is merely mentioned
        or substantively described in the evidence.
        """

        aspect_lower = aspect.lower().strip()

        mention_count = 0
        substantive_count = 0
        supporting_chunks = []

        for chunk in evidence:

            text = " ".join([
                str(chunk.get("text", "")),
                str(chunk.get("summary", "")),
            ]).lower()

            # Aspect not present in actual textual evidence
            if aspect_lower not in text:
                continue

            mention_count += 1

            # -----------------------------------------------------
            # Substantive evidence heuristic
            # -----------------------------------------------------

            # Find sentences containing the aspect
            sentences = re.split(
                r"(?<=[.!?])\s+",
                text,
            )

            relevant_sentences = [
                sentence
                for sentence in sentences
                if aspect_lower in sentence
            ]

            # A sentence must contain enough surrounding information
            # to count as more than a simple name/reference.
            substantive = any(
                len(sentence.split()) >= 8
                for sentence in relevant_sentences
            )

            if substantive:

                substantive_count += 1

                supporting_chunks.append(
                    chunk.get("chunk_id")
                )

        if substantive_count >= 2:
            level = "STRONG"

        elif substantive_count == 1:
            level = "PARTIAL"

        elif mention_count > 0:
            level = "MENTION_ONLY"

        else:
            level = "MISSING"

        return {
            "aspect": aspect,
            "level": level,
            "mention_count": mention_count,
            "substantive_chunks": substantive_count,
            "supporting_chunks": supporting_chunks,
        }




    def _evidence_sufficiency(
    self,
    evidence: list[dict],
    comparison_aspects: list[str] | None = None,
) -> dict:

        if not evidence:
            return {
                "score": 0.0,
                "sufficient": False,
                "aspect_coverage": {},
            }

        strong_chunks = sum(
            1
            for chunk in evidence
            if chunk.get(
                "rrf_score",
                chunk.get("score", 0),
            ) > 0
        )

        unique_sections = len({
            (
                chunk.get("doc_no"),
                chunk.get("section"),
            )
            for chunk in evidence
        })

        # ---------------------------------------------------------
        # Build searchable evidence text
        # ---------------------------------------------------------

        evidence_texts = []

        for chunk in evidence:

            searchable = " ".join([
                str(chunk.get("text", "")),
                str(chunk.get("summary", "")),
                " ".join(chunk.get("entities", [])),
                " ".join(chunk.get("concepts", [])),
                " ".join(chunk.get("keywords", [])),
            ]).lower()

            evidence_texts.append(searchable)

        
        # ---------------------------------------------------------
        # Substantive comparison aspect coverage
        # ---------------------------------------------------------

        aspect_coverage = {}

        for aspect in comparison_aspects or []:

            strength = self._aspect_evidence_strength(
                aspect=aspect,
                evidence=evidence,
            )

            aspect_coverage[aspect] = strength

        all_aspects_covered = (
                all(
                    item["level"] in {
                        "STRONG",
                        "PARTIAL",
                    }
                    for item in aspect_coverage.values()
                )
                if aspect_coverage
                else True
            )

        strong_aspect_coverage = (
                all(
                    item["level"] == "STRONG"
                    for item in aspect_coverage.values()
                )
                if aspect_coverage
                else True
            )


        # ---------------------------------------------------------
        # Base evidence score
        # ---------------------------------------------------------

        score = min(
            1.0,
            (
                0.6 * min(strong_chunks / 4, 1.0)
                +
                0.4 * min(unique_sections / 3, 1.0)
            ),
        )

        # Comparison cannot be fully sufficient if one side is missing
        sufficient = (
            strong_chunks >= 3
            and unique_sections >= 2
            and all_aspects_covered
        )

        return {
                "score": round(score, 2),
                "sufficient": sufficient,
                "aspect_coverage": aspect_coverage,
                "all_aspects_covered": all_aspects_covered,
                "strong_aspect_coverage": strong_aspect_coverage,
}


    def _structured_evidence_package(
        self,
        query: str,
        findings: dict,
    ) -> str:
        
        """
        Build a structured evidence package for answer synthesis.

        This helps the LLM understand:
        - the user's question
        - question intent
        - comparison aspects
        - evidence hierarchy
        - evidence coverage
        - evidence gaps
        """

        evidence = findings.get("evidence", [])
        query_plan = findings.get("query_plan")
        sufficiency = findings.get(
            "evidence_sufficiency",
            {},
        )

        lines = []

        lines.append("=" * 70)
        lines.append("STRUCTURED EVIDENCE PACKAGE")
        lines.append("=" * 70)

        # ---------------------------------------------------------
        # Question
        # ---------------------------------------------------------

        lines.append("\nUSER QUESTION")
        lines.append(query)

        # ---------------------------------------------------------
        # Query understanding
        # ---------------------------------------------------------

        if query_plan:

            lines.append("\nQUESTION ANALYSIS")

            lines.append(
                f"Intent: {getattr(query_plan, 'intent', 'Unknown')}"
            )

            aspects = getattr(
                query_plan,
                "comparison_aspects",
                [],
            )

            if aspects:

                lines.append(
                    "Comparison Aspects: "
                    + ", ".join(aspects)
                )

        # ---------------------------------------------------------
        # Evidence sufficiency
        # ---------------------------------------------------------

        lines.append("\nEVIDENCE COVERAGE")

        lines.append(
            f"Overall Sufficient: "
            f"{sufficiency.get('sufficient', False)}"
        )

        lines.append(
            f"Coverage Score: "
            f"{sufficiency.get('score', 0.0)}"
        )

        aspect_coverage = sufficiency.get(
            "aspect_coverage",
            {},
        )

        if aspect_coverage:

            lines.append("\nAspect Coverage:")

            for aspect, data in aspect_coverage.items():

                lines.append(
                    f"- {aspect}: "
                    f"{data.get('level', 'UNKNOWN')} "
                    f"({data.get('substantive_chunks', 0)} "
                    f"substantive chunk(s))"
                )

        # ---------------------------------------------------------
        # Retrieved evidence
        # ---------------------------------------------------------

        lines.append("\n" + "=" * 70)
        lines.append("RETRIEVED EVIDENCE")
        lines.append("=" * 70)

        for number, chunk in enumerate(
            evidence,
            start=1,
        ):

            lines.append(
                f"\n[E{number}]"
            )

            lines.append(
                f"Document: "
                f"{chunk.get('doc_title', chunk.get('doc_no', 'Unknown'))}"
            )

            lines.append(
                f"Document ID: "
                f"{chunk.get('doc_no', 'Unknown')}"
            )

            lines.append(
                f"Section: "
                f"{chunk.get('section', 'Unknown')}"
            )

            if chunk.get("revision"):

                lines.append(
                    f"Revision: {chunk['revision']}"
                )

            # Retrieval diagnostics
            lines.append(
                f"Retrieval Method: "
                f"{chunk.get('retrieval_method', 'Unknown')}"
            )

            if "reranker_score" in chunk:

                lines.append(
                    f"Reranker Score: "
                    f"{chunk['reranker_score']}"
                )

            # Semantic metadata
            if chunk.get("entities"):

                lines.append(
                    "Entities: "
                    + ", ".join(
                        map(str, chunk["entities"])
                    )
                )

            if chunk.get("concepts"):

                lines.append(
                    "Concepts: "
                    + ", ".join(
                        map(str, chunk["concepts"])
                    )
                )

            # Actual evidence text
            lines.append("\nEvidence Text:")
            lines.append(
                str(chunk.get("text", "")).strip()
            )

        # ---------------------------------------------------------
        # Explicit evidence gaps
        # ---------------------------------------------------------

        missing_aspects = [
            aspect
            for aspect, data in aspect_coverage.items()
            if data.get("level") in {
                "MISSING",
                "MENTION_ONLY",
            }
        ]

        if missing_aspects:

            lines.append("\n" + "=" * 70)
            lines.append("EVIDENCE GAPS")
            lines.append("=" * 70)

            for aspect in missing_aspects:

                lines.append(
                    f"- Insufficient substantive evidence for: {aspect}"
                )

        return "\n".join(lines)
    


    def _llm_self_correct(
        self,
        query: str,
        answer: str,
        findings: dict,
        validation: dict,
        history: list | None = None,
    ) -> tuple[str, dict]:
        """Single-attempt LLM self-correction for an answer that failed
        deterministic validation:

            validate -> failed? -> LLM rewrite (grounded in the same case
            file + targeted diagnostics) -> re-validate -> ACCEPT ONLY IF
            COVERAGE IMPROVED, else keep the original.

        One attempt, monotone-improvement acceptance — no repair loops,
        no accepting a rewrite that scores worse than the original."""
        needs_repair = (
            not validation.get("valid")
            and (validation.get("insufficient_evidence_claims", 0) > 0
                 or validation.get("mismatched_citations", 0) > 0
                 or validation.get("invalid_citations"))
        )
        if not needs_repair:
            return answer, validation

        # Repair is a SELF-CONTAINED task: deliberately no conversation
        # history. With history included, the repair model latched onto the
        # previous turn and regenerated the previous question's answer
        # (caught by the coverage guard, but never feed the trap).
        repaired, provider = self._repair_answer(
            query, answer, findings, validation, history=None,
        )
        if not repaired:
            validation["llm_repair"] = "attempted — no repair output"
            return answer, validation

        # Small models echo the repair-task banner back into their output;
        # strip any preamble before the first markdown heading so instruction
        # text can never leak into an accepted answer.
        repaired = self._strip_repair_echo(repaired)

        repaired, revalidation = self.validator.validate_and_repair(
            answer=repaired,
            evidence=findings.get("evidence", []),
        )
        old_cov = validation.get("coverage", 0.0)
        new_cov = revalidation.get("coverage", 0.0)
        if new_cov > old_cov:
            revalidation["llm_repair"] = (
                f"accepted — coverage {old_cov:.2f} → {new_cov:.2f} "
                f"({provider})"
            )
            revalidation["initial_coverage"] = validation.get(
                "initial_coverage", old_cov,
            )
            return repaired, revalidation

        validation["llm_repair"] = (
            f"rejected — rewrite coverage {new_cov:.2f} did not improve "
            f"on {old_cov:.2f}; original answer kept"
        )
        return answer, validation

    @staticmethod
    def _strip_repair_echo(text: str) -> str:
        """Drop an echoed instruction preamble ('ANSWER REPAIR TASK' banner,
        'The previous draft failed evidence validation', bare ===== rules)
        that small models copy from the repair prompt. The real answer
        starts at the first markdown heading."""
        m = re.search(r"^#{1,3}\s", text, re.MULTILINE)
        if m and m.start() > 0:
            head = text[: m.start()]
            if re.search(
                r"ANSWER REPAIR TASK|previous draft failed|^\s*={4,}\s*$",
                head, re.IGNORECASE | re.MULTILINE,
            ):
                return text[m.start():].strip()
        return text.strip()

    def _repair_answer(
    self,
    query: str,
    answer: str,
    findings: dict,
    validation: dict,
    history: list | None = None,
) -> tuple[str | None, str]:
        """
        Repair an answer that failed deterministic validation.

        The repair is grounded in the same CASE FILE and receives
        targeted validation diagnostics.

        Only one repair attempt should be made by run_case().
        """

        case_file = self._case_file_text(findings)

        failed_claims = []

        for claim in validation.get("claims", []):

            if (
                claim.get("status") == "INSUFFICIENT_EVIDENCE"
                or claim.get("citation_status") == "MISMATCHED"
            ):
                failed_claims.append(
                    {
                        "claim": claim.get("claim", ""),
                        "support_status": claim.get("status"),
                        "citation_status": claim.get(
                            "citation_status"
                        ),
                        "best_evidence": claim.get(
                            "best_evidence"
                        ),
                    }
                )

        repair_instructions = f"""
    ========================================================
    ANSWER REPAIR TASK
    ========================================================

    The previous draft failed evidence validation.

    ORIGINAL QUESTION:
    {query}

    PREVIOUS DRAFT:
    {answer}

    VALIDATION RESULT:
    Evidence verdict:
    {validation.get("evidence_verdict")}

    Coverage:
    {validation.get("coverage")}

    Invalid citations:
    {validation.get("invalid_citations", [])}

    Mismatched citations:
    {validation.get("mismatched_citations", 0)}

    Unsupported claims:
    {validation.get("insufficient_evidence_claims", 0)}

    FAILED CLAIMS:
    {failed_claims}

    ========================================================
    REPAIR RULES
    ========================================================

    Rewrite the answer using ONLY the CASE FILE.

    For every failed claim:

    1. Correct it if the CASE FILE supports a corrected version.
    2. Remove it if the CASE FILE does not support it.
    3. If necessary, explicitly state that the evidence is insufficient.

    Citation rules:

    - Use only valid evidence IDs from the CASE FILE.
    - Place citations immediately after the claims they support.
    - Never reuse a citation merely because it discusses a similar topic.
    - Never invent evidence IDs.
    - Do not add unsupported factual claims while repairing the answer.

    Preserve useful parts of the previous answer only when they are
    supported by the CASE FILE.

    Return the complete repaired answer, not a critique of the draft.

    ========================================================
    CASE FILE
    ========================================================

    {case_file}
    """

        repaired_answer = llm.synthesize(
            query=repair_instructions,
            case_file=case_file,
            history=history,
        )

        provider = llm.last_generation().get("provider") or llm.ACTIVE_PROVIDER
        if not repaired_answer:
            return None, provider

        return repaired_answer.strip(), provider





    def _synthesize(
    self,
    query: str,
    anchor: str | None,
    findings: dict,
    history: list | None = None,
    style_directives: list[str] | None = None,
    profile=None,
) -> tuple[str, str]:

        case_file = self._case_file_text(
                    findings=findings,
                    query=query,
                )

    #     structured_evidence = self._structured_evidence_package(
    #     query=query,
    #     findings=findings,
    # )

    #     case_file += "\n\n"
    #     case_file += structured_evidence

        # ============================================================
        # ANSWER PLAN
        # ============================================================

        plan = findings.get("answer_plan")

        if plan:
            case_file += "\n\n"
            case_file += "=" * 70 + "\n"
            case_file += "ANSWER PLANNING INSTRUCTIONS\n"
            case_file += "=" * 70 + "\n"

            case_file += (
                "Generate the answer using the sections below in order.\n"
                "Use the organized evidence package as the primary source.\n"
                "Do not invent facts.\n"
                "Every important factual claim must be supported by the provided evidence.\n"
                "If evidence is insufficient for a claim, explicitly say so instead of guessing.\n\n"
            )

            case_file += "Sections:\n"

            for section in plan.get("structure", []):
                case_file += f"- {section}\n"

            must_include = plan.get("must_include", [])

            if must_include:
                case_file += "\nMust Include:\n"

                for item in must_include[:20]:
                    case_file += f"- {item}\n"

            avoid = plan.get("avoid", [])

            if avoid:
                case_file += "\nAvoid:\n"

                for item in avoid:
                    case_file += f"- {item}\n"

        # ============================================================
        # EVIDENCE SUFFICIENCY
        # ============================================================

        sufficiency = findings.get(
            "evidence_sufficiency",
            {},
        )

        evidence_sufficient = sufficiency.get(
            "sufficient",
            False,
        )

        sufficiency_score = sufficiency.get(
            "score",
            0.0,
        )

        # ============================================================
        # ANSWER POLICY
        # ============================================================

        case_file += "\n\n"
        case_file += "=" * 70 + "\n"
        case_file += "ANSWER POLICY\n"
        case_file += "=" * 70 + "\n"

        case_file += (
            f"Evidence Sufficiency Score: {sufficiency_score:.2f}\n\n"
        )

        if evidence_sufficient:

            case_file += (
                    "The retrieved evidence appears broadly sufficient, but this does NOT mean "
                    "that every entity or comparison target is supported.\n"
                    "For comparison questions, each side of the comparison must be independently "
                    "supported by explicit evidence.\n"
                    "Never infer capabilities from a product name, context, architecture name, "
                    "or surrounding discussion.\n"
                    "Do not use words such as 'likely', 'implied', 'suggests', or 'appears' to "
                    "introduce unsupported factual claims.\n"
                    "If one side of a comparison lacks evidence, explicitly state that a grounded "
                    "comparison cannot be completed from the retrieved evidence.\n"
                )

        else:

            case_file += (
                "The retrieved evidence may be incomplete.\n\n"
                "First answer everything that can be supported by the retrieved evidence.\n"
                "Do not guess or fill evidence gaps inside the evidence-based sections.\n\n"
                "If additional background knowledge would materially help answer a missing "
                "part of the user's question, you MAY include a separate section titled:\n\n"
                "## General Knowledge (Not from Retrieved Documents)\n\n"
                "Use that section only when necessary.\n"
                "Clearly state that the information comes from model knowledge rather than "
                "the retrieved documents.\n"
                "Never cite retrieved documents as support for general knowledge.\n"
                "Never mix general knowledge into the evidence-based answer.\n"
            )

        # ============================================================
        # LLM SYNTHESIS
        # ============================================================

        answer = llm.synthesize(
            query,
            case_file,
            history,
            question_type=getattr(
                findings.get("query_plan"), "intent", None,
            ),
            style_directives=style_directives,
            profile=profile,
        )

        if answer:
            return answer, (llm.last_generation().get("provider")
                            or llm.ACTIVE_PROVIDER)

        # ============================================================
        # DETERMINISTIC FALLBACK
        # ============================================================

        return (
            self._template_answer(
                query,
                anchor,
                findings,
                profile,
            ),
            "deterministic (no LLM credentials)",
        )
    @staticmethod
    def _extractive_answer(query: str, evidence: list[dict],
                           profile=None) -> str:
        """Readable extractive summary when no LLM is available: pick clean prose
        sentences from the evidence (filtering out code, imports, headings and
        fragments), preferring ones that mention the query's terms.

        Format-aware: when the Adaptive Response Engine detected a requested
        output format (table / checklist / bullets), the extracted points are
        rendered in that shape, so a "in tabular format" request is honoured
        even with no LLM available."""
        stop = {
            "about", "explain", "detail", "details", "what", "which", "that",
            "this", "with", "from", "your", "need", "does", "into", "overview",
        }
        qterms = {
            t.lower()
            for t in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", query)
            if t.lower() not in stop
        }
        useful_terms = {
            "propose", "introduce", "architecture", "transformer", "attention",
            "self-attention", "multi-head", "encoder", "decoder", "parallel",
            "train", "training", "bitnet", "1-bit", "1.58", "ternary",
            "weights", "memory", "latency", "perplexity", "energy",
            "throughput", "quantization",
        }

        def is_code(s: str) -> bool:
            if re.search(r"\b(import|def|return|class|lambda|self|page_content)\b", s):
                return True
            return sum(s.count(c) for c in "{}()[]=<>|/\\") >= 4 or "->" in s or "::" in s

        def clean(s: str) -> str:
            return " ".join(s.split()).strip()

        def is_boilerplate(s: str, section: str) -> bool:
            low = s.lower()
            if "references" in section.lower():
                return True
            bad_bits = [
                "provided proper attribution", "google hereby grants permission",
                "@", "equal contribution", "work performed while",
                "conference on neural information processing systems",
                "we are excited about the future",
            ]
            if any(b in low for b in bad_bits):
                return True
            if re.match(r"^\[[A-Z0-9+]+\]", s):
                return True
            return False

        sents = []
        for evidence_rank, e in enumerate(evidence):
            section = e.get("section", "")
            for raw in re.split(r"(?<=[.!?])\s+|\n{2,}|\n(?=[A-Z][a-z])", e.get("text", "")):
                s = clean(raw)
                words = s.split()
                if not (6 <= len(words) <= 45) or len(s) > 320:
                    continue
                if is_code(s) or is_boilerplate(s, section) or not re.search(r"[a-z]{4}", s):
                    continue
                if sum(1 for w in words if any(c.isalpha() for c in w)) < len(words) * 0.6:
                    continue                                  # mostly symbols/numbers
                low = s.lower()
                score = sum(2 for t in qterms if t in low)
                score += sum(1 for t in useful_terms if t in low)
                if re.search(r"\b(propose|introduce|achieves?|shows?|allows?|uses?|matches?|reduces?)\b", low):
                    score += 2
                score += max(0, 4 - evidence_rank) * 0.25
                sents.append((score, len(sents), s, e))

        sents.sort(key=lambda x: (-x[0], x[1]))
        picked, seen = [], set()
        for _sc, _i, s, e in sents:
            key = s.lower()[:50]
            if key in seen:
                continue
            seen.add(key)
            picked.append((s, e))
            if len(picked) >= 6:
                break
        if not picked:                                        # nothing clean — trim first chunk
            picked = [(clean(evidence[0].get("text", ""))[:260], evidence[0])]

        title = evidence[0].get("doc_title") or evidence[0].get("doc_no", "the retrieved document")
        formats = set(getattr(profile, "formats", []) or [])

        def _cell(text: str) -> str:                    # keep Markdown table cells intact
            return text.replace("|", "\\|").replace("\n", " ").strip()

        parts = ["## Direct Answer",
                 f"From **{title}**: {picked[0][0]}  _({picked[0][1]['doc_no']}, {picked[0][1]['section']})_"]

        if len(picked) > 1 and ({"table", "comparison"} & formats):
            # Requested a table → render the extracted points as a Markdown table.
            parts.append("## Key Points")
            parts.append("| # | Key point | Source |")
            parts.append("|---|-----------|--------|")
            for i, (s, e) in enumerate(picked[1:6], start=1):
                parts.append(f"| {i} | {_cell(s)} | {e['doc_no']} ({_cell(e['section'])}) |")
        elif len(picked) > 1 and "checklist" in formats:
            parts.append("## Key Points")
            for s, e in picked[1:6]:
                parts.append(f"- [ ] {s}  _({e['doc_no']}, {e['section']})_")
        elif len(picked) > 1:
            parts.append("## Key Points")
            for s, e in picked[1:5]:
                parts.append(f"- {s}  _({e['doc_no']}, {e['section']})_")

        parts.append("## Sources")
        seen_src = set()
        for e in evidence[:6]:
            k = (e["doc_no"], e["section"])
            if k in seen_src:
                continue
            seen_src.add(k)
            parts.append(f"- **{e['doc_no']}** ({e['section']})")
        note = ("\n_The AI answer engine is currently unavailable, so this is an "
                "extractive summary of the retrieved evidence rather than a composed answer._")
        if {"table", "comparison"} & formats:
            note += ("\n_Note: without a language model the table lists the extracted "
                     "evidence points; a composed comparison table needs a live LLM "
                     "(add HuggingFace credits or set ANTHROPIC_API_KEY)._")
        parts.append(note)
        return "\n".join(parts)

    @staticmethod
    def _template_answer(query: str, anchor: str | None, findings: dict,
                         profile=None) -> str:
        if not anchor or "predictive" not in findings:
            ev = findings.get("evidence", [])
            if not ev:
                return ("I don't have grounded evidence to answer that, so I won't guess. "
                        "Per the grounded-or-refuse policy this is escalated to a human expert — "
                        "and logged as a **knowledge gap**: if this question matters to operations, "
                        "the procedure should be captured before the expertise walks out the door.")
            return AgentSystem._extractive_answer(query, ev, profile)

        p = findings["predictive"]
        rc = findings["rootcause"]["candidate_cause"]
        m = findings["maintenance"]
        s = findings["safety"]
        r = findings["risk"]
        ev_docs = {e["doc_no"]: i + 1 for i, e in enumerate(findings["evidence"])}

        def cite(doc):
            return f" [{ev_docs[doc]}]" if doc in ev_docs else ""

        return f"""**What is happening.** {anchor} drive-end vibration is {p['latest_vibration']} mm/s and rising at ~{p['trend_mm_s_per_day']} mm/s/day — inside the alert zone ({p['alert_limit']}–{p['danger_limit']} mm/s per SOP-315{cite('SOP-315')}). At the current trend it crosses the {p['danger_limit']} mm/s danger limit in about **{p['rul_days']} days** (RUL). Bearing temperature is normal ({p['bearing_temp_recent_c']} °C), which points at mechanical wear, not lubrication.

**Why.** {rc['likely_cause'].capitalize()} driving {rc['failure_mode']}: the P&ID shows {anchor} coupled to motor M-101, and {findings['rootcause']['similar_count']} past work orders on {anchor} show the same rising-1x-vibration signature, each resolved by BRG-204 replacement plus laser alignment (vendor manual failure mode 1{cite('VM-P101')}).

**Safety — hard gate.** {s['requirements'][0]['permit'] if s['requirements'] else 'A permit'} is REQUIRED before any work: isolate per SAF-12 §3 (breaker MCC-1-04 locked open, valves V-101/V-102 locked closed, zero-energy verified){cite('SAF-12')}. No work proceeds without the permit.

**What to do.** {r['recommendation']} Torque housing covers to 45 N·m and verify post-run vibration < 2.8 mm/s per SOP-207{cite('SOP-207')}. Spares: 2 × BRG-204 in stock (Store A / Bin 14). A draft work order is prepared ({m['draft_work_order']['title']}) — pending your approval."""


    

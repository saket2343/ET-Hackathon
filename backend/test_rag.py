"""Tests for the modular RAG package (backend/rag).

Runs standalone (`python test_rag.py`) and under pytest. Everything is
offline and LLM-free: generators/retrievers are stubbed where needed, and
the real corpus is used for vocabulary/concept-dependent tests.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import (build_understanding, corpus_vocabulary, load_config)
from rag.classify import RuleQueryClassifier
from rag.interfaces import Classification, RetrievalPlan
from rag.orchestrate import IterativeController
from rag.plan import (ConceptAmbiguityDetector, TableRetrievalPlanner,
                      TemplateEvidencePlanner)
from rag.policy import BoundedRetryPolicy, ThresholdConfidencePolicy
from rag.spell import CorpusSpellCorrector

CFG = load_config()


# ------------------------------------------------------------ spell

def _vocab():
    # Deterministic synthetic vocabulary with realistic frequencies.
    return Counter({
        "transformer": 220, "attention": 340, "retrieval": 400,
        "langchain": 800, "embedding": 260, "mechanism": 120,
        "the": 5000, "difference": 150, "between": 400, "langgraph": 300,
        "pump": 90, "bearing": 70, "vibration": 60, "state": 240,
        # high-frequency decoy: "lagraph" must resolve to langgraph, not
        # to the generic "graph" that ties on edit distance
        "graph": 900,
    })


def test_spell_examples():
    sc = CorpusSpellCorrector(_vocab(), CFG["spell"])
    cases = {"transformar": "transformer", "attension": "attention",
             "retrival": "retrieval", "langchan": "langchain",
             "embeding": "embedding", "lagraph": "langgraph",
             "differece": "difference"}
    for typo, want in cases.items():
        res = sc.correct(typo)
        assert res.corrected == want, f"{typo}: {res.corrected} != {want}"
        assert res.corrections and res.corrections[0].confidence >= 0.6


def test_spell_multiword_and_logging():
    sc = CorpusSpellCorrector(_vocab(), CFG["spell"])
    res = sc.correct("attension mechanisam")
    assert res.corrected == "attention mechanism"
    assert len(res.corrections) == 2 and res.confidence >= 0.6
    pairs = {(c.original, c.corrected) for c in res.corrections}
    assert ("attension", "attention") in pairs
    assert ("mechanisam", "mechanism") in pairs


def test_spell_preserves_protected():
    sc = CorpusSpellCorrector(_vocab(), CFG["spell"])
    text = ('check https://docs.langchan.dev and app.py, run '
            'get_embeding() on "attension text", tag P-101, use RAG')
    res = sc.correct(text)
    # URL, filename, snake_case call, quoted string, tag, acronym untouched:
    assert "https://docs.langchan.dev" in res.corrected
    assert "app.py" in res.corrected
    assert "get_embeding()" in res.corrected
    assert '"attension text"' in res.corrected
    assert "P-101" in res.corrected and "RAG" in res.corrected


def test_spell_never_touches_valid_or_short_words():
    sc = CorpusSpellCorrector(_vocab(), CFG["spell"])
    res = sc.correct("the transformer attention")
    assert res.corrected == "the transformer attention"
    assert not res.corrections
    # Short tokens are exempt from EDIT-DISTANCE correction, but the user
    # dictionary (checked before the length guard) may fix known function-
    # word typos — 'teh' -> 'the', 'si' -> 'is' ('what si transformer'
    # previously broke the "what is" intent pattern).
    assert sc.correct("teh").corrected == "the"        # via user dictionary
    assert sc.correct("what si transformer").corrected \
        == "what is transformer"
    assert sc.correct("ax").corrected == "ax"          # short + not in dict


def test_spell_user_dictionary_wins():
    cfg = dict(CFG["spell"], user_dictionary={"attension": "Attention-Is-All"})
    sc = CorpusSpellCorrector(_vocab(), cfg)
    assert sc.correct("attension").corrected == "Attention-Is-All"


# --------------------------------------------------------- classify

def test_classifier_labels():
    clf = RuleQueryClassifier(CFG["classifier"])
    assert clf.classify("What is the difference between X and Y?").label \
        == "comparison"
    assert clf.classify("Why was the Transformer invented?").label \
        == "reasoning"
    assert clf.classify("What is LCEL?").label == "definition"
    assert clf.classify("How do I replace the bearing?").label == "procedure"
    assert clf.classify("Fix this TypeError exception").label == "debugging"
    assert clf.classify("random words entirely").label \
        == CFG["classifier"]["default_label"]


# --------------------------------------------------------- planning

def test_retrieval_planner_table():
    rp = TableRetrievalPlanner(CFG["retrieval_plans"])
    plan = rp.plan(Classification("comparison", 0.9), "q")
    assert plan.decompose and "graph" in plan.legs and plan.k == 8
    default = rp.plan(Classification("unknown-label", 0.4), "q")
    assert default.legs == ["dense", "bm25"]


def test_evidence_planner_facets_and_verify():
    ep = TemplateEvidencePlanner(CFG["evidence_facets"])
    facets = ep.plan("why was the transformer invented",
                     Classification("reasoning", 0.9), ["transformer"])
    names = [f.name for f in facets]
    assert "problem or need" in names and "advantages" in names
    evidence = ["RNNs had a limitation: sequential computation was the "
                "problem. The transformer was proposed as a solution and "
                "improved speed — a major advantage."]
    cov = ep.verify(facets, evidence)
    assert cov == 1.0 and all(f.covered for f in facets)
    cov2 = ep.verify(facets, ["totally unrelated text"])
    assert cov2 == 0.0


def test_ambiguity_clarifies_short_query():
    concepts = ["self-attention", "cross-attention", "multi-head attention",
                "Bahdanau attention", "Luong attention", "LangGraph"]
    det = ConceptAmbiguityDetector(concepts, CFG["ambiguity"])
    clar = det.detect("Explain attention")
    assert clar and clar.term == "attention" and len(clar.options) >= 3
    assert "Which attention?" in clar.question
    # specific query -> no clarification
    assert det.detect("Explain multi-head attention in transformers") is None
    # long query -> no clarification even with the ambiguous term
    assert det.detect(
        "Explain attention as used by the original transformer paper "
        "for machine translation") is None


# ----------------------------------------------------------- policy

def test_confidence_policy_bands():
    pol = ThresholdConfidencePolicy(CFG["confidence_policy"],
                                    max_iterations=2)
    assert pol.decide(0.95, 0).action == "ANSWER"
    noted = pol.decide(0.80, 0)
    assert noted.action == "ANSWER_WITH_NOTE" and noted.note
    assert pol.decide(0.50, 0).action == "RETRY"
    final = pol.decide(0.50, 2)
    assert final.action == "INSUFFICIENT"
    assert "insufficient" in final.note.lower()


# ---------------------------------------------- iterative controller

class _StubUnderstander:
    def __init__(self, qu):
        self.qu = qu

    def process(self, raw):
        return self.qu


def _stub_qu():
    from rag.interfaces import QueryUnderstanding
    ep = TemplateEvidencePlanner(CFG["evidence_facets"])
    facets = ep.plan("q", Classification("default", 0.5), [])
    return QueryUnderstanding(
        original="q", normalized="q", corrected="q",
        plan=RetrievalPlan(legs=["dense"], k=4), facets=facets)


class _StubRetriever:
    """Failure-injection retriever: first call returns weak evidence,
    claim-derived retries return the strong chunk."""

    def __init__(self):
        self.calls = []

    def retrieve(self, query, plan):
        self.calls.append(query)
        if "strong" in query:
            return [{"chunk_id": "c2", "text": "strong supporting evidence "
                     "for the claim", "doc_no": "D2", "section": "s"}]
        return [{"chunk_id": "c1", "text": "weak", "doc_no": "D1",
                 "section": "s"}]


class _StubVerifier:
    """Coverage rises once the strong chunk is present."""

    def verify(self, answer, evidence):
        strong = any("strong" in e["text"] for e in evidence)
        return {"answer": answer, "coverage": 0.95 if strong else 0.4,
                "insufficient_evidence_claims": 0 if strong else 1,
                "claims": [] if strong else [{
                    "status": "INSUFFICIENT_EVIDENCE",
                    "claim": "needs strong evidence [1]"}]}


def test_iterative_loop_recovers():
    ctl = IterativeController(
        understander=_StubUnderstander(_stub_qu()),
        retriever=_StubRetriever(),
        generator=lambda q, cf: "answer text",
        verifier=_StubVerifier(),
        policy=ThresholdConfidencePolicy(CFG["confidence_policy"], 2),
        retry=BoundedRetryPolicy(CFG["retry"]),
        evidence_planner=TemplateEvidencePlanner({"default": []}),
    )
    result = ctl.run("q")
    assert result["status"] in ("ANSWER", "ANSWER_WITH_NOTE")
    assert result["iterations"] >= 1 and result["coverage"] > 0.9
    stages = [t["stage"] for t in result["trace"]]
    assert "iterative_retrieval" in stages


def test_iterative_loop_gives_up_honestly():
    class NeverBetter:
        def verify(self, answer, evidence):
            return {"answer": answer, "coverage": 0.2,
                    "insufficient_evidence_claims": 1, "claims": []}

    class SameChunk:
        def retrieve(self, query, plan):
            return [{"chunk_id": "c1", "text": "weak", "doc_no": "D",
                     "section": "s"}]

    ctl = IterativeController(
        understander=_StubUnderstander(_stub_qu()),
        retriever=SameChunk(),
        generator=lambda q, cf: "answer",
        verifier=NeverBetter(),
        policy=ThresholdConfidencePolicy(CFG["confidence_policy"], 2),
        retry=BoundedRetryPolicy(CFG["retry"]),
        evidence_planner=TemplateEvidencePlanner({"default": []}),
    )
    result = ctl.run("q")
    assert result["status"] == "INSUFFICIENT"
    assert "insufficient" in result["answer"].lower()


def test_clarification_short_circuits():
    qu = _stub_qu()
    from rag.interfaces import Clarification
    qu.clarification = Clarification(
        term="attention",
        options=["self-attention", "cross-attention", "multi-head attention"],
        question="Which attention? self-attention • cross-attention • "
                 "multi-head attention")
    ctl = IterativeController(
        understander=_StubUnderstander(qu), retriever=_StubRetriever(),
        generator=lambda q, cf: "x", verifier=_StubVerifier(),
        policy=ThresholdConfidencePolicy(CFG["confidence_policy"], 2),
        retry=BoundedRetryPolicy(CFG["retry"]),
        evidence_planner=TemplateEvidencePlanner({"default": []}),
    )
    result = ctl.run("Explain attention")
    assert result["status"] == "CLARIFY" and len(result["options"]) == 3


# ----------------------------------------------------- NLI validation

def test_nli_validation_catches_hallucinations_not_truths():
    """Locks in the NLI second stage's contract after two same-day
    regressions: (1) max-contradiction across windows must not veto an
    entailed claim (PDF line-wrap fragments scored 0.98 contradiction on
    verbatim-true text); (2) title/heading windows must not serve as
    premises (a book cover 'A Complete Practitioner's Guide...' entailed
    'LangChain is a comprehensive knowledge base' at 0.996)."""
    from validator import AnswerValidator
    from retrieval import get_nli_verifier, get_validation_reranker
    v = AnswerValidator(semantic_model=get_validation_reranker(),
                        nli_model=get_nli_verifier())
    if v.nli_model is None:
        return  # NLI disabled in this environment — nothing to lock in
    ev = [
        {"evidence_id": 1, "doc_no": "D1", "section": "s", "keywords": [],
         "concepts": [], "entities": [], "summary": "",
         # hard line-wraps mid-sentence, exactly like extracted PDF text
         "text": ("LangGraph is a library for building stateful, multi-step "
                  "AI workflows\nas directed graphs. It handles cycles, "
                  "branching, parallelism,\nhuman-in-the-loop, and "
                  "persistence. With a single simple chain,\nlooping is not "
                  "possible.")},
        {"evidence_id": 2, "doc_no": "D2", "section": "s", "keywords": [],
         "concepts": [], "entities": [], "summary": "",
         # title-page style content: capitalized noun phrases, no assertions
         "text": ("LangChain for AI Engineers A Complete Practitioner's "
                  "Guide To LangChain, LangGraph & Production LLM Systems "
                  "From Prompt Templates To Production Deployment")},
    ]
    true_claim = ("LangGraph is a library for building stateful multi-step "
                  "AI workflows as directed graphs.")
    r = v._assess_against_evidence(true_claim, ev)
    assert r["status"] == "SUPPORTED", (r["score"], r["nli_entailment"],
                                        r["nli_contradiction"])
    swap = ("LangChain is a library for building stateful multi-step AI "
            "workflows as directed graphs.")
    r2 = v._assess_against_evidence(swap, ev)
    assert r2["status"] == "INSUFFICIENT_EVIDENCE", r2["score"]
    wrong = "LangChain is a comprehensive knowledge base for AI engineers."
    r3 = v._assess_against_evidence(wrong, ev)
    assert r3["status"] != "SUPPORTED", (r3["score"], r3["nli_entailment"])
    # (3) conjunctive claims decompose into per-item hypotheses: a correct
    # list-summary spanning several sentences must be SUPPORTED...
    conj = ("LangGraph handles cycles, branching, parallelism, and "
            "human-in-the-loop workflows.")
    r4 = v._assess_against_evidence(conj, ev)
    assert r4["status"] == "SUPPORTED", (r4["score"], r4["nli_entailment"])
    # ...while an enumeration on the WRONG SUBJECT stays caught (premise
    # fusion was measured to entail this at 0.87 — decomposition keeps the
    # claim's own subject in every hypothesis).
    conj_swap = ("LangChain handles cycles, branching, parallelism, and "
                 "human-in-the-loop workflows.")
    r5 = v._assess_against_evidence(conj_swap, ev)
    assert r5["status"] == "INSUFFICIENT_EVIDENCE", (r5["score"],
                                                     r5["nli_entailment"])


# --------------------------------------------- end-to-end (real corpus)

def test_understanding_on_real_corpus():
    from ingest import load_corpus
    corpus = load_corpus()
    pipeline = build_understanding(corpus)
    vocab = corpus_vocabulary(corpus.chunks)

    qu = pipeline.process(
        "what is the diference between langchan and langgraph?")
    fixed = {c.original.lower(): c.corrected.lower()
             for c in qu.spell.corrections}
    if "difference" in vocab:
        assert fixed.get("diference") == "difference"
    if "langchain" in vocab:
        assert fixed.get("langchan") == "langchain"
    assert qu.classification.label == "comparison"
    assert qu.plan.decompose and len(qu.subqueries) >= 2
    assert any(f.name == "differences" for f in qu.facets)
    assert [t["stage"] for t in qu.trace][0] == "normalize"


def _main():
    failures = 0
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())

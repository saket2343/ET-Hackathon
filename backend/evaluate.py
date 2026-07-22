#!/usr/bin/env python
"""Evaluation harness for the AXON RAG pipeline — two layers.

LAYER 1 — RETRIEVAL (default; deterministic, LLM-free, CI-gated)
    Runs the golden set (data/golden_qa.jsonl) through query understanding +
    hybrid retrieval and reports, per case and per category:

        hit@k        expected document(s) present in the top-k evidence
        recall@1/@5  first relevant chunk within rank 1 / rank 5
        MRR          1 / rank of the first relevant chunk
        precision@k  fraction of top-k chunks from the expected document(s)
        nDCG@k       rank quality of relevant chunks (binary, doc-level
                     labels — self-normalised, see _ndcg)
        term cov.    expected facts present in the retrieved evidence text
        intent acc.  detected intent matches the labelled intent
        latency      per-query retrieve() wall time

    Results go to data/eval_results.json; a previous results file is the
    baseline, and any gated metric dropping more than --tolerance FAILS the
    run (non-zero exit). --update-baseline accepts intentional changes.

LAYER 2 — ANSWERS (--answers; runs the FULL pipeline incl. LLM per case)
    Executes run_case() end-to-end and scores the final answer using the
    validator's own report — the metrics the retrieval layer cannot see:

        coverage             validator answer coverage (0..1)
        claim support rate   supported / total claims
        hallucination rate   insufficient-evidence claims / total claims
        citation alignment   citations actually support their claims
        answer terms         expected_answer_terms present in the answer
        refusal accuracy     negative cases (expect_refusal) refused;
                             positive cases answered

    Costs real LLM calls (~10-20s/case) — run nightly or with --limit N,
    not per-commit. Answer metrics are recorded but NOT regression-gated:
    LLM output variance would make the gate flaky; watch trends instead.

Golden case schema (one JSON object per line):
    {"id": "...", "query": "...",
     "expected_doc": "substring" | "expected_docs": ["a", "b"],
     "expected_terms": [...], "intent": "...", "category": "...",
     "expected_answer_terms": [...], "expect_refusal": true}
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

# Metrics compared against the baseline (drop > tolerance -> exit 1).
GATED_METRICS = ("hit_rate", "mrr", "term_coverage", "intent_accuracy",
                 "precision_at_k", "recall_at_5", "ndcg")


def _term_present(term: str, text: str) -> bool:
    return re.search(
        r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])",
        text.lower(),
    ) is not None


def _expected_docs(case: dict) -> list[str]:
    if case.get("expected_docs"):
        return [d.lower() for d in case["expected_docs"]]
    return [case["expected_doc"].lower()] if case.get("expected_doc") else []


def _is_relevant(hit: dict, docs: list[str]) -> bool:
    label = f"{hit['doc_no']} {hit['doc_title']}".lower()
    return any(d in label for d in docs)


def _ndcg(rels: list[int]) -> float:
    """Binary, doc-level nDCG over the retrieved list. Self-normalised:
    IDCG uses the same relevant count, so 1.0 means every relevant chunk
    ranked above every irrelevant one. Doc-level labels (any chunk of the
    expected document counts) — coarser than chunk-level judgments, which
    this corpus does not have."""
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


# ------------------------------------------------------------ layer 1

def run_retrieval(cases: list[dict], k: int, no_rerank: bool) -> dict:
    if no_rerank:
        import retrieval
        retrieval.CrossEncoder = None

    from ingest import load_corpus
    from retrieval import HybridIndex
    from query_processor import QueryProcessor

    corpus = load_corpus()
    index = HybridIndex(corpus.chunks)
    qp = QueryProcessor()

    rows = []
    for case in cases:
        docs = _expected_docs(case)
        if not docs:
            continue                      # negative cases: answers layer only
        plan = qp.process(case["query"])
        t0 = time.time()
        hits = index.retrieve(plan.retrieval_queries[0], query_plan=plan, k=k)
        latency_ms = round((time.time() - t0) * 1000, 1)

        rels = [1 if _is_relevant(h, docs) else 0 for h in hits]
        rank = next((i + 1 for i, r in enumerate(rels) if r), None)
        # multi-doc: EVERY expected document must appear in the top-k
        all_docs_hit = all(
            any(_is_relevant(h, [d]) for h in hits) for d in docs)

        blob = " ".join(h["text"] for h in hits)
        terms = case.get("expected_terms", [])
        covered = [t for t in terms if _term_present(t, blob)]

        wanted_intent = case.get("intent")
        rows.append({
            "id": case["id"],
            "category": case.get("category", "core"),
            "query": case["query"],
            "hit": all_docs_hit,
            "rank": rank,
            "rr": (1.0 / rank) if rank else 0.0,
            "recall_at_1": bool(rels[:1] and rels[0]),
            "recall_at_5": any(rels[:5]),
            "precision_at_k": round(mean(rels), 3) if rels else 0.0,
            "ndcg": round(_ndcg(rels), 3),
            "term_coverage": len(covered) / len(terms) if terms else 1.0,
            "missing_terms": [t for t in terms if t not in covered],
            "intent_ok": (plan.intent == wanted_intent) if wanted_intent else None,
            "detected_intent": plan.intent,
            "latency_ms": latency_ms,
        })

    intent_rows = [r for r in rows if r["intent_ok"] is not None]
    summary = {
        "k": k, "reranker": not no_rerank, "cases": len(rows),
        "hit_rate": round(mean(r["hit"] for r in rows), 3),
        "mrr": round(mean(r["rr"] for r in rows), 3),
        "recall_at_1": round(mean(r["recall_at_1"] for r in rows), 3),
        "recall_at_5": round(mean(r["recall_at_5"] for r in rows), 3),
        "precision_at_k": round(mean(r["precision_at_k"] for r in rows), 3),
        "ndcg": round(mean(r["ndcg"] for r in rows), 3),
        "term_coverage": round(mean(r["term_coverage"] for r in rows), 3),
        "intent_accuracy": round(mean(r["intent_ok"] for r in intent_rows), 3)
        if intent_rows else None,
        "latency_ms_mean": round(mean(r["latency_ms"] for r in rows), 1),
        "latency_ms_max": max(r["latency_ms"] for r in rows),
    }
    return {"summary": summary, "rows": rows}


# ------------------------------------------------------------ layer 2

def run_answers(cases: list[dict], limit: int | None) -> dict:
    """Full-pipeline answer evaluation: retrieval -> generation ->
    validation -> self-correction, scored from the validator's report."""
    from ingest import load_corpus
    from kg import build_graph
    from retrieval import HybridIndex
    from agents import AgentSystem

    corpus = load_corpus()
    system = AgentSystem(corpus, build_graph(corpus), HybridIndex(corpus.chunks))

    if limit:
        # keep the mix: negatives first (cheap, decisive), then positives
        negatives = [c for c in cases if c.get("expect_refusal")]
        positives = [c for c in cases if not c.get("expect_refusal")]
        cases = (negatives + positives)[:limit]

    rows = []
    for case in cases:
        t0 = time.time()
        try:
            result = system.run_case(case["query"], history=[])
        except Exception as exc:
            rows.append({"id": case["id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        latency_s = round(time.time() - t0, 1)

        answer = result.get("answer") or ""
        verdict = str(result.get("verdict", ""))
        refused = "REFUSE" in verdict.upper() or "ESCALATE" in verdict.upper()
        validation = (result.get("findings") or {}).get("validation") or {}
        total_claims = len(validation.get("claims", []))
        insufficient = validation.get("insufficient_evidence_claims", 0)
        supported = validation.get("supported_claims", 0)

        expect_refusal = bool(case.get("expect_refusal"))
        refusal_ok = refused if expect_refusal else not refused

        ans_terms = case.get("expected_answer_terms", [])
        ans_covered = [t for t in ans_terms if _term_present(t, answer)]

        rows.append({
            "id": case["id"],
            "category": case.get("category", "core"),
            "query": case["query"],
            "verdict": verdict[:40],
            "refusal_expected": expect_refusal,
            "refusal_ok": refusal_ok,
            "coverage": validation.get("coverage"),
            "claim_support_rate": round(supported / total_claims, 3)
            if total_claims else None,
            "hallucination_rate": round(insufficient / total_claims, 3)
            if total_claims else None,
            "citation_alignment": validation.get("citation_alignment_score"),
            "answer_term_coverage": len(ans_covered) / len(ans_terms)
            if ans_terms else None,
            "missing_answer_terms": [t for t in ans_terms
                                     if t not in ans_covered],
            "latency_s": latency_s,
        })

    ok_rows = [r for r in rows if "error" not in r]

    def _mean_of(key):
        vals = [r[key] for r in ok_rows if r.get(key) is not None]
        return round(mean(vals), 3) if vals else None

    summary = {
        "cases": len(rows),
        "errors": len(rows) - len(ok_rows),
        "coverage_mean": _mean_of("coverage"),
        "claim_support_rate": _mean_of("claim_support_rate"),
        "hallucination_rate": _mean_of("hallucination_rate"),
        "citation_alignment": _mean_of("citation_alignment"),
        "answer_term_coverage": _mean_of("answer_term_coverage"),
        "refusal_accuracy": _mean_of("refusal_ok"),
        "latency_s_mean": _mean_of("latency_s"),
    }
    return {"summary": summary, "rows": rows}


# ------------------------------------------------------------ reporting

def _print_retrieval(result: dict) -> None:
    rows, s = result["rows"], result["summary"]
    print(f"\n{'id':<5} {'hit':<4} {'rank':<5} {'P@k':<5} {'nDCG':<5} "
          f"{'terms':<6} {'intent':<12} {'ms':<7} query")
    for r in rows:
        print(f"{r['id']:<5} {str(r['hit']):<4} {str(r['rank']):<5} "
              f"{r['precision_at_k']:<5.2f} {r['ndcg']:<5.2f} "
              f"{r['term_coverage']:<6.2f} {r['detected_intent']:<12} "
              f"{r['latency_ms']:<7} {r['query'][:44]}")
        if r["missing_terms"]:
            print(f"      missing terms: {r['missing_terms']}")

    print("\nBY CATEGORY (hit rate | MRR | n):")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat, rs in sorted(by_cat.items()):
        print(f"  {cat:<12} {mean(x['hit'] for x in rs):.2f} | "
              f"{mean(x['rr'] for x in rs):.2f} | {len(rs)}")

    print(f"\nRETRIEVAL SUMMARY  hit@{s['k']}={s['hit_rate']}  "
          f"MRR={s['mrr']}  R@1={s['recall_at_1']}  R@5={s['recall_at_5']}  "
          f"P@{s['k']}={s['precision_at_k']}  nDCG={s['ndcg']}  "
          f"terms={s['term_coverage']}  intent={s['intent_accuracy']}  "
          f"latency mean={s['latency_ms_mean']}ms max={s['latency_ms_max']}ms")


def _print_answers(result: dict) -> None:
    rows, s = result["rows"], result["summary"]
    print(f"\n{'id':<5} {'refusal_ok':<11} {'cover':<6} {'halluc':<7} "
          f"{'cite':<5} {'s':<6} verdict / query")
    for r in rows:
        if "error" in r:
            print(f"{r['id']:<5} ERROR: {r['error']}")
            continue
        print(f"{r['id']:<5} {str(r['refusal_ok']):<11} "
              f"{str(r['coverage']):<6} {str(r['hallucination_rate']):<7} "
              f"{str(r['citation_alignment']):<5} {r['latency_s']:<6} "
              f"{r['verdict'][:26]} | {r['query'][:38]}")
        if r.get("missing_answer_terms"):
            print(f"      missing answer terms: {r['missing_answer_terms']}")

    print(f"\nANSWER SUMMARY  coverage={s['coverage_mean']}  "
          f"claim_support={s['claim_support_rate']}  "
          f"hallucination_rate={s['hallucination_rate']}  "
          f"citation_alignment={s['citation_alignment']}  "
          f"answer_terms={s['answer_term_coverage']}  "
          f"refusal_accuracy={s['refusal_accuracy']}  "
          f"latency mean={s['latency_s_mean']}s")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--golden", type=Path,
                    default=BACKEND.parent / "data" / "golden_qa.jsonl")
    ap.add_argument("--results", type=Path,
                    default=BACKEND.parent / "data" / "eval_results.json")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--no-rerank", action="store_true",
                    help="skip the cross-encoder (fast lexical-only run)")
    ap.add_argument("--answers", action="store_true",
                    help="ALSO run full-pipeline answer evaluation (LLM cost)")
    ap.add_argument("--answers-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="answer mode: evaluate at most N cases")
    args = ap.parse_args()

    cases = [json.loads(line) for line in args.golden.read_text().splitlines()
             if line.strip()]

    result: dict = {}
    exit_code = 0

    if not args.answers_only:
        result["retrieval"] = run_retrieval(cases, args.k, args.no_rerank)
        _print_retrieval(result["retrieval"])

        if args.results.exists() and not args.update_baseline:
            old = json.loads(args.results.read_text())
            baseline = (old.get("retrieval") or old).get("summary", {})
            regressions = []
            print()
            for m in GATED_METRICS:
                prev, now = baseline.get(m), result["retrieval"]["summary"].get(m)
                if prev is None or now is None:
                    continue
                delta = round(now - prev, 3)
                marker = ""
                if delta < -args.tolerance:
                    marker = "  << REGRESSION"
                    regressions.append(m)
                print(f"  {m:<16} baseline={prev}  now={now}  Δ={delta:+}{marker}")
            if regressions:
                print(f"\nFAIL: regression in {regressions} "
                      f"(tolerance {args.tolerance}). Baseline NOT updated; "
                      f"pass --update-baseline to accept intentionally.")
                exit_code = 1
            else:
                print("\nOK: no regression vs baseline.")

    if args.answers or args.answers_only:
        result["answers"] = run_answers(cases, args.limit)
        _print_answers(result["answers"])

    if exit_code == 0:
        args.results.write_text(
            json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Results written to {args.results}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

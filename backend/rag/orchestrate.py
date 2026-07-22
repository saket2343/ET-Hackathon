"""Iterative retrieval orchestrator + adapters onto the existing system.

The loop (the feature the old pipeline lacked end-to-end):

    understand -> retrieve -> generate -> validate
        -> coverage OK?  return
        -> unsupported claims / missing facets -> new retrieval queries
        -> retrieve more -> merge (dedupe) -> generate again
    until coverage > threshold, no new evidence, or iteration cap.

Adapters (strangler-fig migration): HybridIndexRetriever wraps the verified
Phase-2/3 HybridIndex; ValidatorVerifier wraps the Phase-6 AnswerValidator.
Nothing proven in phases 1-7 is discarded — it is composed behind the new
interfaces.
"""
from __future__ import annotations

import re

from .interfaces import (BaseClaimVerifier, BaseConfidencePolicy,
                         BaseEvidencePlanner, BaseRetriever, BaseRetryPolicy,
                         Generator, QueryUnderstanding, RetrievalPlan)


# ------------------------------------------------------------- adapters

class HybridIndexRetriever:
    """BaseRetriever over the existing HybridIndex + QueryProcessor plan."""

    def __init__(self, index, query_processor):
        self.index = index
        self.qp = query_processor

    def retrieve(self, query: str, plan: RetrievalPlan) -> list[dict]:
        qplan = self.qp.process(query)
        return self.index.retrieve(
            qplan.retrieval_queries[0], query_plan=qplan, k=plan.k)


class ValidatorVerifier:
    """BaseClaimVerifier over the existing AnswerValidator."""

    def __init__(self, validator):
        self.validator = validator

    def verify(self, answer: str, evidence: list[dict]) -> dict:
        repaired, report = self.validator.validate_and_repair(
            answer=answer, evidence=evidence)
        report["answer"] = repaired
        return report


# ---------------------------------------------------------- controller

_CITE = re.compile(r"\s*\[\d+\]")


class IterativeController:
    def __init__(
        self,
        understander,
        retriever: BaseRetriever,
        generator: Generator,
        verifier: BaseClaimVerifier,
        policy: BaseConfidencePolicy,
        retry: BaseRetryPolicy,
        evidence_planner: BaseEvidencePlanner,
        case_file_builder=None,
    ):
        self.understander = understander
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
        self.policy = policy
        self.retry = retry
        self.evidence_planner = evidence_planner
        self.case_file_builder = case_file_builder or self._default_case_file

    # -------------------------------------------------------- helpers

    @staticmethod
    def _default_case_file(query: str, evidence: list[dict]) -> str:
        lines = [f"USER QUESTION: {query}", "", "EVIDENCE:"]
        for i, ch in enumerate(evidence, 1):
            ch.setdefault("evidence_id", i)
            lines.append(f"[{ch['evidence_id']}] ({ch.get('doc_no', '?')} | "
                         f"{ch.get('section', '?')})")
            lines.append(str(ch.get("text", ""))[:1800])
        return "\n".join(lines)

    @staticmethod
    def _merge(evidence: list[dict], extra: list[dict]) -> int:
        seen = {ch.get("chunk_id") for ch in evidence}
        added = 0
        next_id = max((ch.get("evidence_id", 0) for ch in evidence),
                      default=0) + 1
        for ch in extra:
            if ch.get("chunk_id") in seen:
                continue
            ch["evidence_id"] = next_id
            next_id += 1
            evidence.append(ch)
            seen.add(ch.get("chunk_id"))
            added += 1
        return added

    @staticmethod
    def _claim_queries(report: dict, limit: int = 3) -> list[str]:
        """Unsupported claims become retrieval queries (citations stripped)."""
        out = []
        for claim in report.get("claims", []):
            if claim.get("status") == "INSUFFICIENT_EVIDENCE":
                text = _CITE.sub("", claim.get("claim", "")).strip()
                if text:
                    out.append(text)
        return out[:limit]

    # ---------------------------------------------------------- run

    def run(self, raw_query: str) -> dict:
        trace: list[dict] = []
        qu: QueryUnderstanding = self.understander.process(raw_query)
        trace.extend(qu.trace)

        if qu.clarification:
            return {"status": "CLARIFY",
                    "question": qu.clarification.question,
                    "options": qu.clarification.options,
                    "trace": trace}

        # -- initial retrieval: main query + planned subqueries -----------
        evidence: list[dict] = []
        for q in [qu.corrected] + qu.subqueries:
            self._merge(evidence, self.retriever.retrieve(q, qu.plan))
        trace.append({"stage": "retrieve", "queries": 1 + len(qu.subqueries),
                      "evidence": len(evidence)})

        best: dict | None = None
        iteration = 0
        while True:
            case_file = self.case_file_builder(qu.corrected, evidence)
            answer = self.generator(qu.corrected, case_file)
            if not answer:
                trace.append({"stage": "generate", "iteration": iteration,
                              "error": "no answer from generator"})
                break

            report = self.verifier.verify(answer, evidence)
            facet_cov = self.evidence_planner.verify(
                qu.facets, [str(ch.get("text", "")) for ch in evidence])
            claim_cov = report.get("coverage", 0.0)
            coverage = round(min(1.0, 0.7 * claim_cov + 0.3 * facet_cov), 3)
            trace.append({
                "stage": "validate", "iteration": iteration,
                "claim_coverage": claim_cov, "facet_coverage": facet_cov,
                "coverage": coverage,
                "missing_facets": [f.name for f in qu.facets
                                   if f.covered is False],
                "unsupported_claims":
                    report.get("insufficient_evidence_claims", 0),
            })

            candidate = {"answer": report.get("answer", answer),
                         "coverage": coverage, "report": report}
            if best is None or coverage > best["coverage"]:
                best = candidate

            decision = self.policy.decide(coverage, iteration)
            trace.append({"stage": "policy", "iteration": iteration,
                          "action": decision.action, "note": decision.note})

            if decision.action in ("ANSWER", "ANSWER_WITH_NOTE"):
                answer_text = best["answer"]
                if decision.action == "ANSWER_WITH_NOTE" and decision.note:
                    answer_text += f"\n\n{decision.note}"
                return {"status": decision.action, "answer": answer_text,
                        "coverage": best["coverage"],
                        "report": best["report"], "evidence": evidence,
                        "iterations": iteration, "trace": trace}

            if decision.action == "INSUFFICIENT":
                break

            # -- RETRY: turn failures into new retrieval queries ----------
            new_queries = self._claim_queries(report)
            new_queries += [f.probe_query for f in qu.facets
                            if f.covered is False][:2]
            added = 0
            for q in dict.fromkeys(new_queries):
                added += self._merge(evidence,
                                     self.retriever.retrieve(q, qu.plan))
            trace.append({"stage": "iterative_retrieval",
                          "iteration": iteration,
                          "queries": list(dict.fromkeys(new_queries)),
                          "new_evidence": added})
            iteration += 1
            if not self.retry.should_retry(iteration, added):
                trace.append({"stage": "retry_stop", "iteration": iteration,
                              "reason": ("no new evidence" if added == 0
                                         else "iteration cap")})
                break

        insufficient = self.policy.decide(-1.0, 10 ** 6)  # forced terminal
        return {"status": "INSUFFICIENT",
                "answer": insufficient.note,
                "coverage": best["coverage"] if best else 0.0,
                "report": best["report"] if best else {},
                "evidence": evidence,
                "iterations": iteration, "trace": trace}

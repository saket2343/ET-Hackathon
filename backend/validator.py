"""
AXON Answer Validator

Deterministically validates LLM output before returning it.

No LLMs are used here.
"""

from __future__ import annotations

import re


class AnswerValidator:

    def __init__(self, semantic_model=None, nli_model=None):
        self.semantic_model = semantic_model
        # Optional NLI second stage. The relevance model measures "is this
        # claim ABOUT this text" and scores an entity-swapped claim
        # ("LangChain is X" vs evidence saying "LangGraph is X") identically
        # to the true one; NLI measures "does this text STATE this claim".
        self.nli_model = nli_model
        self._nli_ent_idx = None
        self._nli_con_idx = None
        if nli_model is not None:
            try:
                id2label = nli_model.model.config.id2label
                self._nli_ent_idx = next(
                    i for i, l in id2label.items()
                    if l.lower().startswith("entail"))
                self._nli_con_idx = next(
                    i for i, l in id2label.items()
                    if l.lower().startswith("contra"))
            except Exception:
                self.nli_model = None

    def _nli_probs(self, claim: str,
                   premises: list[str]) -> list[tuple[float, float]]:
        """(P(entailment), P(contradiction)) of the claim against each
        premise (premise=evidence, hypothesis=claim)."""
        import math
        try:
            logits = self.nli_model.predict(
                [(p[:600], claim) for p in premises])
        except Exception as exc:
            print("NLI verification failed:", exc)
            return []
        out = []
        for row in logits:
            exps = [math.exp(float(x)) for x in row]
            total = sum(exps) or 1.0
            out.append((exps[self._nli_ent_idx] / total,
                        exps[self._nli_con_idx] / total))
        return out

    def _nli_support(self, claim: str, texts: list[str],
                     max_windows: int = 4) -> tuple[float, float] | None:
        """Best (entailment, contradiction) over SENTENCE WINDOWS of the
        evidence. NLI models are calibrated for sentence-pair premises, not
        1,500-char chunks — measured on the entity-swap probe, entailment
        against the clean two-sentence premise is 0.002 (contradiction
        0.99), but against the full chunk it degrades to 0.68 and the swap
        slips through. Windows of two consecutive sentences are ranked by
        lexical overlap with the claim (the sentences that could support it)
        and only the top few are scored, in one batched call."""
        claim_toks = self._normalize_tokens(claim)
        if not claim_toks:
            return None
        windows: list[tuple[float, str]] = []
        for text in texts:
            # Join hard-wrapped PDF lines BEFORE sentence-splitting.
            # Splitting on newlines produced mid-sentence fragments like
            # "subspaces at different positions. With a single attention
            # head, averaging inhibits this." — half of one sentence glued
            # to the inverse-condition sentence — which NLI judged a 0.98
            # CONTRADICTION of a verbatim-true claim.
            flat = " ".join(text.split())
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat)
                     if len(s.split()) >= 4]
            for i, sent in enumerate(sents):
                window = " ".join(sents[i:i + 2])
                # NLI premises must be ASSERTIONS. A window dominated by
                # capitalized words is a title, heading, or table — the
                # book's cover ("LangChain for AI Engineers — A Complete
                # Practitioner's Guide ...") reads to NLI as entailing
                # "LangChain is a comprehensive knowledge base" (0.996),
                # validating a wrong claim from a noun phrase.
                if self._titleish(window):
                    continue
                overlap = len(claim_toks & self._normalize_tokens(window)) \
                    / len(claim_toks)
                windows.append((overlap, window))
        if not windows:
            return None
        windows.sort(key=lambda x: -x[0])
        chosen = [w for ov, w in windows[:max_windows] if ov >= 0.2]
        if not chosen:
            chosen = [w for _, w in windows[:2]]
        probs = self._nli_probs(claim, chosen)
        if not probs:
            return None
        entail = max(p[0] for p in probs)
        contra = max(p[1] for p in probs)

        # CONJUNCTIVE claims ("the projects include A, B, and C") draw their
        # support from several non-adjacent sentences — no single 2-sentence
        # window can entail the whole conjunction, so a correct list-summary
        # of resume bullets scored ~0 entailment. Handled by CLAIM-side
        # decomposition: each conjunct becomes its own hypothesis carrying
        # the claim's subject+verb ("The projects include a SQL analytics
        # project."), verified against the same single-assertion windows,
        # and ALL conjuncts must hold (min) — one fabricated item sinks the
        # claim. Premise-side fusion was measured UNSAFE: concatenating
        # sentences with different subjects let NLI entail an entity-swapped
        # claim at 0.87; per-item hypotheses keep the swap caught (0.998
        # contradiction) while true list-summaries score 0.99.
        if entail < 0.5:
            hyps = self._decompose_conjunctive(claim)
            if hyps:
                premises = [w for _, w in
                            sorted(windows, key=lambda x: -x[0])[:3]]
                pairs = [(p, h) for h in hyps for p in premises]
                flat = self._nli_probs_pairs(pairs)
                if flat:
                    n = len(premises)
                    per_item = [
                        max(flat[i * n:(i + 1) * n], key=lambda x: x[0])
                        for i in range(len(hyps))
                    ]
                    entail = max(entail, min(p[0] for p in per_item))
                    contra = max(contra, max(p[1] for p in per_item))

        # Contradiction is only meaningful when NOTHING entails: if any
        # premise states the claim, another premise "contradicting" it is
        # either a fragment artifact or a different condition being
        # discussed, and must not veto. (Max-ing both independently capped
        # a claim at 0.30 while a window entailed it at 0.992.)
        if entail >= 0.5:
            contra = 0.0
        return (entail, contra)

    _DOC_META_RE = re.compile(
        r"^(?:the\s+)?\S{0,40}(?:resume|document|paper|pdf|file|manual|"
        r"sop|report|book)\b.{0,24}?\b(?:mentions?|contains?|includes?|"
        r"lists?|provides?|describes?|highlights?|shows?)\b",
        re.IGNORECASE)

    _LIST_VERB_RE = re.compile(
        r"\b(?:includes?|including|included|are|is|was|were|has|have|"
        r"provides?|provided|supports?|built|designed|implemented|created|"
        r"developed|demonstrates?|handles?|covers?|covered|mentions?|"
        r"such as|like)\b", re.IGNORECASE)

    @staticmethod
    def _decompose_conjunctive(claim: str) -> list[str] | None:
        """Split a comma-enumerated claim into per-item hypotheses that keep
        the claim's own subject+verb. Returns None when the claim is not a
        comma list or no head verb is found (whole-claim NLI then stands,
        conservatively)."""
        body = re.sub(r"\s*\[\d+\]", "", claim).rstrip(" .")
        # Explicit list markers bind the head directly: "mentions several
        # projects, including A, B, and C" must decompose as
        # "... including A." / "... including B." — the generic comma-split
        # treated "several projects and roles" as the first item, and its
        # (correct) failure to be entailed sank the whole true claim.
        marker = re.search(r"\b(?:including|such as)\b", body, re.IGNORECASE)
        if marker:
            head = body[: marker.end()].strip()
            tail = body[marker.end():].strip()
            items = [p.strip() for p in
                     re.split(r",\s*(?:and\s+|or\s+)?|\s+and\s+", tail)
                     if p.strip()]
            items = [i for i in items if len(i.split()) >= 2][:4]
            if len(items) >= 2:
                return [f"{head} {item}." for item in items]
        parts = [p.strip() for p in
                 re.split(r",\s*(?:and\s+|or\s+)?", body) if p.strip()]
        if len(parts) < 3:                     # head+first item, >=2 more
            return None
        anchors = list(AnswerValidator._LIST_VERB_RE.finditer(parts[0]))
        if not anchors:
            return None
        head = parts[0][: anchors[-1].end()].strip()
        first_item = parts[0][anchors[-1].end():].strip()
        items = ([first_item] if first_item else []) + parts[1:]
        items = [i for i in items if len(i.split()) >= 2][:4]
        if len(items) < 2:
            return None
        return [f"{head} {item}." for item in items]

    def _nli_probs_pairs(self, pairs: list[tuple[str, str]]
                         ) -> list[tuple[float, float]]:
        """(entail, contra) for explicit (premise, hypothesis) pairs."""
        import math
        try:
            logits = self.nli_model.predict(
                [(p[:600], h) for p, h in pairs])
        except Exception as exc:
            print("NLI verification failed:", exc)
            return []
        out = []
        for row in logits:
            exps = [math.exp(float(x)) for x in row]
            total = sum(exps) or 1.0
            out.append((exps[self._nli_ent_idx] / total,
                        exps[self._nli_con_idx] / total))
        return out

    @staticmethod
    def _titleish(text: str) -> bool:
        """True for title/heading/table text — capitalized-word dominated."""
        words = [w for w in text.split() if w[:1].isalpha()]
        return bool(words) and sum(
            1 for w in words if w[:1].isupper()) / len(words) >= 0.6
    # ---------------------------------------------------------
    def _semantic_score(
    self,
    claim: str,
    evidence_text: str,
) -> float:
        """
        Calculate semantic relevance between a claim
        and an evidence passage using the injected CrossEncoder.

        Returns a normalized score between 0.0 and 1.0.
        """

        if self.semantic_model is None:
            return 0.0

        if not claim.strip() or not evidence_text.strip():
            return 0.0

        try:
            raw_score = self.semantic_model.predict(
                [(claim, evidence_text)]
            )[0]

            # Convert CrossEncoder raw logit to 0-1
            # using the sigmoid function.
            import math

            normalized_score = (
                1.0
                /
                (
                    1.0
                    + math.exp(-float(raw_score))
                )
            )

            return round(
                normalized_score,
                4,
            )

        except Exception as exc:

            print(
                "Semantic validation failed:",
                exc,
            )

            return 0.0




    def _extract_claims(
    self,
    answer: str,
) -> list[str]:
        """
        Extract complete factual claim units from the
        evidence-grounded portion of the answer.

        Preserves numbered and bulleted list items while
        excluding headings, structural lead-ins, and fragments.
        """

        claims = []

        # ---------------------------------------------------------
        # 1. Remove markdown headings
        # ---------------------------------------------------------

        text = re.sub(
            r"^#{1,6}\s+.*$",
            "",
            answer,
            flags=re.MULTILINE,
        ).strip()

        # ---------------------------------------------------------
        # 2. Normalize numbered lists onto separate lines
        # ---------------------------------------------------------

        text = re.sub(
            r"(?<!^)(?<!\n)\s+(\d+)\.\s+",
            r"\n\1. ",
            text,
        )

        # ---------------------------------------------------------
        # 3. Process blocks line by line
        # ---------------------------------------------------------

        blocks = re.split(
            r"\n{2,}",
            text,
        )

        for block in blocks:

            block = block.strip()

            if not block:
                continue

            lines = [
                line.strip()
                for line in block.splitlines()
                if line.strip()
            ]

            for line in lines:

                # Remove bullet / numbered-list / blockquote markers.
                # These are LINE PREFIXES, so the remaining sentence is
                # still an exact substring of the original answer.
                line = re.sub(
                    r"^\s*(?:[-*•>]|\d+[.)])\s+",
                    "",
                    line,
                ).strip()

                # Inline markdown (**bold**, `code`) is intentionally
                # PRESERVED: the claim text must remain an exact substring
                # of the answer, because citation repair locates claims
                # with str.replace. The old strip-markdown step meant any
                # formatted sentence could never be found again, silently
                # turning repair into a no-op. Token/semantic scoring is
                # unaffected (the token regex ignores punctuation).

                if not line:
                    continue

                # -------------------------------------------------
                # Skip Markdown table rows and separators.
                #
                # "| Library | Purpose | Capabilities |" is a header row
                # and "| --- | --- |" a separator — presentation, not
                # factual claims. Extracting them produced spurious
                # UNVERIFIABLE / INSUFFICIENT_EVIDENCE flags (11 bogus
                # citations were being inserted into table pipes). The
                # CONTENT of table cells is still validated when the LLM
                # also states it in prose, which it does for comparisons.
                # -------------------------------------------------

                if line.lstrip().startswith("|"):
                    continue
                if re.fullmatch(r"[\s|:\-]+", line):
                    continue

                # -------------------------------------------------
                # Skip structural lead-ins ending with ":"
                #
                # Examples:
                # "LangGraph introduces capabilities like:"
                # "The workflow requires:"
                # "The main differences are:"
                # -------------------------------------------------

                if line.endswith(":"):
                    continue

                # -------------------------------------------------
                # Split normal prose into sentences
                # -------------------------------------------------

                sentences = re.split(
                    r"(?<=[.!?])\s+",
                    line,
                )

                for sentence in sentences:

                    sentence = sentence.strip()

                    # Ignore very short fragments
                    if len(sentence.split()) < 5:
                        continue

                    # Questions are not factual claims — validating them
                    # against evidence produces false unsupported flags.
                    if sentence.rstrip("*_` ").endswith("?"):
                        continue

                    # Skip evidence-metadata ECHOES ("Relevance: medium,
                    # Entities: CUDA, Content: Built ...") — small models
                    # copy case-file header lines into answers — and
                    # meta-citation filler ("This project is mentioned in
                    # [1].", "This is supported by [3]."): statements about
                    # citations, not facts. Both classes produced spurious
                    # INSUFFICIENT_EVIDENCE warnings.
                    if re.match(
                        r"^\s*(?:\[\d+\]\s*)?(?:Relevance|Entities|Content|"
                        r"Source|Keywords|Concepts)\s*:", sentence,
                    ):
                        continue
                    if re.fullmatch(
                        r"(?:\[\d+\]\s*)?(?:This|It|These|Those)"
                        r"(?:\s+[\w-]+){0,4}\s+(?:is|are)\s+"
                        r"(?:mentioned|supported|described|documented)\s+"
                        r"(?:in|by)\s+\[\d+\]\s*\.?",
                        sentence, re.IGNORECASE,
                    ):
                        continue

                    # Interpretive commentary is not a factual claim.
                    # Uncited speculation ("could be a scenario where...",
                    # "would use their experience to...", "has likely
                    # honed...") and skill-praise ("these projects showcase
                    # the individual's skills in...") assert the WRITER'S
                    # inference, not evidence content — demanding citations
                    # for them produced five false flags on one resume
                    # answer. They stay in the answer; they exit strict
                    # claim-level validation. Cited speculation still gets
                    # its citation alignment checked.
                    if (re.search(r"\b(?:would|could|might|likely)\b",
                                  sentence, re.IGNORECASE)
                            and not re.search(r"\[\d+\]", sentence)):
                        continue
                    if re.search(
                        r"\b(?:showcas\w*|demonstrat\w*|highlight\w*|"
                        r"underscor\w*|reflect\w*)\b.{0,60}?"
                        r"\b(?:skills?|abilit\w+|expertise|experience|"
                        r"proficienc\w+)\b",
                        sentence, re.IGNORECASE,
                    ):
                        continue

                    # Skip dangling structural introductions
                    if re.search(
                        r"""
                        (?:
                            requiring
                            | include
                            | includes
                            | including
                            | such\s+as
                            | following
                            | introduces
                            | capabilities
                            | consists\s+of
                            | comprised\s+of
                            | are
                        )
                        \s*(?:like)?\s*:\s*$
                        """,
                        sentence,
                        flags=re.IGNORECASE | re.VERBOSE,
                    ):
                        continue

                    claims.append(sentence)

        # ---------------------------------------------------------
        # 4. Remove duplicates while preserving order
        # ---------------------------------------------------------

        return list(
            dict.fromkeys(claims)
        )
    def _claim_evidence_status(
    self,
    claim: str,
    evidence: list[dict],
) -> dict:

        result = self._assess_against_evidence(
            claim=claim,
            evidence_items=evidence,
        )

        best_evidence = None

        best_chunk = result.get(
            "best_chunk"
        )

        if best_chunk is not None:

            best_evidence = {
                "evidence_id":
                    result["best_evidence_id"],

                "chunk_id":
                    best_chunk.get("chunk_id"),

                "doc_no":
                    best_chunk.get("doc_no"),

                "section":
                    best_chunk.get("section"),

                "matched_terms":
                    result["matched_terms"],

                "lexical_score":
                    result.get("lexical_score"),

                "semantic_score":
                    result.get("semantic_score"),
            }

        return {
            "status":
                result["status"],

            "support_score":
                result["score"],

            "best_evidence":
                best_evidence,

            "supporting_evidence":
                result["supporting_evidence"],

            "support_mode":
                result["support_mode"],

            "matched_terms":
                result["matched_terms"],

            # Exposed so downstream suppression can tell a well-grounded claim
            # a weak model merely mis-cited (no contradiction) from a claim the
            # evidence actively refutes (entity swap → high contradiction).
            "nli_contradiction":
                result.get("nli_contradiction"),
        }

    def _assess_against_evidence(
    self,
    claim: str,
    evidence_items: list[dict],
) -> dict:
        """
        Assess whether one or more evidence chunks support a claim.

        This is the SINGLE source of truth used by:
            - _claim_evidence_status()
            - _citation_alignment()

        Supports:
            - lexical overlap
            - semantic similarity
            - single-evidence support
            - multi-evidence synthesis
        """

        clean_claim = re.sub(
            r"\[\d+\]",
            "",
            claim,
        ).strip()

        claim_tokens = self._normalize_tokens(
            clean_claim
        )

        if not clean_claim or not claim_tokens:
            return {
                "score": 0.0,
                "status": "UNVERIFIABLE",
                "best_evidence_id": None,
                "supporting_evidence": [],
                "support_mode": "NONE",
                "matched_terms": [],
                "individual_scores": {},
            }

        if not evidence_items:
            return {
                "score": 0.0,
                "status": "INSUFFICIENT_EVIDENCE",
                "best_evidence_id": None,
                "supporting_evidence": [],
                "support_mode": "NONE",
                "matched_terms": [],
                "individual_scores": {},
            }

        scored = []

        for position, chunk in enumerate(
            evidence_items,
            start=1,
        ):
            evidence_id = chunk.get(
                "evidence_id",
                position,
            )

            evidence_text = " ".join([
                str(chunk.get("text", "")),
                str(chunk.get("summary", "")),
                " ".join(chunk.get("keywords", [])),
                " ".join(chunk.get("concepts", [])),
                " ".join(chunk.get("entities", [])),
            ]).strip()

            evidence_tokens = self._normalize_tokens(
                evidence_text
            )

            matched_terms = (
                claim_tokens
                &
                evidence_tokens
            )

            lexical_score = (
                len(matched_terms)
                /
                len(claim_tokens)
                if claim_tokens
                else 0.0
            )

            semantic_score = (
                self._semantic_score(
                    clean_claim,
                    evidence_text,
                )
            )

            # Semantic similarity is the primary meaning signal.
            # Lexical overlap provides grounding confirmation.
            score = (
                0.70 * semantic_score
                +
                0.30 * lexical_score
            )

            scored.append({
                "evidence_id": evidence_id,
                "chunk": chunk,
                "score": score,
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "matched_terms": matched_terms,
                "tokens": evidence_tokens,
            })

        scored.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        best = scored[0]

        # Only evidence with some meaningful relationship
        # participates in synthesis.
        synthesis_candidates = [
            item
            for item in scored
            if (
                item["semantic_score"] >= 0.40
                or item["lexical_score"] >= 0.25
            )
        ][:3]

        combined_tokens = set()
        supporting_evidence = []

        for item in synthesis_candidates:
            combined_tokens.update(
                item["tokens"]
            )

            supporting_evidence.append(
                item["evidence_id"]
            )

        combined_matched = (
            claim_tokens
            &
            combined_tokens
        )

        combined_lexical = (
            len(combined_matched)
            /
            len(claim_tokens)
            if claim_tokens
            else 0.0
        )

        best_score = best["score"]

        # Multi-evidence synthesis should improve actual
        # claim coverage, not merely add more chunks.
        if (
            len(synthesis_candidates) > 1
            and combined_lexical
            > best["lexical_score"] + 0.10
        ):
            final_score = max(
                best_score,
                (
                    0.70 * best["semantic_score"]
                    +
                    0.30 * combined_lexical
                ),
            )

            support_mode = (
                "MULTI_EVIDENCE_SYNTHESIS"
            )

        else:
            final_score = best_score

            support_mode = "SINGLE_EVIDENCE"

            supporting_evidence = [
                best["evidence_id"]
            ]

        # NLI second stage (monotone tightener): runs only on claims the
        # relevance stage would bless, against the top evidence candidates.
        # Entailment can only LOWER the score, never raise it — relevance
        # calibration is preserved for cases NLI is unsure about, while
        # entity-swapped / plausible-but-wrong claims (relevance ~1.0,
        # entailment ~0.0) drop below the support thresholds. A hard
        # contradiction (evidence states otherwise) caps the score outright.
        nli_entailment = None
        nli_contradiction = None
        # Document-containment META-claims ("The person_resume mentions a
        # SQL analytics project", "The manual lists torque values") cannot
        # be verified by NLI: no evidence sentence self-references its own
        # document, so entailment is structurally ~0 even when the claim is
        # true (measured: 0.002 for a correct containment claim). Whether a
        # document mentions X IS the lexical/relevance check — those stages'
        # verdict stands; fabricated containment still fails on low lexical
        # coverage.
        is_doc_meta = bool(self._DOC_META_RE.match(clean_claim))
        if self.nli_model is not None and final_score >= 0.45 \
                and not is_doc_meta:
            top_texts = [
                str(item["chunk"].get("text", ""))
                for item in (synthesis_candidates or [best])[:2]
                if item["chunk"].get("text")
            ]
            support = self._nli_support(clean_claim, top_texts)
            if support is not None:
                nli_entailment = round(support[0], 3)
                nli_contradiction = round(support[1], 3)
                lex_part = (combined_lexical
                            if support_mode == "MULTI_EVIDENCE_SYNTHESIS"
                            else best["lexical_score"])
                strict = 0.70 * nli_entailment + 0.30 * lex_part
                final_score = min(final_score, strict)
                if nli_contradiction >= 0.90:
                    final_score = min(final_score, 0.30)

        # Numeric grounding check: semantic similarity cannot tell a real
        # number from a fabricated one ("released in 2019", "10x faster").
        # If the claim asserts digit-numbers that appear in NONE of the
        # supporting passages, it must not be classified fully SUPPORTED.
        unsupported_numbers: list[str] = []
        claim_numbers = set(re.findall(r"\d+(?:\.\d+)?", clean_claim))
        if claim_numbers:
            support_pool = synthesis_candidates or [best]
            evidence_blob = " ".join(
                str(item["chunk"].get("text", "")) for item in support_pool
            )
            unsupported_numbers = sorted(
                n for n in claim_numbers if n not in evidence_blob
            )
            if unsupported_numbers:
                final_score = min(final_score, 0.65)

        # One classification policy only.
        if final_score >= 0.70:
            status = "SUPPORTED"

        elif final_score >= 0.45:
            status = "PARTIALLY_SUPPORTED"

        else:
            status = "INSUFFICIENT_EVIDENCE"

        return {
            "score": round(final_score, 2),
            "status": status,

            "nli_entailment": nli_entailment,
            "nli_contradiction": nli_contradiction,

            "unsupported_numbers": unsupported_numbers,

            "best_evidence_id":
                best["evidence_id"],

            "best_chunk":
                best["chunk"],

            "supporting_evidence":
                supporting_evidence,

            "support_mode":
                support_mode,

            "matched_terms":
                sorted(combined_matched),

            "individual_scores": {
                item["evidence_id"]: round(
                    item["score"],
                    2,
                )
                for item in scored
            },

            "semantic_score": round(
                best["semantic_score"],
                2,
            ),

            "lexical_score": round(
                best["lexical_score"],
                2,
            ),
        }





    def _grounded_answer_text(
    self,
    answer: str,
) -> str:
        """
        Return only answer sections containing positive factual claims
        that should be validated against retrieved evidence.

        Excludes:
            - Limitations
            - General Knowledge
            - Sources
        """

        text = answer

        # ---------------------------------------------------------
        # 1. Remove General Knowledge section and everything after it
        # ---------------------------------------------------------

        text = re.split(
            r"##?\s*General Knowledge"
            r"\s*\(Not from Retrieved Documents\)",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        # ---------------------------------------------------------
        # 2. Remove Sources section and everything after it
        # ---------------------------------------------------------

        text = re.split(
            r"##?\s*Sources",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        # ---------------------------------------------------------
        # 3. Remove ONLY the Limitations section
        #
        # Preserve later sections such as Summary.
        # ---------------------------------------------------------

        text = re.sub(
            r"(?ims)"
            r"^##?\s*Limitations\s*$"
            r".*?"
            r"(?=^##?\s+\S|\Z)",
            "",
            text,
        )

        # ---------------------------------------------------------
        # 4. Remove the Suggested Follow-up Questions section.
        #
        # The answer template asks the model to END with suggested
        # follow-up questions. They are questions, not factual
        # claims — validating them against evidence counted every
        # suggestion as INSUFFICIENT_EVIDENCE and dragged coverage
        # down for answers that correctly followed the template.
        # ---------------------------------------------------------

        text = re.sub(
            r"(?ims)"
            r"^##?\s*Suggested Follow-up Questions?\s*$"
            r".*?"
            r"(?=^##?\s+\S|\Z)",
            "",
            text,
        )

        # Verification notices appended by the critic on a previous
        # pass are pipeline output, not model claims.
        text = re.sub(
            r"(?ms)^⚠️ Verification Notice\n.*?(?=\n\n|\Z)",
            "",
            text,
        )

        return text.strip()


    def _normalize_tokens(
    self,
    text: str,
) -> set[str]:
        """
        Normalize text into comparable semantic tokens.

        Shared by claim-support validation and citation alignment
        so both validators use the same lexical rules.
        """

        stopwords = {
            "this", "that", "these", "those",
            "with", "from", "into", "using",
            "used", "also", "than", "then",
            "have", "has", "had", "does",
            "were", "was", "are", "the",
            "and", "for", "but", "not",
            "can", "could", "would", "should",
            "about", "through", "between",
            "its", "their", "there", "which",
            "when", "where", "while",
        }

        canonical_terms = {
            "loops": "loop",
            "looping": "loop",

            "pauses": "pause",
            "paused": "pause",
            "pausing": "pause",

            "states": "state",
            "stateful": "state",

            "checkpointers": "checkpoint",
            "checkpointer": "checkpoint",
            "checkpointing": "checkpoint",
            "checkpoints": "checkpoint",

            "schemas": "schema",

            "simpler": "simple",
            "simplest": "simple",
            "simplicity": "simple",

            "workflows": "workflow",

            "sequences": "sequence",
            "sequential": "sequence",

            "branches": "branch",
            "branching": "branch",
            "branched": "branch",

            "durability": "durable",

            "iterations": "iteration",
            "iterative": "iteration",

            "capabilities": "capability",

            "applications": "application",

            "graphs": "graph",

            "subgraphs": "subgraph",
        }

        tokens = set()

        for token in re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_\-]*",
            text.lower(),
        ):
            if len(token) <= 2:
                continue

            if token in stopwords:
                continue

            token = canonical_terms.get(
                token,
                token,
            )

            tokens.add(token)

        return tokens

        
    def _citation_alignment(
    self,
    claim: str,
    evidence: list[dict],
) -> dict:

        citation_ids = list(
            dict.fromkeys(
                int(c)
                for c in re.findall(
                    r"\[(\d+)\]",
                    claim,
                )
            )
        )

        if not citation_ids:
            return {
                "citation_status": "UNCITED",
                "citations": [],
                "best_citation": None,
                "alignment_score": 0.0,
                "individual_alignment_scores": {},
                "combined_alignment_score": 0.0,
                "support_mode": "NONE",
                "matched_terms": [],
            }

        evidence_by_id = {
            chunk.get(
                "evidence_id",
                position,
            ): chunk
            for position, chunk in enumerate(
                evidence,
                start=1,
            )
        }

        cited_evidence = [
            evidence_by_id[citation_id]
            for citation_id in citation_ids
            if citation_id in evidence_by_id
        ]

        valid_citations = [
            citation_id
            for citation_id in citation_ids
            if citation_id in evidence_by_id
        ]

        if not cited_evidence:
            return {
                "citation_status": "MISMATCHED",
                "citations": [],
                "best_citation": None,
                "alignment_score": 0.0,
                "individual_alignment_scores": {},
                "combined_alignment_score": 0.0,
                "support_mode": "NONE",
                "matched_terms": [],
            }

        result = self._assess_against_evidence(
            claim=claim,
            evidence_items=cited_evidence,
        )

        status_map = {
            "SUPPORTED":
                "ALIGNED",

            "PARTIALLY_SUPPORTED":
                "WEAKLY_ALIGNED",

            "INSUFFICIENT_EVIDENCE":
                "MISMATCHED",

            "UNVERIFIABLE":
                "UNVERIFIABLE",
        }

        return {
            "citation_status":
                status_map[result["status"]],

            "citations":
                valid_citations,

            "best_citation":
                result["best_evidence_id"],

            "alignment_score":
                result["score"],

            "individual_alignment_scores":
                result["individual_scores"],

            "combined_alignment_score":
                result["score"],

            "support_mode":
                result["support_mode"],

            "matched_terms":
                result["matched_terms"],
        }
        
    def _repair_missing_citations(
    self,
    answer: str,
    report: dict,
    min_support_score: float = 0.70,
) -> tuple[str, int]:
        """
        Add a citation only when:

        1. A factual claim is currently UNCITED.
        2. The validator found strong evidence for it.
        3. The support score meets the minimum threshold.
        4. A valid best evidence ID exists.

        This method does NOT repair mismatched citations.
        It only repairs missing citations.
        """

        repaired_answer = answer
        repairs = 0

        for claim_report in report.get(
            "claims",
            [],
        ):

            # ---------------------------------------------
            # Only repair uncited claims
            # ---------------------------------------------

            if (
                claim_report.get("citation_status")
                != "UNCITED"
            ):
                continue

            # ---------------------------------------------
            # Require strong enough evidence support
            # ---------------------------------------------

            support_score = claim_report.get(
                "support_score",
                0.0,
            )

            if support_score < min_support_score:
                continue

            # ---------------------------------------------
            # Get strongest supporting evidence
            # ---------------------------------------------

            best_evidence = claim_report.get(
                "best_evidence"
            )

            if not best_evidence:
                continue

            evidence_id = best_evidence.get(
                "evidence_id"
            )

            if evidence_id is None:
                continue

            # ---------------------------------------------
            # Find the exact original claim
            # ---------------------------------------------

            claim = claim_report.get(
                "claim",
                "",
            ).strip()

            if not claim:
                continue

            if claim not in repaired_answer:
                continue

            # ---------------------------------------------
            # Prevent accidental duplicate citation
            # ---------------------------------------------

            cited_claim = (
                f"{claim} [{evidence_id}]"
            )

            repaired_answer = repaired_answer.replace(
                claim,
                cited_claim,
                1,
            )

            repairs += 1

        return repaired_answer, repairs



    def validate(
    self,
    answer: str,
    evidence: list[dict],
) -> tuple[str, dict]:
        """
        Validate the generated answer against retrieved evidence.

        Checks:
        - Citation validity using stable evidence IDs
        - Claim-level evidence support
        - Citation-to-claim alignment
        - Required answer structure
        - Overall evidence coverage
        """

        report = {
            "valid": True,
            "issues": [],
            "coverage": 0.0,

            # Citation validity
            "citation_count": 0,
            "invalid_citations": [],

            # Claim verification
            "claims": [],
            "supported_claims": 0,
            "partially_supported_claims": 0,
            "insufficient_evidence_claims": 0,
            "claim_support_score": 0.0,

            # Citation-to-claim alignment
            "aligned_citations": 0,
            "weakly_aligned_citations": 0,
            "mismatched_citations": 0,
            "uncited_claims": 0,
            "citation_alignment_score": 0.0,

            # Format
            "format_score": 0.0,

            # Final evidence verdict
            "evidence_verdict": "UNKNOWN",
        }

        # =====================================================
        # 1. Citation validity
        # =====================================================

        citations = re.findall(
            r"\[(\d+)\]",
            answer,
        )

        report["citation_count"] = len(citations)

        # Build the set of valid stable evidence IDs.
        #
        # If evidence_id exists, use it.
        # Otherwise fall back to the original list position.
        valid_evidence_ids = {
            chunk.get("evidence_id", position)
            for position, chunk in enumerate(
                evidence,
                start=1,
            )
        }

        for citation in citations:

            idx = int(citation)

            if idx not in valid_evidence_ids:
                report["invalid_citations"].append(idx)

        # Remove duplicate invalid citation numbers
        # while preserving their order.
        report["invalid_citations"] = list(
            dict.fromkeys(
                report["invalid_citations"]
            )
        )

        if report["invalid_citations"]:

            report["issues"].append(
                "Invalid citation numbers."
            )

        # =====================================================
        # 2. Extract grounded claims
        # =====================================================

        # General Knowledge and Sources are removed before
        # evidence-grounding checks.
        grounded_text = self._grounded_answer_text(
            answer
        )

        claims = self._extract_claims(
            grounded_text
        )

        # =====================================================
        # 3. Claim support + citation alignment
        # =====================================================

        for claim in claims:

            # ---------------------------------------------
            # Check whether the claim is supported by
            # any retrieved evidence.
            # ---------------------------------------------

            assessment = self._claim_evidence_status(
                claim=claim,
                evidence=evidence,
            )

            # ---------------------------------------------
            # Check whether citations attached to this
            # claim actually support the claim.
            # ---------------------------------------------

            citation_alignment = self._citation_alignment(
                claim=claim,
                evidence=evidence,
            )

            status = assessment["status"]

            citation_status = citation_alignment[
                "citation_status"
            ]

            # Store complete diagnostics for this claim.
            report["claims"].append({
                "claim": claim,
                **assessment,
                **citation_alignment,
            })

            # ---------------------------------------------
            # Claim support counters
            # ---------------------------------------------

            if status == "SUPPORTED":

                report["supported_claims"] += 1

            elif status == "PARTIALLY_SUPPORTED":

                report[
                    "partially_supported_claims"
                ] += 1

            else:

                report[
                    "insufficient_evidence_claims"
                ] += 1

            # ---------------------------------------------
            # Citation alignment counters
            # ---------------------------------------------

            if citation_status == "ALIGNED":

                report["aligned_citations"] += 1

            elif citation_status == "WEAKLY_ALIGNED":

                report[
                    "weakly_aligned_citations"
                ] += 1

            elif citation_status == "MISMATCHED":

                report[
                    "mismatched_citations"
                ] += 1

            elif citation_status == "UNCITED":

                report["uncited_claims"] += 1

            # UNVERIFIABLE is intentionally not counted
            # as a citation mismatch.

        # =====================================================
        # 4. Claim support score
        # =====================================================

        total_claims = len(
            report["claims"]
        )

        if total_claims:

            claim_score = (
                report["supported_claims"]
                +
                0.5
                * report[
                    "partially_supported_claims"
                ]
            ) / total_claims

        else:

            claim_score = 0.0

        report["claim_support_score"] = round(
            claim_score,
            2,
        )

        # =====================================================
        # 5. Citation alignment score
        # =====================================================

        citation_checked = (
            report["aligned_citations"]
            +
            report["weakly_aligned_citations"]
            +
            report["mismatched_citations"]
        )

        if citation_checked:

            citation_alignment_score = (
                report["aligned_citations"]
                +
                0.5
                * report[
                    "weakly_aligned_citations"
                ]
            ) / citation_checked

        else:

            citation_alignment_score = 0.0

        report["citation_alignment_score"] = round(
            citation_alignment_score,
            2,
        )

        # =====================================================
        # 6. Format checks
        # =====================================================

        if "Direct Answer" not in answer:

            report["issues"].append(
                "Missing Direct Answer section."
            )

        if "Sources" not in answer:

            report["issues"].append(
                "Missing Sources section."
            )

        if len(answer.strip()) < 80:

            report["issues"].append(
                "Answer suspiciously short."
            )

        # Summary is intentionally NOT mandatory.
        # Step 4 allows unnecessary sections to be omitted.

        format_score = max(
            0.0,
            1.0
            - len(report["issues"])
            * 0.15,
        )

        report["format_score"] = round(
            format_score,
            2,
        )

        # =====================================================
        # 7. Overall coverage
        # =====================================================

        report["coverage"] = round(
            0.65 * claim_score
            +
            0.25 * citation_alignment_score
            +
            0.10 * format_score,
            2,
        )

        # =====================================================
        # 8. Evidence verdict
        # =====================================================

        insufficient = report[
            "insufficient_evidence_claims"
        ]

        partial = report[
            "partially_supported_claims"
        ]

        supported = report[
            "supported_claims"
        ]

        if (
            insufficient == 0
            and partial == 0
            and supported > 0
        ):

            verdict = "SUPPORTED"

        elif (
            insufficient == 0
            and (
                supported > 0
                or partial > 0
            )
        ):

            verdict = "PARTIALLY_SUPPORTED"

        elif supported > insufficient:

            verdict = "PARTIALLY_SUPPORTED"

        else:

            verdict = "INSUFFICIENT_EVIDENCE"

        report["evidence_verdict"] = verdict

        # =====================================================
        # 9. Final validity
        # =====================================================

        report["valid"] = (
            report["coverage"] >= 0.70
            and not report["invalid_citations"]
            and report["mismatched_citations"] == 0
        )

        return answer, report
    

    def validate_and_repair(
    self,
    answer: str,
    evidence: list[dict],
    min_support_score: float = 0.60,
) -> tuple[str, dict]:
        """
        Validate an answer and repair citation problems once.

        Repairs:
            1. Strongly supported uncited claims
            2. Mismatched citations when better evidence exists
            3. Weakly aligned citations when stronger evidence exists

        Does NOT:
            - invent evidence IDs
            - cite unsupported claims
            - enter repeated repair loops
        """

        # =====================================================
        # 1. Initial validation
        # =====================================================

        _, initial_report = self.validate(
            answer=answer,
            evidence=evidence,
        )

        repaired_answer = answer
        repair_count = 0

        # Valid evidence IDs
        valid_evidence_ids = {
            chunk.get("evidence_id", position)
            for position, chunk in enumerate(
                evidence,
                start=1,
            )
        }

        # =====================================================
        # 2. Process each validated claim
        # =====================================================

        for claim_data in initial_report.get(
            "claims",
            [],
        ):

            original_claim = claim_data.get(
                "claim",
                "",
            )

            if not original_claim:
                continue

            support_status = claim_data.get(
                "status"
            )

            support_score = claim_data.get(
                "support_score",
                0.0,
            )

            citation_status = claim_data.get(
                "citation_status"
            )

            # -------------------------------------------------
            # Never attach evidence to unsupported claims
            # -------------------------------------------------

            if (
                support_status
                == "INSUFFICIENT_EVIDENCE"
                or support_score < min_support_score
            ):
                continue

            # -------------------------------------------------
            # Find strongest supporting evidence IDs
            # -------------------------------------------------

            supporting_ids = [
                evidence_id
                for evidence_id in claim_data.get(
                    "supporting_evidence",
                    [],
                )
                if evidence_id in valid_evidence_ids
            ]

            # Fall back to best evidence
            if not supporting_ids:

                best_evidence = claim_data.get(
                    "best_evidence"
                )

                if best_evidence:

                    best_id = best_evidence.get(
                        "evidence_id"
                    )

                    if best_id in valid_evidence_ids:
                        supporting_ids = [
                            best_id
                        ]

            if not supporting_ids:
                continue

            # Remove duplicates while preserving order
            supporting_ids = list(
                dict.fromkeys(
                    supporting_ids
                )
            )

            new_citations = "".join(
                f"[{evidence_id}]"
                for evidence_id in supporting_ids
            )

            # -------------------------------------------------
            # Case A — Claim is uncited
            # -------------------------------------------------

            if citation_status == "UNCITED":

                repaired_claim = (
                    original_claim.rstrip()
                    + " "
                    + new_citations
                )

            # -------------------------------------------------
            # Case B — Existing citation is mismatched
            # -------------------------------------------------

            elif citation_status == "MISMATCHED":

                clean_claim = re.sub(
                    r"\s*\[\d+\]",
                    "",
                    original_claim,
                ).rstrip()

                repaired_claim = (
                    clean_claim
                    + " "
                    + new_citations
                )

            # -------------------------------------------------
            # Case C — Weak citation alignment
            #
            # Replace only when validator found stronger
            # supporting evidence than the current citations.
            # -------------------------------------------------

            elif citation_status == "WEAKLY_ALIGNED":

                current_citations = set(
                    claim_data.get(
                        "citations",
                        [],
                    )
                )

                stronger_citations = set(
                    supporting_ids
                )

                # Nothing useful to change
                if (
                    current_citations
                    == stronger_citations
                ):
                    continue

                clean_claim = re.sub(
                    r"\s*\[\d+\]",
                    "",
                    original_claim,
                ).rstrip()

                repaired_claim = (
                    clean_claim
                    + " "
                    + new_citations
                )

            else:
                continue

            # -------------------------------------------------
            # Replace exact claim once
            # -------------------------------------------------

            if original_claim in repaired_answer:

                repaired_answer = (
                    repaired_answer.replace(
                        original_claim,
                        repaired_claim,
                        1,
                    )
                )

                repair_count += 1

        # =====================================================
        # 3. Nothing changed
        # =====================================================

        if repair_count == 0:

            initial_report[
                "citation_repairs"
            ] = 0

            return answer, initial_report

        # =====================================================
        # 4. Revalidate exactly once
        # =====================================================

        _, final_report = self.validate(
            answer=repaired_answer,
            evidence=evidence,
        )

        final_report[
            "citation_repairs"
        ] = repair_count

        return repaired_answer, final_report
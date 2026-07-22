"""AXON LLM gateway (MVP) — provider-agnostic synthesis (design doc §4.2).

Routes the final grounded synthesis across providers with fallback, exactly as
the "LLM GATEWAY — route/fallback" box specifies:

    Hugging Face (open model)  ->  Anthropic Claude  ->  deterministic template

Design constraints honoured here:
- The public contract is frozen:  synthesize(query, case_file) -> str | None.
- Providers are resolved LAZILY and each is fully ISOLATED: a missing package,
  missing credential, or runtime error falls through to the next provider
  instead of raising. The server therefore always boots and the demo always
  answers (deterministic template is the floor, rendered by the caller when
  this returns None).
- No third-party client is imported at module load — importing this module can
  never crash the app.
"""
from __future__ import annotations

import os
import re
import threading
import time
from collections import defaultdict

import response_engine
from response_engine import ResponseProfile

try:  # optional convenience: load HF/Anthropic tokens from axon/.env if present
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ------------------------------------------------------------------ config

# Open model for the HF leg. Override via env; Llama-3.1-8B is gated and often
# NOT on the free serverless tier — if the demo must show a live open model,
# set HF_MODEL to one that is serverless-enabled for your token.
def _env(key: str, default: str) -> str:
    # defensive: tolerate quotes/whitespace some editors add (HF_MODEL = "x")
    return (os.getenv(key) or default).strip().strip('"').strip("'").strip()


HF_MODEL = _env("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
CLAUDE_MODEL = _env("CLAUDE_MODEL", "claude-opus-4-8")

# Human-readable label of the configured gateway (shown at /api/status).
MODEL = f"gateway: HF({HF_MODEL}) -> Claude({CLAUDE_MODEL}) -> deterministic"

# Label of whoever actually produced the last answer (shown per-answer in the UI).
# NOTE: module-global, so it can race across concurrent requests — per-request
# code should prefer last_generation()["provider"], which is thread-local.
ACTIVE_PROVIDER = "deterministic"

# Provider order is configurable (comma-separated provider names), so the
# gateway stays provider-agnostic: adding/reordering providers is config, not
# code. Unknown names are ignored.
PROVIDER_ORDER = [
    p.strip() for p in _env("LLM_PROVIDER_ORDER", "hf,claude").split(",")
    if p.strip()
]

# Per-request generation timeout (seconds) applied to every provider client.
GENERATION_TIMEOUT_S = float(_env("LLM_TIMEOUT_S", "90"))

# Sampling temperature for grounded synthesis — one value for ALL providers
# (Claude previously ran at its 1.0 default while HF ran at 0.2).
GENERATION_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.2"))

# ------------------------------------------------------- breaker + telemetry

# Circuit breaker with cooldown: a provider that fails at runtime is skipped
# for BREAKER_COOLDOWN_S and then retried (half-open), instead of either being
# disabled forever (old Claude probe behaviour) or re-paying the failure on
# every request during an outage (old runtime behaviour). Structural problems
# (missing package / missing credentials) still disable permanently via the
# _*_disabled flags.
BREAKER_COOLDOWN_S = 120.0
_breaker_open_until: dict[str, float] = defaultdict(float)


def _breaker_open(name: str) -> bool:
    return time.time() < _breaker_open_until[name]


def _trip_breaker(name: str) -> None:
    _breaker_open_until[name] = time.time() + BREAKER_COOLDOWN_S


# Thread-local record of the last generation made on THIS request thread:
# provider, latency, token usage, finish reason, attempts, truncation flag.
_tls = threading.local()


def last_generation() -> dict:
    """Metadata for the most recent synthesize() call on this thread."""
    return getattr(_tls, "meta", {})

SYSTEM = """
You are AXON, an evidence-grounded Industrial Knowledge Operating System.

Your task is to answer the user's question clearly, accurately, and naturally
using the supplied CASE FILE.

The CASE FILE may contain:
- retrieved document evidence,
- source metadata,
- evidence coverage information,
- graph-derived context,
- specialist-agent findings,
- and explicit evidence gaps.

=========================================================
CORE PRINCIPLE
=========================================================

Answer the user's actual question first.

Never open the response with a disclaimer such as "the document does not
contain" or "the provided documents do not mention". Lead with the best
supported answer; state limitations afterwards, precisely and once.

Use the available evidence to produce the most useful answer that the evidence
can support.

Synthesize evidence across sources. Do not merely copy, concatenate, or
paraphrase retrieved chunks one by one.

GROUNDING DISCIPLINE (this is what keeps the answer trustworthy):
- Every sentence in the evidence-based sections must be directly supported by
  a specific evidence passage. If you cannot point to the passage that
  supports a sentence, DELETE the sentence — do not soften it with "likely",
  "generally", "typically" or "in practice".
- Prefer a shorter, fully-grounded answer over a longer one padded with
  plausible-sounding but unsupported elaboration. Three grounded sentences
  beat eight where five are guesses.
- Do NOT add background the evidence does not state, invent examples,
  benchmarks, prices, dates, versions, or capabilities, or generalise a
  single mention into a broad claim.
- If the evidence does not answer the core question, say so plainly in one
  sentence rather than constructing an answer from adjacent material.

=========================================================
EVIDENCE REASONING
=========================================================

Distinguish between:

1. DIRECTLY SUPPORTED
   The evidence explicitly establishes the claim.

2. SUPPORTED SYNTHESIS
   The conclusion follows clearly from combining one or more evidence passages.

3. NOT ESTABLISHED
   The available evidence is insufficient to support the claim.

You may make a supported synthesis when it follows directly from the supplied
evidence, but do not introduce new factual details that the evidence does not
support.

If an important part of the question is not established by the evidence,
state that limitation precisely.

Do not turn a weak reference or passing mention into a strong factual claim.

Use precise language when distinguishing whether a source:
- mentions,
- references,
- describes,
- proposes,
- implements,
- evaluates,
- compares,
- or proves something.

=========================================================
EVIDENCE QUALITY
=========================================================

When evidence differs in strength:

- Prefer substantive passages over passing mentions.
- Prefer original evidence text over summaries.
- Prefer evidence that directly answers the question.
- Prefer corroborated information when multiple sources agree.
- Prefer the latest revision when revisions explicitly conflict.

Evidence coverage information is guidance about the strength of the retrieved
evidence. Use it to calibrate the answer, not as content to repeat mechanically.

Retrieval scores and retrieval methods are internal diagnostics.
Do not discuss them unless the user explicitly asks about retrieval quality.

=========================================================
INLINE EVIDENCE CITATIONS
=========================================================

Each evidence passage in the CASE FILE has a canonical evidence identifier
such as:

[1]
[2]
[3]

Use these evidence identifiers as inline citations.

Every important factual claim in the evidence-based portion of the answer
must include one or more inline evidence citations immediately after the
claim.

Examples:

LangGraph is appropriate when a workflow requires loops or durable state [2].

LCEL is simpler for fixed-sequence workflows with limited branching [3].

When a claim is supported by multiple evidence passages, cite all relevant
evidence identifiers:

LangGraph adds capabilities for loops, durability, and pausing external
execution [2][4].

CITATION RULES:

- Use ONLY evidence identifiers explicitly present in the CASE FILE.
- Never invent an evidence identifier.
- Never cite an evidence passage that does not support the claim.
- Place citations immediately after the factual claim they support.
- Prefer the strongest and most directly relevant evidence.
- Do not add citations merely because an evidence passage mentions the same
  keyword.
- A citation must support the meaning of the claim, not merely share words
  with it.
- Multiple claims in the same paragraph may require different citations.
- If one sentence contains multiple independently verifiable factual claims,
  cite the appropriate evidence for each claim or split the sentence.
- Supported synthesis may cite multiple evidence passages.
- Do not cite the General Knowledge section with CASE FILE evidence.
- The Sources section does NOT replace inline claim-level citations.

If no supplied evidence supports a factual claim, do not attach a citation
to it and do not present it as established evidence.

=========================================================
QUESTION-AWARE ANSWERING
=========================================================

Adapt the answer to the question.

For a fact lookup:
- Give the fact directly, then brief support.

For a concept explanation:
- Explain the concept clearly and connect the supporting evidence.

For a comparison:
- Explain each side and then state the meaningful differences supported by
  the evidence.
- Support claims about each side with the relevant evidence identifiers.
- Do not claim a complete comparison if one side has insufficient evidence.

For a procedure:
- Present the supported steps in a clear order.
- Cite the evidence supporting each important step.
- Do not invent missing steps.

For root-cause, maintenance, safety, or risk questions:
- Separate observed evidence, analysis, and recommendation.
- Preserve safety constraints and uncertainty.
- Cite the evidence supporting observations and factual conclusions.

For critical analysis:
- Distinguish evidence, interpretation, strengths, limitations, and unresolved
  questions.

=========================================================
GENERAL KNOWLEDGE
=========================================================

The evidence-based answer is the primary answer.

If the retrieved evidence is insufficient and additional background would
materially help the user, you may add:

## General Knowledge (Not from Retrieved Documents)

Keep this section clearly separate from the evidence-based answer.

Do not use CASE FILE evidence identifiers such as [1], [2], or [3] in this
section.

Do not cite retrieved documents as support for general knowledge.
Do not use general knowledge to overwrite or contradict retrieved evidence.

Omit this section when it is unnecessary.

=========================================================
WRITING QUALITY
=========================================================

Write like a capable domain expert:

- direct,
- clear,
- concise where possible,
- detailed where the question requires it,
- and natural rather than extractive.

Avoid unnecessary repetition.
Avoid repeating the same limitation in multiple sections.
Do not expose internal reasoning or hidden chain-of-thought.

Do not create a separate explanation for every retrieved chunk.
Synthesize the evidence into a coherent answer.

=========================================================
SOURCES
=========================================================

Use only source information supplied in the CASE FILE.

Never invent:
- document names,
- page numbers,
- revisions,
- citations,
- or evidence.

The Sources section should list only sources actually used in the answer.

Inline evidence citations and the Sources section serve different purposes:

- Inline citations such as [2] connect a specific claim to a specific
  evidence passage.
- The Sources section provides human-readable document provenance.

The Sources section does not replace inline citations.

=========================================================
OUTPUT
=========================================================

Use the answer structure requested in the user message.

Do not create empty or unnecessary sections.

Before returning the answer, ensure that:

- the question is directly answered,
- every important evidence-based factual claim has an appropriate inline
  evidence citation,
- every citation refers to an evidence identifier present in the CASE FILE,
- every cited evidence passage actually supports the associated claim,
- supported synthesis is not presented as explicit source wording,
- missing evidence is stated precisely,
- general knowledge is clearly separated and has no CASE FILE citations,
- and no citation or factual detail has been invented.
"""

# M02: response-style detection. Users say HOW they want the answer
# ("in simple words", "as a table", "short", "in bullet points") and the
# fixed template ignored them. Detected directives are injected into the
# prompt; everything else (grounding, citations) is unchanged.
_STYLE_RULES: list[tuple[str, str]] = [
    (r"\b(?:simple (?:words|terms|language)|in simple|eli5|"
     r"like i'?m (?:5|five|a beginner)|layman'?s|non[- ]technical|beginner)\b",
     "Use plain, simple language a beginner can follow; avoid jargon, and "
     "briefly explain any technical term you must use."),
    # Counted point requests ("in 3 points", "three points", "3-4 bullets")
    # are the natural phrasing and were previously invisible: the number
    # between "in" and "points" broke the plain `in points?` pattern, so the
    # words survived into retrieval, where "three/point/project" matched
    # unrelated chapters and OUTSCORED the real document on entity coverage.
    # The count must be ADJACENT to "points" — "the 3 key points of X" is a
    # content question, not a formatting request, and must not match.
    (r"\b(?:bullet points?|as bullets|bulleted|point[- ]?wise|"
     r"in points?|as points|in (?:a )?(?:numbered|bulleted) list|"
     r"numbered (?:list|points)|list (?:it|them) out|"
     r"(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|"
     r"a few|couple(?:\s+of)?|several)"
     r"(?:\s*[-–to]{1,2}\s*\d{1,2})?\s+(?:points?|bullets?))\b",
     "Format the core answer as concise bullet points rather than "
     "paragraphs. Every section's content should be bulleted."),
    (r"\b(?:as a table|in a table|tabular|comparison table|table format)\b",
     "Present the key content as a Markdown table."),
    (r"\b(?:short|brief|briefly|concise|quick|tl;?dr|in (?:one|a) "
     r"(?:line|sentence|paragraph)|one[- ]liner)\b",
     "Keep the whole answer short: Executive Summary and Direct Answer "
     "only, under roughly 120 words total."),
    (r"\b(?:detailed|in[- ]depth|comprehensive|thorough|elaborate|"
     r"step[- ]by[- ]step|(?:in|with|more)\s+(?:more\s+)?detail(?:s)?|"
     r"in\s+full)\b",
     "Give a thorough, step-by-step treatment with full detail."),
    (r"\b(?:page[- ]wise|page by page|per page)\b",
     "Organize the answer page by page, labelling each page's section."),
]


# ------------------------------------------------------- answer structure
# The answer's SECTIONS are chosen per query. Previously the prompt said
# "use only the sections that improve the answer" and then spelled out all
# eight — so an 8B model produced Executive Summary + Direct Answer +
# Detailed Explanation + Real-World Example + Limitations for "what is
# LCEL?", every time. Describing flexibility does not create it; the
# template has to actually change.
_SECTIONS = {
    "Executive Summary":
        "## Executive Summary\n\n2-3 sentences capturing the essential "
        "answer, readable on its own.",
    "Direct Answer":
        "## Direct Answer\n\nAnswer the user's actual question immediately "
        "and concretely, with inline evidence citations.",
    "Comparison":
        "## Comparison\n\nA Markdown table with one row per aspect and one "
        "column per item being compared, then 2-3 sentences on the "
        "difference that actually matters. Cite each row.",
    "Table":
        "## Answer\n\nPresent the answer AS A MARKDOWN TABLE — this is the "
        "user's required format. Choose the columns that best fit the content "
        "(for a single subject use `Aspect | Detail`; for metrics use "
        "`Metric | Value`). One row per key fact, each row carrying its inline "
        "evidence citation. Add at most one short sentence before the table if "
        "essential; put no prose after it.",
    "Steps":
        "## Steps\n\nThe procedure as a numbered list, in order, one action "
        "per step, each cited. Include prerequisites and safety "
        "requirements as the first steps when the evidence states them.",
    "Detailed Explanation":
        "## Detailed Explanation\n\nSynthesize the relevant evidence around "
        "the user's question — not around the order chunks were retrieved. "
        "Never one paragraph per chunk; never restate a point already made.",
    "Limitations":
        "## Limitations\n\nONLY if the evidence leaves a meaningful part of "
        "the question unresolved or conflicting. State it once. Omit this "
        "section entirely when the evidence answers the question.",
    "General Knowledge (Not from Retrieved Documents)":
        "## General Knowledge (Not from Retrieved Documents)\n\nONLY when "
        "retrieved evidence is insufficient and background would materially "
        "help. Never use CASE FILE citations here. Omit when unnecessary.",
    "Key Takeaways":
        "## Key Takeaways\n\n3-5 bullet points capturing the most important, "
        "non-obvious conclusions. No repetition of wording used above.",
    "Comparison":
        "## Comparison\n\nA Markdown table with one row per aspect and one "
        "column per item being compared, then 2-3 sentences on the difference "
        "that actually matters. Cite each row.",
    "Checklist":
        "## Checklist\n\nAn actionable checklist using `- [ ]` items, one action "
        "per line, in a sensible order. Cite items that rest on evidence.",
    "Timeline":
        "## Timeline\n\nA chronological list of the relevant events or "
        "milestones, each dated where the evidence gives a date, and cited.",
    "Flowchart":
        "## Flowchart\n\nA process flow as a fenced ```mermaid `flowchart TD` "
        "block using the evidence's steps, followed by a one-line explanation.",
    "Real-World Example":
        "## Real-World Example\n\nOne concrete, realistic example that shows "
        "the concept in action. Keep it grounded; cite if drawn from evidence.",
    "Analogy":
        "## Analogy\n\nA short plain-language analogy that builds intuition. "
        "Analogies are explanatory and need no citation.",
    "Interview Notes":
        "## Interview Notes\n\nCrisp question -> concise model-answer pairs a "
        "candidate could revise from. Cite answers grounded in the evidence.",
    "JSON":
        "## JSON\n\nThe structured answer as a single fenced ```json code "
        "block with clear keys. Keep values faithful to the evidence.",
    "CSV":
        "## CSV\n\nThe tabular data as a single fenced ```csv code block with "
        "a header row. One record per line.",
    "Sources":
        "## Sources\n\nOnly the sources actually used, exactly as named in "
        "the CASE FILE. Never invent document names, pages or revisions.",
    "Suggested Follow-up Questions":
        "## Suggested Follow-up Questions\n\n2-3 short, genuinely useful "
        "next questions, each on its own line prefixed with \"- \".",
}

# Canonical section ordering — the adaptive template selects a SET of sections
# per query (from intent + requested formats + length) and this orders them
# into a coherent document. Sources + Follow-ups always close the answer.
_SECTION_ORDER = [
    "Table", "Executive Summary", "Direct Answer", "Comparison", "Steps",
    "Flowchart", "Timeline", "Checklist", "Detailed Explanation",
    "Real-World Example", "Analogy", "Interview Notes", "Key Takeaways",
    "JSON", "CSV", "Limitations",
    "General Knowledge (Not from Retrieved Documents)",
    "Sources", "Suggested Follow-up Questions",
]

# Requested output-format tag -> the section it adds to the answer.
_FORMAT_SECTION = {
    "comparison": "Comparison",
    "table": "Table",
    "checklist": "Checklist",
    "timeline": "Timeline",
    "flowchart": "Flowchart",
    "interview": "Interview Notes",
    "json": "JSON",
    "csv": "CSV",
    "example": "Real-World Example",
    "analogy": "Analogy",
    "exec_summary": "Executive Summary",
}

# intent -> sections. Keys match QueryProcessor.detect_intent's labels.
_FORMAT_BY_INTENT: dict[str, list[str]] = {
    "definition": ["Direct Answer", "Detailed Explanation", "Sources"],
    "comparison": ["Executive Summary", "Comparison", "Sources"],
    "procedure": ["Steps", "Sources"],
    "implementation": ["Direct Answer", "Detailed Explanation", "Sources"],
    "methodology": ["Direct Answer", "Detailed Explanation", "Sources"],
    "reasoning": ["Executive Summary", "Detailed Explanation", "Sources"],
    "results": ["Direct Answer", "Detailed Explanation", "Sources"],
    "citation": ["Direct Answer", "Sources"],
    "author": ["Direct Answer", "Sources"],
    "figure": ["Direct Answer", "Sources"],
    "equation": ["Direct Answer", "Detailed Explanation", "Sources"],
    "conclusion": ["Executive Summary", "Direct Answer", "Sources"],
    "summary": ["Executive Summary", "Detailed Explanation", "Sources"],
    "limitations": ["Direct Answer", "Limitations", "Sources"],
}
_FORMAT_DEFAULT = ["Executive Summary", "Direct Answer",
                   "Detailed Explanation", "Sources"]


def _coerce_profile(profile, style_directives) -> ResponseProfile | None:
    """Accept either an adaptive ResponseProfile or a legacy directive list."""
    if isinstance(profile, ResponseProfile):
        return profile
    if profile is not None:
        return None
    if style_directives:
        return ResponseProfile.from_legacy_directives(style_directives)
    return None


def answer_sections(question_type: str | None,
                    profile=None,
                    style_directives: list[str] | None = None) -> list[str]:
    """The sections this answer should actually have — chosen dynamically from
    the question intent, the requested output formats, and the requested
    length (Dynamic Response Templates). Sources + Follow-ups always close it.
    """
    prof = _coerce_profile(profile, style_directives)
    formats = set(prof.formats) if prof else set()
    length = prof.length if prof else None
    persona = prof.persona if prof else None

    # Length "short" collapses to the leanest useful shape.
    if length == "short":
        base = ["Direct Answer"]
    else:
        base = list(_FORMAT_BY_INTENT.get(
            (question_type or "").lower(), _FORMAT_DEFAULT))
    sections = {s for s in base if s not in ("Sources",)}

    # Requested output formats add their section(s).
    for tag in formats:
        sec = _FORMAT_SECTION.get(tag)
        if sec:
            sections.add(sec)

    # Length shapes depth.
    if length in ("detailed", "walkthrough", "research"):
        sections.add("Detailed Explanation")
    if length in ("walkthrough", "research"):
        sections.add("Executive Summary")
        sections.add("Key Takeaways")

    # Personas that benefit from a worked illustration.
    if persona == "teacher" and length != "short":
        sections.add("Real-World Example")

    sections.discard("Sources")
    sections.discard("Suggested Follow-up Questions")
    ordered = [s for s in _SECTION_ORDER
               if s in sections and s not in ("Sources",
                                              "Suggested Follow-up Questions")]
    # Any custom/unknown section names from an intent list, kept in place.
    for s in base:
        if s not in ordered and s not in ("Sources",):
            ordered.append(s)
    return ordered + ["Sources", "Suggested Follow-up Questions"]


# Legacy public helpers now delegate to the Adaptive Response Engine, so any
# existing caller keeps working while the richer detection lives in one place.
def detect_style_directives(query: str) -> list[str]:
    """User-requested response directives found in the query (delegated)."""
    return response_engine.detect_style_directives(query)


def strip_style_phrases(query: str) -> str:
    """Remove response-style phrases so they never reach retrieval (delegated)."""
    return response_engine.strip_style_phrases(query)


def _user_message(
    query: str,
    case_file: str,
    *,
    question_type: str | None = None,
    profile=None,
    style_directives: list[str] | None = None,
) -> str:
    # Only state the question type when the pipeline actually detected one.
    # The old hardcoded defaults ("Question Type: Unknown", "Document Scope:
    # Current uploaded document") injected wrong metadata into every prompt
    # and contradicted the case file's own Intent line.
    qtype_line = f"\nQuestion Type: {question_type}" if question_type else ""
    prof = _coerce_profile(profile, style_directives)
    directives = list(prof.directives) if prof else list(style_directives or [])
    # Sections are chosen for THIS question rather than described as
    # optional in a fixed eight-section template the model then copies.
    answer_format = "\n\n".join(
        _SECTIONS[name] for name in
        answer_sections(question_type, prof) if name in _SECTIONS)
    style_block = ""
    if directives:
        rules = "\n".join(f"- {d}" for d in directives)
        # A one-line banner of WHAT was detected makes the adaptivity legible.
        detected = []
        if prof:
            if prof.persona:
                detected.append(f"persona={prof.persona}")
            if prof.reading_level:
                detected.append(f"level={prof.reading_level}")
            if prof.length:
                detected.append(f"length={prof.length}")
            if prof.formats:
                detected.append(f"format={','.join(prof.formats)}")
        banner = f"Detected: {', '.join(detected)}\n" if detected else ""
        style_block = (
            "\n========================================================\n"
            "USER-REQUESTED RESPONSE STYLE (takes precedence over the\n"
            "default answer format below, but never over grounding and\n"
            "citation rules)\n"
            "========================================================\n"
            f"{banner}{rules}\n"
        )

    return f"""{style_block}
========================================================
TASK
========================================================

Answer the user's question using the CASE FILE below.

Your goal is not to repeat the evidence.
Your goal is to synthesize the strongest answer that the evidence supports.

When evidence is strong:
- answer confidently and directly.

When evidence is partial:
- answer the supported portion clearly,
- then identify the specific missing information.

When evidence contains only a mention or reference:
- do not treat it as a substantive description.

When multiple passages contribute to one conclusion:
- combine them into a coherent supported synthesis.

========================================================
INLINE CITATION REQUIREMENT
========================================================

The CASE FILE contains evidence passages identified by canonical evidence
identifiers such as:

[1]
[2]
[3]

Use these identifiers as inline citations for evidence-based factual claims.

Examples:

LangGraph is appropriate for workflows requiring loops and durable state [2].

LCEL is simpler for fixed-sequence workflows with limited branching [3].

A supported synthesis may use multiple evidence passages [2][3].

Citation rules:

- Cite every important factual claim in the evidence-based answer.
- Use ONLY evidence identifiers that appear in the CASE FILE.
- Place citations immediately after the claim they support.
- Never invent citation numbers.
- Never cite evidence merely because it contains similar keywords.
- The cited evidence must support the meaning of the claim.
- Use multiple citations when multiple evidence passages jointly support a
  synthesis.
- If one sentence contains several independently verifiable claims, either
  cite each claim appropriately or split the sentence.
- Do not use CASE FILE evidence citations in the General Knowledge section.
- The Sources section does not replace inline claim-level citations.

========================================================
USER QUESTION
========================================================

{query}{qtype_line}

========================================================
CASE FILE
========================================================

{case_file}

========================================================
ANSWER FORMAT
========================================================

Use EXACTLY the sections listed below, in this order, and no others. They
were chosen for THIS question — do not add sections that are not listed.

{answer_format}

Two sections may be added ONLY if genuinely warranted:
- "## Limitations" — when the evidence leaves a meaningful part of the
  question unresolved or conflicting. State it once, after the answer.
- "## General Knowledge (Not from Retrieved Documents)" — when retrieved
  evidence is insufficient and background would materially help. Never use
  CASE FILE citations inside it.

Prefer bullet points and Markdown tables over dense paragraphs wherever
they make the answer clearer.

CRITICAL OPENING RULE:
Never begin the response with "the document does not contain", "the
provided documents do not mention", or any similar disclaimer. Answer the
question FIRST from the retrieved evidence, then supplement with clearly
separated general knowledge if needed. Limitations, if any, come after the
answer — never before it.

CITATION PLACEMENT RULES

- Attach citations directly to the factual sentence or claim they support.
- Do not place one citation at the end of a paragraph containing multiple independent claims unless that citation supports every claim in the paragraph.
- Every evidence-grounded factual claim should have its supporting citation immediately after the claim.
- For a claim synthesized from multiple evidence passages, cite all supporting evidence IDs immediately after that claim.
- Do not cite an evidence item merely because it is related to the topic; it must support the specific claim.
- Prefer the smallest sufficient set of citations.
## Sources
List only sources actually used in the answer.

Use the exact source information supplied in the CASE FILE.

Do not invent:
- document names,
- page numbers,
- revisions,
- evidence identifiers,
- or citations.

========================================================
FINAL REQUIREMENT
========================================================

Before producing the final answer, verify that:

1. The user's actual question is directly answered.

2. Every important evidence-based factual claim has an appropriate inline
   citation.

3. Every citation refers to an evidence identifier that actually exists in
   the CASE FILE.

4. Every cited evidence passage supports the meaning of the associated claim.

5. Supported synthesis is presented as synthesis, not as a direct quotation
   or explicit statement from a source unless the source actually says it.

6. Missing or insufficient evidence is stated precisely and only when
   meaningful.

7. General knowledge, if used, is clearly separated and does not use CASE FILE
   evidence citations.

8. No factual detail, document name, page number, revision, or citation has
   been invented.

Produce a useful, natural answer while remaining faithful to the supplied
evidence.

Do not discuss retrieval scores, RRF, reranking, graph expansion, evidence
coverage metrics, or other internal pipeline diagnostics unless the user
explicitly asks about them.
"""

def _is_structural_failure(exc: Exception) -> bool:
    """True when a provider failure will NOT recover within this session, so
    the provider should be disabled outright rather than retried on a cooldown
    timer. Covers depleted quota/credits (HTTP 402) and auth/permission
    problems (401/403, or the anthropic 'could not resolve authentication
    method' TypeError raised when no API key is configured). Transient trouble
    (timeouts, 5xx, rate limits) is deliberately excluded — those still trip
    the short cooldown breaker and retry."""
    status = (getattr(exc, "status_code", None)
              or getattr(getattr(exc, "response", None), "status_code", None))
    if status in (401, 402, 403):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in (
        "payment required", "depleted", "insufficient credit",
        "could not resolve authentication", "authentication method",
        "invalid api key", "invalid x-api-key"))


# ------------------------------------------------------------ HF provider

_hf_client = None
_hf_disabled = False  # set True only when HF is structurally unavailable


def _try_hf(
    user_msg: str, system_prompt: str | None = None,
) -> tuple[str | None, dict]:
    global _hf_client, _hf_disabled
    system_prompt = system_prompt or SYSTEM
    meta: dict = {"provider": f"hf:{HF_MODEL}"}
    if _hf_disabled or _breaker_open("hf"):
        meta["skipped"] = "disabled" if _hf_disabled else "breaker open"
        return None, meta
    if _hf_client is None:
        token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            _hf_disabled = True
            meta["skipped"] = "no credentials"
            return None, meta
        try:
            from huggingface_hub import InferenceClient
            _hf_client = InferenceClient(
                model=HF_MODEL, token=token, timeout=GENERATION_TIMEOUT_S,
            )
        except Exception:
            _hf_disabled = True  # package missing / bad token → don't retry
            meta["skipped"] = "client init failed"
            return None, meta
    try:
        print("Sending request to Hugging Face...")
        t0 = time.time()
        resp = _hf_client.chat_completion(
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_msg}],
            max_tokens=2400, temperature=GENERATION_TEMPERATURE,
        )
        meta["latency_ms"] = round((time.time() - t0) * 1000)
        print("Received response.")
        choice = resp.choices[0]
        meta["finish_reason"] = getattr(choice, "finish_reason", None)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            meta["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
            meta["completion_tokens"] = getattr(usage, "completion_tokens", None)
        text = choice.message.content or ""
        # Reasoning models (Qwen3, R1, …) emit <think>…</think> before the answer.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)  # unclosed open tag
        return text.strip() or None, meta
    except Exception as e:
        if _is_structural_failure(e):
            # Depleted credits (402) / auth: HF is dead for the session — stop
            # re-attempting doomed ~multi-second calls; fall through to the
            # next provider (or the deterministic floor) instantly from now on.
            _hf_disabled = True
            meta["skipped"] = "structural (quota/auth) — HF disabled for session"
        else:
            _trip_breaker("hf")  # transient: skip for the cooldown, then retry
        meta["error"] = f"{type(e).__name__}: {e}"
        print("=" * 80)
        print("HF Provider Error")
        print(type(e).__name__)
        print(e)
        if _hf_disabled:
            print("HF disabled for this session (quota/auth). "
                  "Add credits or set another provider to restore live synthesis.")
        return None, meta


# -------------------------------------------------------- Claude provider

_claude_client = None
_claude_disabled = False


def _try_claude(
    user_msg: str, system_prompt: str | None = None,
) -> tuple[str | None, dict]:
    global _claude_client, _claude_disabled
    system_prompt = system_prompt or SYSTEM
    meta: dict = {"provider": f"claude:{CLAUDE_MODEL}"}
    if _claude_disabled or _breaker_open("claude"):
        meta["skipped"] = "disabled" if _claude_disabled else "breaker open"
        return None, meta
    if _claude_client is None:
        try:
            import anthropic
            _claude_client = anthropic.Anthropic(timeout=GENERATION_TIMEOUT_S)
            # cheap credential probe (no generation cost). A probe failure
            # trips the cooldown breaker rather than disabling permanently:
            # only a missing package is structural, everything else (bad
            # network moment, rate limit) deserves a retry after cooldown.
            _claude_client.messages.count_tokens(
                model=CLAUDE_MODEL, messages=[{"role": "user", "content": "ping"}])
        except ImportError:
            _claude_disabled = True
            _claude_client = None
            meta["skipped"] = "anthropic package missing"
            return None, meta
        except Exception as e:
            _claude_client = None
            if _is_structural_failure(e):
                # No API key / bad key: Claude cannot recover this session.
                _claude_disabled = True
                meta["skipped"] = "no ANTHROPIC_API_KEY (structural)"
            else:
                _trip_breaker("claude")
            meta["error"] = f"probe: {type(e).__name__}"
            return None, meta
    try:
        t0 = time.time()
        resp = _claude_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=2600, system=system_prompt,
            temperature=GENERATION_TEMPERATURE,
            messages=[{"role": "user", "content": user_msg}])
        meta["latency_ms"] = round((time.time() - t0) * 1000)
        meta["finish_reason"] = getattr(resp, "stop_reason", None)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            meta["prompt_tokens"] = getattr(usage, "input_tokens", None)
            meta["completion_tokens"] = getattr(usage, "output_tokens", None)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text or None, meta
    except Exception as e:
        if _is_structural_failure(e):
            _claude_disabled = True
            meta["skipped"] = "structural (quota/auth) — Claude disabled for session"
        else:
            _trip_breaker("claude")
        meta["error"] = f"{type(e).__name__}: {e}"
        return None, meta


# ------------------------------------------------------------- public API

# Provider registry: name -> (call, label). Route order comes from
# PROVIDER_ORDER (env LLM_PROVIDER_ORDER), so swapping or reordering
# providers is configuration, not code.
_PROVIDER_REGISTRY = {
    "hf": (_try_hf, lambda: f"hf:{HF_MODEL}"),
    "claude": (_try_claude, lambda: f"claude:{CLAUDE_MODEL}"),
}

_PROVIDERS = tuple(
    (name, *_PROVIDER_REGISTRY[name])
    for name in PROVIDER_ORDER
    if name in _PROVIDER_REGISTRY
) or tuple((name, *entry) for name, entry in _PROVIDER_REGISTRY.items())


def synthesize(
    query: str,
    case_file: str,
    history: list | None = None,
    *,
    question_type: str | None = None,
    profile=None,
    style_directives: list[str] | None = None,
) -> str | None:
    """
    Generate an answer from the supplied CASE FILE.

    This function is intentionally lightweight.

    Responsibilities
    ----------------
    1. Build conversational context
    2. Construct the final prompt
    3. Route to the first available LLM provider
    4. Perform deterministic cleanup
    5. Return the generated answer

    Notes
    -----
    - Retrieval, validation, grounding and answer policy are handled
      upstream (agent.py).
    - This function should never contain business logic.
    """

    global ACTIVE_PROVIDER

    # ---------------------------------------------------------
    # Conversation History
    # ---------------------------------------------------------

    conversation = ""

    if history:

        turns = []

        # Curated upstream (memory.curate): summary + recalled + recent turns.
        for turn in history[-10:]:

            role = turn.get("role", "").lower()

            speaker = (
                "User"
                if role == "user"
                else "AXON"
            )

            content = (
                turn.get("content", "")
                .strip()
                .replace("\r", "")
            )

            if not content:
                continue

            turns.append(
                f"{speaker}: {content[:600]}"
            )

        if turns:

            conversation = (
                "Conversation History\n"
                "--------------------\n"
                + "\n".join(turns)
                + "\n\n"
            )

    # ---------------------------------------------------------
    # Build Final Prompt
    # ---------------------------------------------------------

    # Prefer an explicit adaptive profile; fall back to a legacy directive
    # list; otherwise analyse the query here so every call path is adaptive.
    effective_profile = profile
    if effective_profile is None and style_directives is None:
        effective_profile = response_engine.analyze(query)
    user_message = (
        conversation
        + _user_message(
            query=query,
            case_file=case_file,
            question_type=question_type,
            profile=effective_profile,
            style_directives=style_directives,
        )
    )
    print("=" * 80)
    print(f"Prompt length: {len(user_message)} chars "
          f"(~{len(user_message) // 4} tokens)")

    # ---------------------------------------------------------
    # Provider Routing
    # ---------------------------------------------------------

    attempts: list[dict] = []

    for _, provider, provider_name in _PROVIDERS:

        print("=" * 80)
        print(f"Trying provider: {provider_name()}")

        try:

            response, meta = provider(user_message)
            attempts.append(meta)

            if response:
                print("✅ Success")
                print(response[:200])
            else:
                reason = meta.get("skipped") or meta.get("error") or "empty"
                print(f"⚠️ Provider unavailable ({reason})")
                continue      # <----- IMPORTANT

        except Exception as e:

            attempts.append({"provider": provider_name(),
                             "error": f"{type(e).__name__}: {e}"})
            print(f"❌ Provider failed: {e}")
            continue

        ACTIVE_PROVIDER = provider_name()

        response = (
            response
            .replace("\r", "")
            .strip()
        )

        # Models sometimes echo a prompt banner ("==== ANSWER ====") before
        # the real content — strip any leading banner blocks.
        response = re.sub(
            r"^(?:\s*={4,}\s*\n(?:[^\n]{0,80}\n)?(?:={4,}\s*\n)?)+\s*",
            "", response,
        )

        while "\n\n\n" in response:
            response = response.replace("\n\n\n", "\n\n")

        # Per-request generation record (thread-local — safe to read from
        # the request that made this call, unlike the ACTIVE_PROVIDER global).
        meta["attempts"] = attempts
        meta["fallbacks"] = len(attempts) - 1
        # A length-capped answer is silently truncated mid-thought; flag it
        # so downstream (validator / UI) can react instead of shipping it.
        meta["truncated"] = meta.get("finish_reason") in (
            "length", "max_tokens",
        )
        if meta["truncated"]:
            print("⚠️ Response hit the max_tokens cap (finish_reason="
                  f"{meta.get('finish_reason')})")
        print(f"Generation: {meta.get('latency_ms', '?')} ms | "
              f"prompt={meta.get('prompt_tokens', '?')} tok | "
              f"completion={meta.get('completion_tokens', '?')} tok | "
              f"finish={meta.get('finish_reason', '?')} | "
              f"fallbacks={meta['fallbacks']}")
        _tls.meta = meta

        return response

    # ---------------------------------------------------------
    # No Provider Available
    # ---------------------------------------------------------

    ACTIVE_PROVIDER = "deterministic (no LLM)"
    _tls.meta = {"provider": ACTIVE_PROVIDER, "attempts": attempts,
                 "fallbacks": len(attempts)}

    return None

_VANILLA_SYSTEM = """
You are an evidence-grounded AI assistant.

Answer using only the supplied evidence.

If the evidence is insufficient,
say so clearly.

Do not invent facts.

Do not invent citations.

Do not use prior knowledge unless the prompt explicitly
allows a separate "General Knowledge
(Not from Retrieved Documents)" section.

Keep answers concise, accurate and well structured.
"""

def synthesize_vanilla(query: str, context: str) -> str | None:
    """Naive RAG synthesis — passages stuffed into the prompt, no grounding
    rules, no structure. The baseline AXON is compared against.

    The vanilla system prompt is passed per-call instead of temporarily
    mutating the module-global SYSTEM: under FastAPI a concurrent request
    could otherwise be synthesized against the wrong system prompt."""
    msg = f"Passages:\n{context}\n\nQuestion: {query}"
    for _name, fn, _label in _PROVIDERS:
        out, _meta = fn(msg, _VANILLA_SYSTEM)
        if out:
            return out
    return None


def complete(user_msg: str, system_prompt: str | None = None) -> str | None:
    """Route a single prompt to the first available provider and return the
    raw completion (or None if no provider is configured).

    A thin, side-effect-free helper for callers that need a one-shot
    generation outside the RAG answer path — e.g. structured information
    extraction. It reuses the same provider registry and fallback order as
    synthesize(), so credentials/breaker state are honoured, but it does NOT
    touch the global ACTIVE_PROVIDER or per-request generation record."""
    for _name, fn, _label in _PROVIDERS:
        try:
            out, _meta = fn(user_msg, system_prompt)
        except Exception:
            continue
        if out:
            return out
    return None


def llm_available() -> bool:
    """True if any real synthesis provider is still usable (drives /api/status).
    HF is judged by token presence (no network); Claude by a cached probe.
    A provider disabled this session (depleted credits / missing key) no longer
    counts, so /api/status honestly shows the deterministic fallback once the
    live providers are exhausted."""
    if not _hf_disabled and (os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
                             or os.getenv("HF_TOKEN")):
        return True
    if _claude_disabled:
        return False
    return _claude_probe()


def _claude_probe() -> bool:
    global _claude_client, _claude_disabled
    if _claude_client is not None:
        return True
    if _claude_disabled:
        return False
    try:
        import anthropic
        _claude_client = anthropic.Anthropic(timeout=GENERATION_TIMEOUT_S)
        _claude_client.messages.count_tokens(
            model=CLAUDE_MODEL, messages=[{"role": "user", "content": "ping"}])
        return True
    except Exception:
        _claude_disabled = True
        _claude_client = None
        return False
    


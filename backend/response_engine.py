"""Adaptive Response Engine (Module 02).

Analyses every user query BEFORE answer generation and produces a
``ResponseProfile`` describing HOW the user wants the answer:

  * output formats   — bullets, table, comparison, checklist, interview notes,
                       JSON, CSV, Markdown, timeline, flowchart, page-wise,
                       executive summary, example, analogy
  * reading level    — beginner, student, engineer, manager, expert
  * response length  — short, detailed, research, walkthrough
  * persona          — teacher, engineer, researcher, auditor, manager,
                       technician, interviewer

The profile is turned into (a) a list of natural-language *directives* injected
into the synthesis prompt and (b) a stripped retrieval query with the styling
phrases removed (so "explain attention as a table for a beginner" retrieves on
"attention", not on "table"/"beginner").

This module is intentionally dependency-free (only ``re``/``dataclasses``) and
imports nothing from the rest of the backend, so it can be reused anywhere and
can never cause an import cycle. ``llm.py`` delegates its legacy
``detect_style_directives`` / ``strip_style_phrases`` helpers here, and
``agents.py`` calls :func:`analyze` to drive the whole adaptive flow.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Queries longer than this are pasted content / instructions, not styling
# requests — scanning them would misread ordinary prose as directives.
_MAX_SCAN_CHARS = 400


# --------------------------------------------------------------------------
# Detection tables.  Each entry: (tag, compiled_regex, directive_sentence).
# The directive is the instruction handed to the model; the tag is the stable
# identifier the template layer switches on.
# --------------------------------------------------------------------------

def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# ---- Output formats -------------------------------------------------------
FORMAT_RULES: list[tuple[str, re.Pattern, str]] = [
    ("bullets", _rx(
        r"\b(?:bullet points?|as bullets|bulleted|point[- ]?wise|in points?|"
        r"as points|in (?:a )?(?:numbered|bulleted) list|numbered (?:list|points)|"
        r"list (?:it|them) out|"
        r"(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|a few|"
        r"couple(?:\s+of)?|several)(?:\s*[-–to]{1,2}\s*\d{1,2})?\s+"
        r"(?:points?|bullets?))\b"),
     "Format the core answer as concise bullet points rather than paragraphs."),
    ("comparison", _rx(
        r"\b(?:compare|comparison|comparison table|comparison matrix|"
        r"versus|vs\.?|difference between|pros and cons|trade[- ]?offs?)\b"),
     "Present the comparison as a Markdown table — one row per aspect, one "
     "column per item compared — then one line on the difference that matters. "
     "Cite each row."),
    ("table", _rx(r"\b(?:as a table|in a table|tabular|table format|"
                  r"in table form)\b"),
     "Present the key content as a Markdown table."),
    ("checklist", _rx(r"\b(?:checklist|check[- ]?list|to[- ]?do list|"
                      r"action items|as (?:a )?tasks?)\b"),
     "Format the answer as an actionable checklist using `- [ ]` items, each "
     "grounded and cited where it rests on evidence."),
    ("interview", _rx(r"\b(?:interview (?:notes|questions|prep|preparation|"
                      r"perspective)|q ?& ?a|question and answer|"
                      r"viva|revision notes)\b"),
     "Format as interview notes: crisp question → concise model-answer pairs, "
     "each answer cited where evidence-based."),
    ("json", _rx(r"\b(?:as json|in json|json format|json output|"
                 r"return json)\b"),
     "Return the structured answer as a fenced ```json code block with clear "
     "keys; keep prose minimal and put provenance in the Sources section."),
    ("csv", _rx(r"\b(?:as csv|in csv|csv format|comma[- ]separated)\b"),
     "Return the tabular data as a fenced ```csv code block with a header row."),
    ("timeline", _rx(r"\b(?:timeline|chronolog(?:y|ical)|history of|"
                     r"sequence of events|over time)\b"),
     "Organize the answer as a chronological timeline, each entry dated where "
     "the evidence gives a date, and cited."),
    ("flowchart", _rx(r"\b(?:flow ?chart|flow diagram|as a diagram|"
                      r"process (?:flow|diagram)|mermaid)\b"),
     "Include a process flow as a fenced ```mermaid `flowchart TD` block, then "
     "a short prose explanation."),
    ("page_wise", _rx(r"\b(?:page[- ]wise|page by page|per page|"
                      r"page[- ]by[- ]page)\b"),
     "Organize the answer page by page, labelling each page's section and "
     "citing the page it summarizes."),
    ("exec_summary", _rx(r"\b(?:executive summary|exec summary|"
                         r"management summary|bl[uo]f|bottom line up front)\b"),
     "Open with a tight Executive Summary a busy reader can act on alone."),
    ("example", _rx(r"\b(?:with an example|give (?:an|me an) example|"
                    r"real[- ]world (?:example|use case)|use case|"
                    r"concrete example|worked example)\b"),
     "Include at least one concrete, real-world example that illustrates the "
     "answer."),
    ("analogy", _rx(r"\b(?:analogy|analogies|like i'?m explaining to|"
                    r"metaphor|intuitively|intuition behind)\b"),
     "Include a plain-language analogy that builds intuition (analogies are "
     "explanatory and do not need citations)."),
    ("markdown", _rx(r"\b(?:in markdown|as markdown|markdown format|"
                     r"structured (?:report|document))\b"),
     "Use clean Markdown structure with headings, bold key terms, and tables "
     "or lists where they aid clarity."),
]

# ---- Reading level --------------------------------------------------------
READING_LEVEL_RULES: list[tuple[str, re.Pattern, str]] = [
    ("beginner", _rx(r"\b(?:simple (?:words|terms|language)|in simple|eli5|"
                     r"explain like i'?m (?:5|five|a beginner)|layman'?s?|"
                     r"non[- ]technical|for (?:a )?beginners?|plain english)\b"),
     "Write for a BEGINNER: plain language, no unexplained jargon, define any "
     "technical term you must use."),
    ("student", _rx(r"\b(?:for (?:a )?students?|study notes|exam(?:s)? (?:prep|"
                    r"notes|revision)|for (?:my )?(?:class|course|homework))\b"),
     "Write for a STUDENT: teach the concept step by step with clear examples "
     "and the reasoning behind each point."),
    ("manager", _rx(r"\b(?:for (?:a )?managers?|for (?:an )?executives?|"
                    r"business (?:audience|perspective|terms)|"
                    r"for leadership|for (?:a )?non[- ]technical audience)\b"),
     "Write for a MANAGER: lead with impact, decisions and trade-offs; keep "
     "technical depth minimal and business-relevant."),
    ("expert", _rx(r"\b(?:for (?:an )?experts?|expert[- ]level|advanced|"
                   r"deeply technical|for (?:a )?(?:phd|researcher|specialist)|"
                   r"rigorous)\b"),
     "Write for a TECHNICAL EXPERT: assume domain fluency, be precise and "
     "rigorous, and do not over-explain fundamentals."),
    ("engineer", _rx(r"\b(?:for (?:an )?engineers?|technical detail|"
                     r"engineering (?:detail|perspective)|implementation detail)\b"),
     "Write for an ENGINEER: concrete, technically precise, focused on how it "
     "works and how to apply it."),
]

# ---- Response length ------------------------------------------------------
LENGTH_RULES: list[tuple[str, re.Pattern, str]] = [
    ("short", _rx(r"\b(?:short|brief|briefly|concise|quick|tl;?dr|"
                  r"in (?:one|a) (?:line|sentence|paragraph)|one[- ]liner|"
                  r"in a nutshell)\b"),
     "Keep the whole answer SHORT — under roughly 120 words, essentials only."),
    ("research", _rx(r"\b(?:research[- ]level|research report|"
                     r"comprehensive report|deep dive|exhaustive|"
                     r"in[- ]depth report|literature review)\b"),
     "Produce a RESEARCH-LEVEL treatment: thorough, well-structured, with "
     "background, analysis and clearly separated takeaways."),
    ("walkthrough", _rx(r"\b(?:walk[- ]?through|complete walkthrough|"
                        r"comprehensive walkthrough|whole document|"
                        r"entire document|full document|section by section)\b"),
     "Produce a COMPREHENSIVE WALKTHROUGH covering the document(s) end to end "
     "in a logical order."),
    ("detailed", _rx(r"\b(?:detailed|in[- ]depth|comprehensive|thorough|"
                     r"elaborate|step[- ]by[- ]step|(?:in|with|more)\s+"
                     r"(?:more\s+)?detail(?:s)?|in\s+full)\b"),
     "Give a thorough, step-by-step treatment with full detail."),
]

# ---- Personas -------------------------------------------------------------
PERSONA_RULES: list[tuple[str, re.Pattern, str]] = [
    ("teacher", _rx(r"\b(?:as (?:a )?teacher|like (?:a )?teacher|teach me|"
                    r"as (?:a )?tutor|explain like a (?:professor|teacher))\b"),
     "Adopt the TEACHER persona: patient and structured, build understanding "
     "with analogies and examples, check the reader follows each step."),
    ("researcher", _rx(r"\b(?:as (?:a )?researcher|like (?:a )?researcher|"
                       r"academic (?:tone|perspective)|as (?:a )?scientist)\b"),
     "Adopt the RESEARCHER persona: precise, evidence-first, note assumptions "
     "and limitations, distinguish established findings from open questions."),
    ("auditor", _rx(r"\b(?:as (?:an )?auditor|audit perspective|"
                    r"compliance perspective|for (?:an )?audit)\b"),
     "Adopt the AUDITOR persona: verify against evidence, flag gaps and "
     "unsupported claims explicitly, prefer traceable, cited statements."),
    ("manager", _rx(r"\b(?:as (?:a )?manager|like (?:a )?manager|as (?:a )?pm|"
                    r"as (?:a )?project manager|managerial perspective)\b"),
     "Adopt the MANAGER persona: focus on decisions, risk, cost, timeline and "
     "impact; summarize technical detail into what matters for action."),
    ("technician", _rx(r"\b(?:as (?:a )?technician|like (?:a )?technician|"
                       r"as (?:a )?(?:maintenance|field) engineer|"
                       r"hands[- ]on perspective)\b"),
     "Adopt the TECHNICIAN persona: practical and procedural, emphasise steps, "
     "parts, safety and what to do on the equipment."),
    ("interviewer", _rx(r"\b(?:as (?:an )?interviewer|interviewer perspective|"
                        r"interview me|mock interview)\b"),
     "Adopt the INTERVIEWER persona: pose sharp questions and give model "
     "answers, highlighting what a strong response should cover."),
    ("engineer", _rx(r"\b(?:as (?:an )?engineer|like (?:an )?engineer|"
                     r"engineer'?s perspective)\b"),
     "Adopt the ENGINEER persona: pragmatic and precise, focus on how it works "
     "and how to implement or fix it."),
]


@dataclass
class ResponseProfile:
    """The detected adaptive-response preferences for one query."""
    formats: list[str] = field(default_factory=list)
    reading_level: str | None = None
    length: str | None = None
    persona: str | None = None
    directives: list[str] = field(default_factory=list)
    stripped_query: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.formats or self.reading_level
                    or self.length or self.persona)

    def has_format(self, tag: str) -> bool:
        return tag in self.formats

    def as_dict(self) -> dict:
        """Compact, UI-friendly view (drives the response-profile chips)."""
        return {
            "formats": list(self.formats),
            "reading_level": self.reading_level,
            "length": self.length,
            "persona": self.persona,
            "adaptive": not self.is_empty,
        }

    @classmethod
    def from_legacy_directives(cls, directives: list[str] | None) -> "ResponseProfile":
        """Wrap a bare directive list (legacy ``style_directives``) so older
        call paths keep working with the profile-based template layer."""
        directives = list(directives or [])
        formats: list[str] = []
        length = None
        joined = " ".join(directives).lower()
        if "bullet points" in joined:
            formats.append("bullets")
        if "markdown table" in joined:
            formats.append("table")
        if "under roughly 120 words" in joined or "short" in joined:
            length = "short"
        elif "thorough, step-by-step" in joined:
            length = "detailed"
        return cls(formats=formats, length=length, directives=directives)


def _scan(query: str, rules: list[tuple[str, re.Pattern, str]],
          *, first_only: bool) -> tuple[list[str], list[str]]:
    """Return (matched tags, matched directives) for a rule table.

    ``first_only`` picks a single winner (reading level / length / persona are
    scalar); formats allow several. Rules are ordered most-specific-first so the
    first match wins."""
    tags: list[str] = []
    directives: list[str] = []
    for tag, rx, directive in rules:
        if rx.search(query):
            tags.append(tag)
            directives.append(directive)
            if first_only:
                break
    return tags, directives


def analyze(query: str) -> ResponseProfile:
    """Detect the adaptive-response profile for ``query``.

    Returns an empty profile (no directives, ``stripped_query == query``) for
    empty or over-long inputs, so the default answer format is used unchanged.
    """
    if not query or len(query) > _MAX_SCAN_CHARS:
        return ResponseProfile(stripped_query=query or "")

    fmt_tags, fmt_dirs = _scan(query, FORMAT_RULES, first_only=False)
    lvl_tags, lvl_dirs = _scan(query, READING_LEVEL_RULES, first_only=True)
    len_tags, len_dirs = _scan(query, LENGTH_RULES, first_only=True)
    per_tags, per_dirs = _scan(query, PERSONA_RULES, first_only=True)

    directives = fmt_dirs + lvl_dirs + len_dirs + per_dirs
    profile = ResponseProfile(
        formats=fmt_tags,
        reading_level=lvl_tags[0] if lvl_tags else None,
        length=len_tags[0] if len_tags else None,
        persona=per_tags[0] if per_tags else None,
        directives=directives,
        stripped_query=_strip(query),
    )
    return profile


# Optional lead-in before a style phrase ("give answers in points",
# "please answer as a table", "explain it for a beginner").
_STYLE_LEADIN = (r"[.,;]?\s*(?:please\s+)?(?:(?:give|answer|respond|reply|"
                 r"explain|present|show|keep|make|write|format)\s+(?:me\s+)?"
                 r"(?:the\s+)?(?:answers?|it|this|them)?\s*)?")
_ALL_RULES = FORMAT_RULES + READING_LEVEL_RULES + LENGTH_RULES + PERSONA_RULES

# Scaffolding words that merely FRAME a style request ("in a detailed MANNER",
# "in tabular FORMAT", "in bullet FORM") and carry no retrieval meaning. Left
# orphaned when the style phrase itself is stripped, they leak into retrieval
# and the entity list ("manner" became a search entity), so remove them too.
_STYLE_FILLER = re.compile(
    r"\b(?:in\s+(?:a\s+|an\s+)?)?"
    r"(?:manner|fashion|format|formatted|form|style)\b",
    re.IGNORECASE)
# Connectives left dangling at either end after removals ("... llm and").
_DANGLING = re.compile(
    r"^(?:and|or|in|as|with|for|the|a|an|of|to|me|please)\b[\s,]*"
    r"|[\s,]*\b(?:and|or|in|as|with|for|the|a|an|of|to)\s*$",
    re.IGNORECASE)


def _strip(query: str) -> str:
    """Remove styling phrases from the query so they never reach retrieval."""
    if not query or len(query) > _MAX_SCAN_CHARS:
        return query
    out = query
    for _tag, rx, _d in _ALL_RULES:
        out = re.sub(_STYLE_LEADIN + r"(?:in|as|with|for|like)?\s*" + rx.pattern,
                     " ", out, flags=re.IGNORECASE)
    # Drop orphaned styling scaffolding the phrase match left behind.
    out = _STYLE_FILLER.sub(" ", out)
    out = re.sub(r"\s{2,}", " ", out).strip(" .,;?-")
    # Iteratively trim dangling connectives at the ends ("... and", "in ...").
    prev = None
    while prev != out:
        prev = out
        out = _DANGLING.sub(" ", out)
        out = re.sub(r"\s{2,}", " ", out).strip(" .,;?-")
    # If stripping removed everything, the whole query was a styling request
    # with no subject of its own — keep the original for retrieval.
    return out if out.split() else query


# ---- Backwards-compatible helpers (llm.py delegates to these) -------------

def detect_style_directives(query: str) -> list[str]:
    """Legacy shim: the flat list of directive sentences for ``query``."""
    return analyze(query).directives


def strip_style_phrases(query: str) -> str:
    """Legacy shim: ``query`` with styling phrases removed."""
    return _strip(query)

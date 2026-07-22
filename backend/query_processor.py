"""
AXON Query Processor

This module sits between the Planner and Retrieval.

Responsibilities
----------------
1. Detect query intent
2. Rewrite the query into a retrieval-friendly form
3. Expand domain terminology
4. Extract entities
5. Generate retrieval keywords

This module NEVER retrieves documents.
"""

from __future__ import annotations

from dataclasses import dataclass , field
import re


@dataclass
class ProcessedQuery:

    original: str

    rewritten: str

    expanded: str

    intent: str

    entities: list[str]

    keywords: list[str]


@dataclass
class QueryPlan:
    """
    Complete query understanding used by retrieval.
    """

    original_query: str

    rewritten_query: str

    expanded_query: str

    intent: str

    entities: list[str]

    keywords: list[str]

    retrieval_queries: list[str]
    comparison_aspects: list[str] = field(
    default_factory=list
)

    # NEW

    boost_sections: list[str] = field(default_factory=list)

    boost_entities: list[str] = field(default_factory=list)

    boost_keywords: list[str] = field(default_factory=list)

    preferred_documents: list[str] = field(default_factory=list)

class QueryProcessor:
    COMMON_TERM_CORRECTIONS = {
    "langraph": "LangGraph",
    "langgraph": "LangGraph",
    "langchain": "LangChain",
    "lcel": "LCEL",
}

    STOPWORDS = {
        "the","a","an","of","to","for","is","are","was","were",
        "what","who","why","how","does","do","did",
        "explain","describe","tell","about","please",
        "in","on","at","with","using","and","or",
        # Comparison/definition scaffolding words: these carry the user's
        # INTENT (already captured by detect_intent) but are not entities
        # or retrieval-worthy keywords in their own right. Without this,
        # extract_entities/extract_keywords treated "key", "difference",
        # "between", "compare" etc. as if they were named things to boost
        # in retrieval — diluting the signal for the real entities
        # (e.g. "LangChain", "LangGraph").
        "key","difference","differences","between","compare","comparison",
        "versus","vs","define","definition","meaning","overview","summary",
        "different","differs","differ","from","than","unlike",
        # "explain X in detail(s)" scaffolding — not entities
        "detail","details","depth","detailed",
        # answer-request scaffolding ("give answers in points")
        "give","answer","answers",
        # Counts and generic nouns are NEVER entities. They don't merely add
        # noise — they PENALISE the right chunk, because entity coverage
        # divides by the entity count: "three key difference between
        # LangChain and LangGraph" gave a stray LCEL code sample coverage
        # 1.00 (it contains "three") while the actual LangGraph definition
        # scored 0.59 and sank to rank 6. Likewise "project" matched
        # LANGCHAIN_PROJECT=my-project boilerplate and buried the HashRing
        # chunk. An IDF filter cannot fix this — measured on this corpus,
        # "project" (4.01) and "three" (3.18) outrank "langgraph" (3.08) and
        # "lcel" (3.00): the corpus is LangChain-heavy, so the REAL entities
        # are common and the junk words are rare. These words still take
        # part in BM25/dense search; they are only barred from the entity
        # list that drives coverage boosting.
        "one","two","three","four","five","six","seven","eight","nine","ten",
        "first","second","third","couple","few","several","many",
        "point","points","project","projects","thing","things",
        # Quantifiers/qualifiers ("explain attention in more details" made
        # "more" an entity, so every chunk scored coverage 0.5 on a term
        # that identifies nothing).
        "more","most","less","much","lot","bit","brief","briefly","simple",
        "words","short","long","full","complete","whole","entire","again",
        # Imperative command verbs that open a request. Recurring bug: an
        # instruction verb becomes an entity and PENALISES the right chunk
        # (entity coverage divides by the entity count). "write three
        # difference between LangChain and LangGraph" made "write" an
        # entity, so Agents/code chunks containing add_node("write", ...)
        # scored coverage 1.00 and buried the LangGraph definition (0.61,
        # rank 6) — which then produced a fabricated "LangChain is a
        # multi-agent library" answer.
        "write","list","give","provide","show","tell","name","mention",
        "describe","outline","summarize","summarise","create","generate",
        "make","state","find","get","need","want","explain",
    }
    INTENT_RULES = {

    "definition": [
        "what is",
        "what are",      # plural: "what are transformers?" was falling to
        "what're",       # "general" and losing definition-section boosting
        "what's",
        "define",
        "meaning",
        "explain",
    ],

    "author": [
        "author",
        "authors",
        "contributor",
        "contributors",
        "who proposed",
        "who wrote",
        "who developed",
    ],

    "comparison": [
        "compare",
        "difference",
        "versus",
        "vs",
    ],

    "methodology": [
        "method",
        "approach",
        "architecture",
        "pipeline",
        "algorithm",
        "workflow",
    ],

    "results": [
        "result",
        "results",
        "performance",
        "accuracy",
        "evaluation",
        "benchmark",
    ],

    "limitations": [
        "limitation",
        "limitations",
        "drawback",
        "weakness",
        "future work",
    ],

    "implementation": [
        "implementation",
        "code",
        "example",
    ],

    "summary": [
        "summary",
        "overview",
    ],

    "citation": [
        "citation",
        "reference",
        "bibliography",
    ],

    "figure": [
        "figure",
        "table",
        "diagram",
        "chart",
    ],

}
    DOMAIN_TERMS = {

        "bitnet": [

            "BitNet",

            "BitNet b1.58",

            "1-bit LLM",

            "ternary weights",

            "quantization",

            "low-bit inference",

        ],

        "transformer": [

            "Transformer",

            "self-attention",

            "encoder",

            "decoder",

            "Attention Is All You Need",

        ],

        "rag": [

            "Retrieval Augmented Generation",

            "GraphRAG",

            "hybrid retrieval",

            "vector search",

            "BM25",

        ],

        "langchain": [

            "LCEL",

            "RunnableSequence",

            "RunnableParallel",

            "RunnableLambda",

        ],

    }
    def normalize_terms(self, query: str) -> str:
        words = query.split()

        return " ".join(
            self.COMMON_TERM_CORRECTIONS.get(
                word.lower().strip(".,?!"),
                word,
            )
            for word in words
        )

    def extract_comparison_aspects(
    self,
    query: str,
) -> list[str]:
        """
        Extract the two sides of a comparison query.

        Examples
        --------
        "difference between LangChain and LangGraph"
            -> ["LangChain", "LangGraph"]

        "LangChain vs LangGraph"
            -> ["LangChain", "LangGraph"]
        """

        query = query.strip()

        patterns = [
            r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
            r"\bcompare\s+(.+?)\s+(?:with|and|to)\s+(.+?)(?:\?|$)",
            r"^(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)",
            r"\bdifference\s+(?:between\s+)?(.+?)\s+and\s+(.+?)(?:\?|$)",
            # "how is X different from Y" / "X differs from Y" — this form
            # previously matched nothing, so comparison queries phrased with
            # "different from" got no aspect-aware retrieval at all.
            r"\bhow\s+(?:is|are|do|does)\s+(.+?)\s+differ(?:ent)?\s+from\s+(.+?)(?:\?|$)",
            r"\b(\S+(?:\s+\S+)?)\s+(?:is|are)?\s*differ(?:s|ent)?\s+from\s+(.+?)(?:\?|$)",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )

            if match:

                left = match.group(1).strip()
                right = match.group(2).strip()

                # Remove common filler words
                left = re.sub(
                    r"^(the|a|an)\s+",
                    "",
                    left,
                    flags=re.IGNORECASE,
                )

                right = re.sub(
                    r"^(the|a|an)\s+",
                    "",
                    right,
                    flags=re.IGNORECASE,
                )

                return [left, right]

        return []

    # -------------------------------------------------------------

    def process(
    self,
    query: str,
) -> QueryPlan:

        # Step 0: canonical terminology
        query = self.normalize_terms(query)

        rewritten = self.rewrite(query)

        # Detect intent on the CLEAN query, not the rewritten one:
        # rewrite() appends helper tokens ("definition overview",
        # "comparison") to the string, so classifying the rewritten form
        # let the rewriter bias its own classifier.
        intent = self.detect_intent(query)

        # Extract actual user-mentioned entities.
        #
        # IMPORTANT: this must run on the clean `query`, not `rewritten`.
        # rewrite() appends retrieval-helper tokens (e.g. "definition
        # overview comparison") to the END of the string for comparison/
        # definition questions. Running entity/aspect extraction on that
        # augmented string lets those helper tokens get swept up as if
        # they were part of the question — e.g. extract_comparison_aspects
        # greedily captures everything up to the end of the string, so
        # "difference between LangChain and LangGraph" (rewritten to
        # "... LangGraph definition overview comparison") was returning
        # "LangGraph definition overview comparison" as the right-hand
        # side instead of just "LangGraph", which then produced garbled,
        # redundant retrieval sub-queries.
        entities = self.extract_entities(query)

        # Detect comparison sides
        comparison_aspects = []

        if intent.lower() == "comparison":
            comparison_aspects = (
                self.extract_comparison_aspects(query)
            )

        # Expand only for retrieval (intent-gated)
        expanded = self.expand(rewritten, intent)

        keywords = self.extract_keywords(expanded)

        # Normal retrieval subqueries
        retrieval_queries = self.decompose(
                rewritten,
                intent,
            )

        # ---------------------------------------------------------
        # Aspect-aware retrieval
        # ---------------------------------------------------------

        if comparison_aspects:

            left, right = comparison_aspects

            aspect_queries = [
                left,
                right,
                f"{left} definition purpose architecture",
                f"{right} definition purpose architecture",
                f"{left} {right} comparison difference",
            ]

            # Add without duplicates
            existing = {
                q.lower()
                for q in retrieval_queries
            }

            for aspect_query in aspect_queries:

                if aspect_query.lower() not in existing:

                    retrieval_queries.append(
                        aspect_query
                    )

                    existing.add(
                        aspect_query.lower()
                    )

        hints = self.retrieval_hints(
            intent,
            entities,
            keywords,
        )

        return QueryPlan(
            original_query=query,
            rewritten_query=rewritten,
            expanded_query=expanded,
            intent=intent,
            entities=entities,
            keywords=keywords,
            retrieval_queries=retrieval_queries,
            boost_sections=hints["boost_sections"],
            boost_entities=hints["boost_entities"],
            boost_keywords=hints["boost_keywords"],
            preferred_documents=hints["preferred_documents"],
            comparison_aspects=comparison_aspects,
        )    # -------------------------------------------------------------

    def detect_intent(self, query: str) -> str:
        """
        Detect the user's retrieval intent.

        More specific intents are checked before broader intents
        to prevent queries such as:

            "What is the difference between X and Y?"

        from being incorrectly classified as "definition".
        """

        q = query.lower().strip()

        # =========================================================
        # 1. HIGH-PRIORITY COMPARISON INTENT
        # =========================================================

        comparison_patterns = [
            "difference between",
            "differences between",
            "compare",
            "comparison",
            " versus ",
            " vs ",
            "better than",
            "different from",
            "similarities between",
        ]

        if any(
            pattern in q
            for pattern in comparison_patterns
        ):
            return "comparison"

        # =========================================================
        # 2. HIGH-PRIORITY REASONING INTENT
        # =========================================================

        if any(x in q for x in [
            "why",
            "reason",
            "because",
            "cause",
        ]):
            return "reasoning"

        # =========================================================
        # 3. RESULTS
        # =========================================================

        if any(x in q for x in [
            "result",
            "results",
            "performance",
            "accuracy",
            "evaluation",
            "benchmark",
            "experiment",
        ]):
            return "results"

        # =========================================================
        # 4. METHODOLOGY
        # =========================================================

        if any(x in q for x in [
            "method",
            "approach",
            "pipeline",
            "workflow",
            "architecture",
            "algorithm",
        ]):
            return "methodology"

        # =========================================================
        # 5. LIMITATIONS
        # =========================================================

        if any(x in q for x in [
            "future",
            "limitation",
            "limitations",
            "drawback",
            "weakness",
            "challenge",
        ]):
            return "limitations"

        # =========================================================
        # 6. CONCLUSION
        # =========================================================

        if any(x in q for x in [
            "conclusion",
            "summary",
            "takeaway",
        ]):
            return "conclusion"

        # =========================================================
        # 7. CITATION
        # =========================================================

        if any(x in q for x in [
            "citation",
            "reference",
            "bibliography",
        ]):
            return "citation"

        # =========================================================
        # 8. FIGURE / VISUAL
        # =========================================================

        if any(x in q for x in [
            "figure",
            "table",
            "chart",
            "diagram",
        ]):
            return "figure"

        # =========================================================
        # 9. EQUATION
        # =========================================================

        if any(x in q for x in [
            "equation",
            "formula",
            "proof",
            "theorem",
        ]):
            return "equation"

        # =========================================================
        # 10. IMPLEMENTATION
        # =========================================================

        if any(x in q for x in [
            "code",
            "implementation",
            "example",
            "sample",
        ]):
            return "implementation"

        # =========================================================
        # 11. EXISTING CONFIGURED INTENT RULES
        # =========================================================
        # These run AFTER specific intent checks so a broad rule
        # such as "what is" cannot override "difference between".

        for intent, patterns in self.INTENT_RULES.items():

            for pattern in patterns:

                if pattern in q:
                    return intent

        # =========================================================
        # 12. FALLBACK
        # =========================================================

        return "general"
        # -------------------------------------------------------------

    # API-symbol expansions (RunnableSequence, RunnableParallel, …) only
    # help code/implementation questions; injected into a definition or
    # comparison query they are pure retrieval noise.
    _EXPAND_INTENT_GATE = {
        "langchain": {"implementation", "methodology"},
    }

    def expand(self, query: str, intent: str = "") -> str:

        expanded = [query]

        lower = query.lower()

        for trigger, terms in self.DOMAIN_TERMS.items():

            # Word-boundary match: the substring test made "storage",
            # "leverage" and "paragraph" trigger the "rag" expansion.
            if not re.search(
                r"(?<![a-z0-9])" + re.escape(trigger) + r"(?![a-z0-9])",
                lower,
            ):
                continue

            allowed = self._EXPAND_INTENT_GATE.get(trigger)
            if allowed and intent not in allowed:
                continue

            expanded.extend(terms)

        return " ".join(dict.fromkeys(expanded))

    # -------------------------------------------------------------

    def extract_entities(self, query: str) -> list[str]:

        entities = []

        tokens = re.findall(r"[A-Za-z0-9_.+-]+", query)

        for token in tokens:

            # Sentence punctuation is not part of a name: "attention." from
            # "what is multi-head attention. give answers..." failed every
            # entity match because of the trailing dot. Interior dots
            # (app.py, node.js) are preserved.
            token = token.strip(".,")

            if len(token) < 3:
                continue

            if token.lower() in self.STOPWORDS:
                continue

            entities.append(token)

        return list(dict.fromkeys(entities))

    # -------------------------------------------------------------

    def extract_keywords(self, query: str) -> list[str]:

        words = []

        for word in re.findall(r"[a-zA-Z0-9]+", query.lower()):

            if word in self.STOPWORDS:
                continue

            if len(word) < 3:
                continue

            words.append(word)

        return list(dict.fromkeys(words))
    

    def rewrite(self, query: str) -> str:
        """
        Rewrite a natural-language question into a retrieval-friendly query.

        The goal is NOT to answer the question, but to maximize retrieval quality.
        """

        q = " ".join(query.strip().split())

        lower = q.lower()

        rewrites = []

        # ---------------------------------------------------------
        # Definition Questions
        # ---------------------------------------------------------

        if any(x in lower for x in [
            "what is",
            "what are",
            "what're",
            "what's",
            "define",
            "meaning",
            "explain",
        ]):

            rewrites.append("definition")
            rewrites.append("overview")

        # ---------------------------------------------------------
        # Author Questions
        # ---------------------------------------------------------

        if any(x in lower for x in [
            "author",
            "authors",
            "contributor",
            "contributors",
            "who wrote",
            "who proposed",
        ]):

            rewrites.extend([
                "authors",
                "contributors",
                "paper",
            ])

        # ---------------------------------------------------------
        # Comparison Questions
        # ---------------------------------------------------------

        if any(x in lower for x in [
            "compare",
            "difference",
            "versus",
            "vs",
        ]):

            rewrites.append("comparison")

        # ---------------------------------------------------------
        # Method Questions
        # ---------------------------------------------------------

        if any(x in lower for x in [
            "how",
            "algorithm",
            "method",
            "approach",
            "architecture",
            "pipeline",
        ]):

            rewrites.extend([
                "method",
                "architecture",
                "implementation",
            ])

        # ---------------------------------------------------------
        # Results Questions
        # ---------------------------------------------------------

        if any(x in lower for x in [
            "result",
            "performance",
            "accuracy",
            "evaluation",
            "experiment",
        ]):

            rewrites.extend([
                "results",
                "evaluation",
                "performance",
            ])

        # ---------------------------------------------------------
        # Limitations
        # ---------------------------------------------------------

        if any(x in lower for x in [
            "limitation",
            "drawback",
            "future work",
            "weakness",
        ]):

            rewrites.extend([
                "limitations",
                "future work",
            ])

        # Original query always stays
        rewrites.insert(0, q)

        return " ".join(dict.fromkeys(rewrites))
    



    def decompose(
        self,
        query: str,
        intent: str,
    ) -> list[str]:

        queries = [query]

        lower = query.lower()

        if intent == "comparison":

            parts = re.split(
                r"\b(compare|vs|versus|and)\b",
                lower
            )

            for p in parts:

                p = p.strip()

                if len(p) > 4:

                    queries.append(p)

        if " and " in lower:

            for p in lower.split(" and "):

                if len(p.strip()) > 4:

                    queries.append(p.strip())

        if " because " in lower:

            for p in lower.split(" because "):

                if len(p.strip()) > 4:

                    queries.append(p.strip())

        return list(dict.fromkeys(queries)) 
    



    def retrieval_hints(
    self,
    intent: str,
    entities: list[str],
    keywords: list[str],
):
        """
        Produce retrieval hints for the retriever.
        """

        section_map = {

            "definition": [
                "Abstract",
                "Introduction",
                "Overview",
            ],

            "author": [
                "Authors",
                "Title",
                "Abstract",
            ],

            "comparison": [
                "Results",
                "Discussion",
                "Evaluation",
            ],

            "methodology": [
                "Method",
                "Approach",
                "Architecture",
                "Pipeline",
            ],

            "results": [
                "Results",
                "Evaluation",
                "Experiments",
            ],

            "limitations": [
                "Discussion",
                "Limitations",
                "Future Work",
            ],

            "implementation": [
                "Implementation",
                "Algorithm",
                "Code",
            ],

            "citation": [
                "References",
                "Bibliography",
            ],

        }

        return {

            "boost_sections":
                section_map.get(intent, []),

            "boost_entities":
                entities,

            "boost_keywords":
                keywords,

            "preferred_documents": [],
        }
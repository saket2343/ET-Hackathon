from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter


@dataclass
class ConversationContext:
    """
    Lightweight conversation memory.

    Stores only the information required to improve
    retrieval for follow-up questions.

    No embeddings.
    No LLM.
    No vector database.
    """

    topic: str | None = None

    last_intent: str | None = None

    entities: set[str] = field(default_factory=set)

    concepts: set[str] = field(default_factory=set)

    documents: set[str] = field(default_factory=set)

    citations: list[dict] = field(default_factory=list)
    # ---------------------------------------------------------

    def update(
    self,
    evidence: list[dict],
    intent: str,
    query: str | None = None,
):
        """
        Update the conversation context after every
        successful retrieval.

        `query` is what the USER asked. Without it the topic was simply the
        most frequent entity across the evidence, which is not the same
        thing: asking "explain attention in detail" set the topic to "GPUs"
        (the Attention paper mentions "8 P100 GPUs" often enough to win the
        count), so the next turn's "explain it in simple words" resolved to
        "explain GPUs in simple words" and answered about graphics cards.
        Terms the user actually named now win; evidence frequency only
        ranks among them, and remains the fallback when the query named
        nothing that appears in the evidence.
        """

        self.last_intent = intent
        topic_score = Counter()

        for chunk in evidence:

            # Documents
            self.documents.add(chunk["doc_no"])

            # ---------- Entities ----------
            entities = chunk.get("entities", [])

            self.entities.update(entities)

            for entity in entities:
                topic_score[entity] += 3

            # ---------- Concepts ----------
            concepts = chunk.get("concepts", [])

            self.concepts.update(concepts)

            for concept in concepts:
                topic_score[concept] += 2

            # ---------- Keywords ----------
            keywords = chunk.get("keywords", [])

            for keyword in keywords:
                topic_score[keyword] += 1

        # Choose the topic: prefer candidates the USER named in the query.
        if topic_score:
            in_query = []
            if query:
                import re
                low = query.lower()
                in_query = [
                    (term, score) for term, score in topic_score.items()
                    if len(term) >= 3
                    and re.search(r"(?<![a-z0-9])" + re.escape(term.lower())
                                  + r"(?![a-z0-9])", low)
                ]
            if in_query:
                self.topic = max(in_query, key=lambda x: (x[1], len(x[0])))[0]
            else:
                self.topic = topic_score.most_common(1)[0][0]

    # ---------------------------------------------------------

    # Pronouns that, in a short follow-up, stand for the running topic.
    # "that" is deliberately excluded — it is far more often a relative
    # pronoun ("the paper that introduced ...") than a reference.
    _PRONOUNS = ("it", "its", "they", "them", "these", "this", "those")

    # Follow-up openers that carry no subject of their own.
    _FOLLOWUP_STARTS = (
        "how", "why", "who", "when", "where", "compare", "difference",
        "advantages", "limitations", "examples", "implementation",
    )

    # Only short questions are treated as follow-ups; a long, self-contained
    # question that happens to contain "it" is not asking about the topic.
    _MAX_FOLLOWUP_WORDS = 14

    def rewrite(
        self,
        query: str,
    ) -> str:
        """
        Resolve a follow-up question against the running topic.

        Pronouns are SUBSTITUTED, not prepended: "explain it in simple
        words" becomes "explain attention in simple words". Substitution
        matters because the style phrase is stripped next — prepending left
        "attention explain it in simple words", which strips down to a
        single word and trips the stripper's own safety guard, so the raw
        query reached retrieval unchanged and ranked on the meaningless
        words "explain it in simple words". The old rule also only looked at
        the START of the query, so "explain it ..." was never a follow-up at
        all.
        """
        import re

        if not self.topic:
            return query

        lower = query.lower()

        # Topic already named — nothing to resolve.
        if self.topic.lower() in lower:
            return query

        if len(query.split()) > self._MAX_FOLLOWUP_WORDS:
            return query

        # 1. Substitute the first standalone pronoun with the topic.
        pronoun = r"\b(?:" + "|".join(self._PRONOUNS) + r")\b"
        if re.search(pronoun, lower):
            return re.sub(pronoun, self.topic, query, count=1,
                          flags=re.IGNORECASE)

        # 2. Subject-less follow-up opener — prepend the topic instead.
        if any(lower.startswith(p) for p in self._FOLLOWUP_STARTS):
            return f"{self.topic} {query}"

        return query

    # ---------------------------------------------------------

    def clear(self):
        """
        Reset the conversation context.
        """

        self.topic = None

        self.entities.clear()

        self.concepts.clear()

        self.documents.clear()

        self.citations.clear()
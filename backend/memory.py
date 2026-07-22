"""Conversation memory for long chats.

Three jobs:
1. Detect questions that are ABOUT the conversation itself ("what did we
   discuss", "summarize our chat", "what was my first question") so they can
   be answered from the transcript instead of being refused by the
   grounded-or-refuse gate (the corpus knows nothing about the chat).
2. Curate the history that reaches the LLM on every normal turn: a compact
   summary of older turns + the older turns most relevant to the current
   query (recall) + the most recent turns verbatim. This is what lets a long
   chat "connect back" to something said forty turns ago.
3. Provide a deterministic transcript-based answer when no LLM is available.
"""
from __future__ import annotations

import re
from collections import Counter

# --------------------------------------------------------------- meta queries

# Pure conversation-referential queries. Deliberately conservative: mixed
# queries like "how does this connect to the bearing issue we discussed?"
# should go through normal retrieval (with recall injected), not this path.
_META_PATTERNS = [
    r"\bsummari[sz]e\b.*\b(conversation|chat|discussion|what we|so far)\b",
    r"\b(recap|summary) of\b.*\b(conversation|chat|discussion)\b",
    r"\bwhat (did|have) (we|i|you)\b.*\b(discuss|talk|ask|say|said|cover)\w*\b",
    r"\bwhat (was|were) (my|our|the) (first|last|previous|earlier) (question|query|message)s?\b",
    r"\bwhat did you (say|tell|answer|recommend)\b",
    r"\bhow many questions\b.*\b(ask|asked)\b",
    r"\b(list|show)\b.*\b(questions|topics)\b.*\b(asked|discussed|so far|conversation|chat)\b",
    r"\bwhat (have|did) we (covered|done) so far\b",
    r"\bearlier (question|answer|topic)s?\b",
]
_META_RES = [re.compile(p, re.I) for p in _META_PATTERNS]


def is_meta_query(query: str) -> bool:
    q = " ".join(query.split())
    return any(rx.search(q) for rx in _META_RES)


# ------------------------------------------------------------------ recall

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "it", "its", "this", "that", "what", "which", "how", "why",
    "does", "do", "did", "can", "could", "should", "would", "with", "about",
    "we", "you", "i", "my", "our", "your", "me", "be", "as", "at", "by",
    "from", "have", "has", "had", "will", "there", "their", "they", "them",
    "please", "tell", "explain", "more", "also", "any", "some",
}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9_.-]{1,}", text.lower())
            if w not in _STOP}


def relevant_turns(query: str, history: list, k: int = 2,
                   skip_recent: int = 6) -> list[dict]:
    """Older turns (beyond the recent window) most relevant to this query,
    scored by token overlap. Returns [] when nothing clears a minimal bar."""
    older = history[:-skip_recent] if len(history) > skip_recent else []
    if not older:
        return []
    q = _tokens(query)
    if not q:
        return []
    scored = []
    for turn in older:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        overlap = q & _tokens(content)
        if len(overlap) >= 2:
            scored.append((len(overlap), turn))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:k]]


def _topic_line(text: str, limit: int = 110) -> str:
    line = " ".join(text.split())
    return line[:limit] + ("…" if len(line) > limit else "")


def summarize_older(history: list, skip_recent: int = 6) -> str:
    """Compact deterministic summary of the turns that no longer travel
    verbatim: the questions asked, in order."""
    older = history[:-skip_recent] if len(history) > skip_recent else []
    qs = [t for t in older if t.get("role") == "user" and (t.get("content") or "").strip()]
    if not qs:
        return ""
    lines = [f"{i + 1}. {_topic_line(t['content'])}" for i, t in enumerate(qs)]
    return ("Earlier in this conversation the user asked (oldest first):\n"
            + "\n".join(lines))


def curate(query: str, history: list) -> list[dict]:
    """History as it should reach the LLM: [summary] + [recalled older turns]
    + recent turns verbatim. At most ~10 turns so the prompt stays bounded."""
    history = history or []
    recent = history[-6:]
    out: list[dict] = []
    summary = summarize_older(history)
    if summary:
        out.append({"role": "assistant", "content": summary})
    for t in relevant_turns(query, history):
        role = "User" if t.get("role") == "user" else "AXON"
        out.append({"role": "assistant",
                    "content": f"[Recalled from earlier in this conversation — {role} said] "
                               + _topic_line(t.get("content", ""), 500)})
    out.extend(recent)
    return out[-10:]


# ------------------------------------------------------- transcript answers

def transcript(history: list, max_turns: int = 40) -> str:
    turns = []
    for t in history[-max_turns:]:
        content = (t.get("content") or "").strip()
        if not content:
            continue
        speaker = "User" if t.get("role") == "user" else "AXON"
        turns.append(f"{speaker}: {' '.join(content.split())[:700]}")
    return "\n\n".join(turns)


def conversation_answer(query: str, history: list) -> str:
    """Deterministic answer to a meta query when no LLM is available."""
    user_qs = [t.get("content", "").strip() for t in history
               if t.get("role") == "user" and (t.get("content") or "").strip()]
    if not user_qs:
        return "This conversation has no earlier questions yet — this is the first one."
    lines = [f"{i + 1}. {_topic_line(q, 160)}" for i, q in enumerate(user_qs)]
    tail = ""
    ql = query.lower()
    if "first" in ql:
        tail = f"\n\nYour first question was: “{_topic_line(user_qs[0], 200)}”"
    elif "last" in ql or "previous" in ql:
        tail = f"\n\nYour most recent question was: “{_topic_line(user_qs[-1], 200)}”"
    return ("## Conversation so far\n"
            f"You have asked {len(user_qs)} question(s) in this chat:\n"
            + "\n".join(f"- {l}" for l in lines) + tail)

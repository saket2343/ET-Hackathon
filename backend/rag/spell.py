"""Corpus-grounded spell correction (SymSpell-style, dependency-free).

Design decisions (from the Phase-1 review of this system):
- Validity = membership in the CORPUS vocabulary, not English. A term that
  exists in the indexed documents is never touched, so rare technical terms
  (pydantic, LCEL, qdrant) cannot be "corrected" away.
- Corrections can only map TO corpus terms, ranked by edit distance then
  corpus frequency — the corrector can never invent a word retrieval
  wouldn't find.
- Protected spans (URLs, filenames, code identifiers, quoted strings,
  equipment tags/IDs, ALL-CAPS acronyms, numbers) are never corrected.
- Below the confidence threshold the original token is kept: augmentation
  is upstream's job; destructive replacement requires confidence.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .interfaces import Correction, SpellResult

_WORD = re.compile(r"[A-Za-z]+")

# Spans that must never be corrected. Order matters only for reporting.
_PROTECTED = [
    ("url", re.compile(r"https?://\S+|www\.\S+")),
    ("quoted", re.compile(r"\"[^\"]*\"|'[^']*'|`[^`]*`")),
    ("filename", re.compile(r"\b\S+\.(?:py|md|pdf|json|yaml|yml|txt|csv|js|ts|ipynb)\b")),
    ("id", re.compile(r"\b[A-Z]{1,4}-\d+\b")),                      # SOP-101, P-101
    ("code", re.compile(r"\b\w+(?:_\w+)+\b|\b[a-z]+(?:[A-Z]\w+)+\b"  # snake/camel
                        r"|\b[A-Z][a-z0-9]+(?:[A-Z]\w+)+\b"          # PascalCase
                        r"|\b\w+\.\w+(?:\.\w+)*\(?")),               # dotted.path
    ("acronym", re.compile(r"\b[A-Z]{2,6}s?\b")),
    ("number", re.compile(r"\b\d[\w.\-]*\b")),
]


def _deletes(word: str, distance: int) -> set[str]:
    out = {word}
    frontier = {word}
    for _ in range(distance):
        nxt = set()
        for w in frontier:
            for i in range(len(w)):
                nxt.add(w[:i] + w[i + 1:])
        out |= nxt
        frontier = nxt
    return out


def _damerau(a: str, b: str, cap: int = 3) -> int:
    """Bounded Damerau-Levenshtein distance."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + cost)
        prev2, prev = prev, cur
        if min(prev) > cap:
            return cap + 1
    return prev[len(b)]


class CorpusSpellCorrector:
    """BaseSpellCorrector backed by the corpus vocabulary."""

    def __init__(self, vocabulary: dict[str, int] | Counter, cfg: dict):
        self.cfg = cfg
        self.max_edit = int(cfg.get("max_edit_distance", 2))
        self.long_len = int(cfg.get("long_token_len", 6))
        self.min_len = int(cfg.get("min_token_len", 4))
        self.min_conf = float(cfg.get("min_confidence", 0.6))
        self.user_dict = {k.lower(): v for k, v in
                          (cfg.get("user_dictionary") or {}).items()}
        self._extra_protected = [
            ("user", re.compile(p)) for p in (cfg.get("protected_extra") or [])
        ]
        self.vocab: dict[str, int] = {
            w.lower(): int(f) for w, f in dict(vocabulary).items()
            if _WORD.fullmatch(w)
        }
        self._max_log = math.log(1 + max(self.vocab.values(), default=1))
        # SymSpell-style precomputed deletes: delete-form -> candidate terms.
        self._index: dict[str, list[str]] = defaultdict(list)
        for term in self.vocab:
            d = self.max_edit if len(term) >= self.long_len else 1
            for form in _deletes(term, d):
                self._index[form].append(term)

    # ------------------------------------------------------------ intern

    def _protected_spans(self, text: str) -> list[tuple[int, int, str]]:
        spans = []
        for name, rx in _PROTECTED + self._extra_protected:
            for m in rx.finditer(text):
                spans.append((m.start(), m.end(), name))
        return spans

    def _confidence(self, token: str, term: str, dist: int) -> float:
        """Edit distance dominates (0.75); corpus frequency is a tiebreaker
        (0.25), NOT a gate. With the old 0.6/0.4 split, a distance-1 fix to
        a corpus-rare word ('detials' -> 'details', a single transposition)
        scored below the acceptance threshold purely because 'details' is
        uncommon in the indexed books — vetoing the most reliable class of
        correction. Rarity should only matter when distance ties."""
        freq = self.vocab.get(term, 1)
        freq_part = math.log(1 + freq) / self._max_log
        dist_part = 1.0 - dist / max(len(token), 1)
        return round(0.75 * dist_part + 0.25 * freq_part, 3)

    def _lookup(self, token: str) -> tuple[str, int] | None:
        d = self.max_edit if len(token) >= self.long_len else 1
        cands: set[str] = set()
        for form in _deletes(token, d):
            cands.update(self._index.get(form, ()))
        best: tuple[float, int, str] | None = None  # (adj_dist, -freq, term)
        best_dist = 0
        for term in cands:
            dist = _damerau(token, term, cap=d)
            if dist > d or dist == 0:
                continue
            # First-letter prior: typos rarely mangle the initial letter.
            # Without this, "lagraph" resolves to the high-frequency generic
            # "graph" (distance 2) instead of "langgraph" (also distance 2).
            adjusted = dist + (0.0 if term[0] == token[0] else 0.5)
            key = (adjusted, -self.vocab[term], term)
            if best is None or key < best:
                best = key
                best_dist = dist
        if best is None:
            return None
        return best[2], best_dist

    # ------------------------------------------------------------ public

    def correct(self, text: str) -> SpellResult:
        spans = self._protected_spans(text)
        protected_texts = sorted({text[a:b] for a, b, _ in spans})

        corrections: list[Correction] = []
        out: list[str] = []
        last = 0
        for m in _WORD.finditer(text):
            out.append(text[last:m.start()])
            last = m.end()
            token = m.group()
            lower = token.lower()
            replacement = token

            inside_protected = any(
                a <= m.start() and m.end() <= b for a, b, _ in spans
            )
            if inside_protected:
                pass
            elif lower in self.user_dict:
                replacement = self.user_dict[lower]
            elif lower in self.vocab or len(token) < self.min_len:
                pass                                   # valid or too short
            else:
                hit = self._lookup(lower)
                if hit:
                    term, dist = hit
                    conf = self._confidence(lower, term, dist)
                    if conf >= self.min_conf:
                        replacement = (term.capitalize()
                                       if token[0].isupper() else term)
                        corrections.append(Correction(
                            original=token, corrected=replacement,
                            distance=dist, confidence=conf,
                            reason=(f"edit-distance-{dist}, corpus "
                                    f"freq {self.vocab[term]}"),
                        ))
            out.append(replacement)
        out.append(text[last:])

        corrected = "".join(out)
        return SpellResult(
            original=text,
            corrected=corrected,
            corrections=corrections,
            confidence=round(min(
                (c.confidence for c in corrections), default=1.0), 3),
            protected=protected_texts,
        )

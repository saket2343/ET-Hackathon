"""Persistent conversation store (JSON-file backed).

Adds ChatGPT-style conversation management on top of the existing stateless
/api/ask contract:

- every conversation is persisted (messages, title, pin state, linked docs),
- titles are auto-generated from the first user question,
- conversations can be renamed, deleted, searched, and pinned,
- uploaded documents can be linked to the conversation they came from.

The store is deliberately simple — a single JSON file with an in-memory
mirror — matching the project's zero-heavy-infra philosophy (the interface
is the same one a real DB-backed store would expose).
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path


def _now() -> float:
    return time.time()


def _title_from(text: str, limit: int = 60) -> str:
    """Deterministic title from the first user message."""
    line = " ".join((text or "").split())
    line = re.sub(r"^(please|hey|hi|hello|can you|could you|tell me)\s+", "",
                  line, flags=re.IGNORECASE).strip() or line
    if len(line) > limit:
        cut = line[:limit]
        line = cut[: cut.rfind(" ")] if " " in cut else cut
        line += "…"
    return line[:1].upper() + line[1:] if line else "New conversation"


class ConversationStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    # ------------------------------------------------------------- persist

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)

    # --------------------------------------------------------------- CRUD

    def create(self, title: str = "") -> dict:
        with self._lock:
            cid = uuid.uuid4().hex[:12]
            conv = {"id": cid, "title": title or "New conversation",
                    "created": _now(), "updated": _now(),
                    "pinned": False, "messages": [], "documents": [],
                    "entities": [], "topics": []}
            self._data[cid] = conv
            self._save()
            return conv

    def get(self, cid: str) -> dict | None:
        return self._data.get(cid)

    def list(self, q: str = "") -> list[dict]:
        """Conversation summaries, pinned first, then most-recent."""
        items = []
        needle = (q or "").lower().strip()
        for conv in self._data.values():
            if needle:
                hay = conv["title"].lower() + " " + " ".join(
                    (m.get("content") or "").lower()
                    for m in conv["messages"][-20:])
                if needle not in hay:
                    continue
            items.append({
                "id": conv["id"], "title": conv["title"],
                "created": conv["created"], "updated": conv["updated"],
                "pinned": conv.get("pinned", False),
                "messages": len(conv["messages"]),
                "documents": conv.get("documents", []),
                "preview": next((m.get("content", "")[:120]
                                 for m in reversed(conv["messages"])
                                 if m.get("role") == "user"), ""),
            })
        items.sort(key=lambda c: (not c["pinned"], -c["updated"]))
        return items

    def rename(self, cid: str, title: str) -> dict | None:
        with self._lock:
            conv = self._data.get(cid)
            if conv is None:
                return None
            conv["title"] = (title or "").strip() or conv["title"]
            conv["updated"] = _now()
            self._save()
            return conv

    def set_pinned(self, cid: str, pinned: bool) -> dict | None:
        with self._lock:
            conv = self._data.get(cid)
            if conv is None:
                return None
            conv["pinned"] = bool(pinned)
            conv["updated"] = _now()
            self._save()
            return conv

    def delete(self, cid: str) -> bool:
        with self._lock:
            if cid not in self._data:
                return False
            del self._data[cid]
            self._save()
            return True

    # ------------------------------------------------------------ messages

    def append(self, cid: str, role: str, content: str,
               meta: dict | None = None) -> dict | None:
        """Append a message; auto-title on the first user message; harvest
        discussed entities/topics for long-term memory."""
        with self._lock:
            conv = self._data.get(cid)
            if conv is None:
                return None
            msg = {"role": role, "content": content, "ts": _now()}
            if meta:
                msg["meta"] = meta
            conv["messages"].append(msg)
            conv["updated"] = _now()
            if role == "user" and conv["title"] in ("", "New conversation"):
                conv["title"] = _title_from(content)
            if meta:
                for e in meta.get("entities", [])[:8]:
                    if e not in conv["entities"]:
                        conv["entities"].append(e)
                conv["entities"] = conv["entities"][-40:]
            self._save()
            return conv

    def link_document(self, cid: str, doc_no: str) -> dict | None:
        with self._lock:
            conv = self._data.get(cid)
            if conv is None:
                return None
            if doc_no and doc_no not in conv["documents"]:
                conv["documents"].append(doc_no)
                conv["updated"] = _now()
                self._save()
            return conv

    def history(self, cid: str, max_turns: int = 40) -> list[dict]:
        """History in the exact shape /api/ask already consumes."""
        conv = self._data.get(cid)
        if conv is None:
            return []
        return [{"role": m["role"], "content": m["content"]}
                for m in conv["messages"][-max_turns:]]

"""AXON knowledge graph (MVP).

Ontology-anchored graph built from the ingested corpus. In production this is
Neo4j; here it's an in-process adjacency structure with the same relationship
vocabulary (CONNECTED_TO, GOVERNS, REQUIRES, PERFORMED_ON, CAUSED_BY,
MEASURED_BY, DOCUMENTED_BY).
"""
from __future__ import annotations

import re
from collections import defaultdict

from ingest import Corpus


class KnowledgeGraph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}          # id -> {type, label, ...}
        self.edges: list[dict] = []               # {source, target, rel}
        self._adj: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def add_node(self, node_id: str, node_type: str, label: str, **props):
        if node_id not in self.nodes:
            self.nodes[node_id] = {"id": node_id, "type": node_type, "label": label, **props}

    def add_edge(self, source: str, target: str, rel: str):
        if any(e["source"] == source and e["target"] == target and e["rel"] == rel for e in self.edges):
            return
        self.edges.append({"source": source, "target": target, "rel": rel})
        self._adj[source].append((target, rel))
        self._adj[target].append((source, rel))

    def neighbors(self, node_id: str, rel: str | None = None) -> list[tuple[str, str]]:
        return [(n, r) for n, r in self._adj.get(node_id, []) if rel is None or r == rel]

    def subgraph(self, anchor: str, hops: int = 2) -> tuple[set[str], list[dict]]:
        """Local GraphRAG: expand k hops from the anchor entity."""
        return self.expand({anchor}, hops)

    def expand(self, seeds: set[str], hops: int = 2) -> tuple[set[str], list[dict]]:
        """Local GraphRAG traversal: expand k hops from a set of seed entities/
        concepts, returning the reachable node set and the induced edges. This
        is the graph doing work — entities in, relationally-expanded context out."""
        seen = set(s for s in seeds if s in self.nodes)
        frontier = set(seen)
        for _ in range(hops):
            nxt = set()
            for n in frontier:
                for m, _ in self._adj.get(n, []):
                    if m not in seen:
                        nxt.add(m)
            seen |= nxt
            frontier = nxt
        edges = [e for e in self.edges if e["source"] in seen and e["target"] in seen]
        return seen, edges

    # ------------------------------------------------------------------
    # Reasoning upgrades: weighted expansion, node ranking, path finding
    # ------------------------------------------------------------------

    #: how strongly each relationship type propagates relevance
    REL_WEIGHTS = {
        "CAUSED_BY": 1.0, "REQUIRES": 0.9, "GOVERNS": 0.9,
        "HAS_FAILURE": 0.95, "USED_PART": 0.75, "RECOMMENDED": 0.7,
        "MEASURED_BY": 0.8, "CONNECTED_TO": 0.7, "PERFORMED_ON": 0.8,
        "DOCUMENTED_BY": 0.6, "MENTIONS": 0.5, "RELATED_TO": 0.4,
    }

    def weighted_expand(self, seeds: set[str], hops: int = 2,
                        limit: int = 60) -> tuple[dict[str, float], list[dict]]:
        """Relevance-weighted traversal: seeds start at 1.0 and relevance
        decays through relationship weights, so causally-linked nodes outrank
        loosely-associated ones. Returns {node: score} and induced edges."""
        scores: dict[str, float] = {s: 1.0 for s in seeds if s in self.nodes}
        frontier = dict(scores)
        for _ in range(hops):
            nxt: dict[str, float] = {}
            for n, s in frontier.items():
                for m, rel in self._adj.get(n, []):
                    w = s * self.REL_WEIGHTS.get(rel, 0.5) * 0.85
                    if w > scores.get(m, 0.0):
                        nxt[m] = max(nxt.get(m, 0.0), w)
            for m, w in nxt.items():
                if w > scores.get(m, 0.0):
                    scores[m] = w
            frontier = nxt
            if not frontier:
                break
        top = dict(sorted(scores.items(), key=lambda x: -x[1])[:limit])
        edges = [e for e in self.edges
                 if e["source"] in top and e["target"] in top]
        return top, edges

    def rank_nodes(self, iterations: int = 20, damping: float = 0.85) -> dict[str, float]:
        """Degree-seeded PageRank over the graph — used to rank nodes for
        visualization sizing and evidence prioritization."""
        n = len(self.nodes) or 1
        rank = {k: 1.0 / n for k in self.nodes}
        for _ in range(iterations):
            nxt = {k: (1 - damping) / n for k in self.nodes}
            for node, r in rank.items():
                nbrs = self._adj.get(node, [])
                if not nbrs:
                    continue
                share = damping * r / len(nbrs)
                for m, _ in nbrs:
                    if m in nxt:
                        nxt[m] += share
            rank = nxt
        return rank

    def shortest_path(self, a: str, b: str, max_hops: int = 5) -> list[dict]:
        """BFS shortest path between two nodes — evidence for 'how are X and
        Y related' reasoning. Returns the edge sequence, or []."""
        if a not in self.nodes or b not in self.nodes:
            return []
        prev: dict[str, tuple[str, str]] = {}
        frontier, seen = [a], {a}
        for _ in range(max_hops):
            nxt = []
            for n in frontier:
                for m, rel in self._adj.get(n, []):
                    if m in seen:
                        continue
                    seen.add(m)
                    prev[m] = (n, rel)
                    if m == b:
                        path = []
                        cur = b
                        while cur != a:
                            p, rel2 = prev[cur]
                            path.append({"source": p, "target": cur, "rel": rel2})
                            cur = p
                        return list(reversed(path))
                    nxt.append(m)
            frontier = nxt
            if not frontier:
                break
        return []


def build_graph(corpus: Corpus) -> KnowledgeGraph:
    g = KnowledgeGraph()

    # P&ID topology -> Equipment nodes + CONNECTED_TO edges (the vision "wow")
    for sym in corpus.pid["symbols"]:
        g.add_node(sym["tag"], sym["type"], f'{sym["tag"]} {sym["label"]}', bbox=sym["bbox"],
                   drawing=corpus.pid["drawing"])
    for conn in corpus.pid["connections"]:
        rel = "MEASURED_BY" if corpus.pid_symbol_type(conn["from"]) == "Instrument" else "CONNECTED_TO"
        g.add_edge(conn["from"], conn["to"], rel)

    # Asset360's generated register can include equipment that is not yet on
    # the P&ID. Merge that data into the live graph without replacing it.
    for asset_id, meta in getattr(corpus, "assets", {}).items():
        asset_type = meta.get("equipment_type") or meta.get("asset_type") or "Equipment"
        label = meta.get("equipment_name") or f"{asset_id} {asset_type}"
        if asset_id not in g.nodes:
            g.add_node(asset_id, asset_type, label)
        g.nodes[asset_id].update({
            "manufacturer": meta.get("manufacturer", ""),
            "model_number": meta.get("model_number", ""),
            "serial_number": meta.get("serial_number", ""),
            "health_score": meta.get("health_score", ""),
            "status": meta.get("status", ""),
            "location": meta.get("location", ""),
            "last_modified": meta.get("last_modified", ""),
            "asset_folder": f"assets/{asset_id}",
        })

    # Documents -> GOVERNS / REQUIRES edges
    for doc_no, meta in corpus.docs.items():
        g.add_node(doc_no, meta.get("type", "Document"), f'{doc_no} {meta.get("title", "")}',
                   revision=meta.get("revision"))
        for tag in meta.get("governs", []):
            if tag in g.nodes:
                g.add_edge(doc_no, tag, "GOVERNS")
        permit = meta.get("requires_permit")
        if permit:
            g.add_node(f"PERMIT-{permit}", "Permit", f"{permit} Permit")
            g.add_edge(doc_no, f"PERMIT-{permit}", "REQUIRES")
        # uploaded docs: link to every known asset they mention (entity linking)
        for tag in meta.get("mentions", []):
            if tag in g.nodes and g.nodes[tag]["type"] not in ("SOP", "Manual", "Uploaded"):
                g.add_edge(tag, doc_no, "DOCUMENTED_BY")

    # Uploaded documents -> Concept nodes + MENTIONS + co-occurrence RELATED_TO.
    # This is what makes uploading ANY PDF build a knowledge graph, not just a
    # bag of chunks. ENTITY EXTRACT happened in ingest; here we do RELATION
    # DISCOVER (concept co-occurrence within the document) and KG MERGE.
    from collections import Counter
    for doc_no, meta in corpus.docs.items():
        concepts = meta.get("concepts") or []
        if not concepts:
            continue
        cset = set(concepts)
        for c in concepts:
            if c in g.nodes:                              # concept shared across docs
                sd = g.nodes[c].setdefault("source_docs", [g.nodes[c].get("source_doc")])
                if doc_no not in sd:
                    sd.append(doc_no)
            else:
                g.add_node(c, "Concept", c, source_doc=doc_no, source_docs=[doc_no])
            g.add_edge(doc_no, c, "MENTIONS")
        co: Counter = Counter()
        for ch in corpus.chunks:
            if ch.doc_no != doc_no:
                continue
            present = sorted({e for e in ch.entities if e in cset})
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    co[(present[i], present[j])] += 1
        # keep the strongest concept relationships, capped so the graph stays legible
        for (a, b), n in sorted(co.items(), key=lambda x: -x[1])[:15]:
            if n >= 2:
                g.add_edge(a, b, "RELATED_TO")

    # Work orders -> PERFORMED_ON / CAUSED_BY
    for wo in corpus.maintenance:
        g.add_node(wo["wo_number"], "WorkOrder",
                   f'{wo["wo_number"]} {wo["failure_mode"]} ({wo["date"]})',
                   date=wo["date"], failure_mode=wo["failure_mode"], cause=wo["cause"])
        if wo["equipment"] in g.nodes:
            g.add_edge(wo["wo_number"], wo["equipment"], "PERFORMED_ON")
        if wo["failure_mode"] == "bearing failure":
            g.add_node("FM-BEARING-WEAR", "FailureMode", "Bearing wear / failure")
            g.add_edge(wo["wo_number"], "FM-BEARING-WEAR", "CAUSED_BY")

    # Richer maintenance semantics (design doc): asset HAS_FAILURE mode,
    # failure CAUSED_BY root cause, repair USED_PART, inspection RECOMMENDED
    # action. Derived generically from every maintenance row (seeded + events
    # extracted from uploaded documents), so an uploaded report immediately
    # extends the graph. All calls are idempotent (add_node/add_edge dedupe).
    _enrich_maintenance(g, corpus.maintenance)

    return g


_NONE = {"", "none", "n/a", "na", "-", "scheduled pm"}


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", (text or "").strip()).strip("-").upper()


def _parts_of(value: str) -> list[str]:
    parts = []
    for raw in re.split(r"[,;]", value or ""):
        base = re.split(r"\s*x\s*\d+\s*$", raw.strip(), flags=re.I)[0].strip()
        if base and base.lower() not in _NONE:
            parts.append(base)
    return parts


def _enrich_maintenance(g: "KnowledgeGraph", maintenance: list[dict]) -> None:
    for wo in maintenance:
        wo_no = wo.get("wo_number", "")
        equip = wo.get("equipment", "")
        # An uploaded event may reference an asset not on the seeded P&ID.
        if equip and equip not in g.nodes:
            g.add_node(equip, "Equipment", equip)

        failure = (wo.get("failure_mode") or "").strip()
        fm_id = None
        if failure.lower() not in _NONE:
            fm_id = f"FM:{_slug(failure)}"
            g.add_node(fm_id, "FailureMode", failure.title())
            if equip in g.nodes:
                g.add_edge(equip, fm_id, "HAS_FAILURE")
            if wo_no in g.nodes:
                g.add_edge(wo_no, fm_id, "CAUSED_BY")

        cause = (wo.get("root_cause") or wo.get("cause") or "").strip()
        if cause.lower() not in _NONE and fm_id:
            cu_id = f"CAUSE:{_slug(cause)[:40]}"
            g.add_node(cu_id, "Cause", cause[:60])
            g.add_edge(fm_id, cu_id, "CAUSED_BY")

        for part in _parts_of(wo.get("parts_used", "")):
            g.add_node(part.upper(), "Part", part.upper())
            if wo_no in g.nodes:
                g.add_edge(wo_no, part.upper(), "USED_PART")

        recommend = (wo.get("preventive_action") or "").strip()
        if recommend.lower() not in _NONE and wo_no in g.nodes:
            rc_id = f"REC:{_slug(recommend)[:40]}"
            g.add_node(rc_id, "Recommendation", recommend[:60])
            g.add_edge(wo_no, rc_id, "RECOMMENDED")


# Small helper attached to Corpus for readability in build_graph
def _pid_symbol_type(self, tag: str) -> str:
    for s in self.pid["symbols"]:
        if s["tag"] == tag:
            return s["type"]
    return "Unknown"


Corpus.pid_symbol_type = _pid_symbol_type

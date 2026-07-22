"""AXON MVP server — FastAPI app serving the API and the single-page UI."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import re

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import llm
import predictive
from agents import AgentSystem
from conversations import ConversationStore
from ingest import (DATA_DIR, UPLOADS_DIR, load_corpus,
                    supported_upload_extensions)
from kg import build_graph
from knowledge_gaps import GapStore, load_experts, suggest_sme
from maintenance_service import MaintenanceService
from asset_history import ASSET_TYPES, AssetHistoryGenerator, detect_assets
from retrieval import HybridIndex

app = FastAPI(title="AXON — Industrial Knowledge OS (MVP)")

# Local demo: allow the UI to call the API from file:// or another origin.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

corpus = load_corpus()
graph = build_graph(corpus)
index = HybridIndex(corpus.chunks)
system = AgentSystem(corpus, graph, index)
gaps = GapStore(DATA_DIR / "knowledge_gaps.json")
experts = load_experts(DATA_DIR / "experts.csv")
conversations = ConversationStore(DATA_DIR / "conversations.json")
maintenance = MaintenanceService(DATA_DIR)
asset_history = AssetHistoryGenerator(DATA_DIR)


def _reingest():
    """Rebuild corpus, graph, index and agents after an upload."""
    global corpus, graph, index, system
    corpus = load_corpus()
    graph = build_graph(corpus)
    index = HybridIndex(corpus.chunks)
    system = AgentSystem(corpus, graph, index)


def _norm_doc_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower().replace(".pdf", ""))


def _resolve_document(doc_no: str) -> tuple[str, dict] | tuple[None, None]:
    """Resolve a document from API path input or UI display identifiers."""
    meta = corpus.docs.get(doc_no)
    if meta is not None:
        return doc_no, meta
    wanted = _norm_doc_id(doc_no)
    for candidate, candidate_meta in corpus.docs.items():
        aliases = {
            candidate,
            candidate_meta.get("title", ""),
            candidate_meta.get("source_file", ""),
            Path(candidate_meta.get("source_file", "")).stem,
        }
        if any(_norm_doc_id(alias) == wanted for alias in aliases if alias):
            return candidate, candidate_meta
    return None, None


FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


class AskRequest(BaseModel):
    query: str
    history: list = []
    conversation_id: Optional[str] = None


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    pinned: Optional[bool] = None


@app.get("/")
def home():
    return FileResponse(FRONTEND / "index.html")


@app.get("/app.js")
def frontend_js():
    return FileResponse(FRONTEND / "app.js", media_type="application/javascript")


@app.get("/api/status")
def status():
    concepts = sum(1 for n in graph.nodes.values() if n.get("type") == "Concept")
    entities = sum(1 for c in corpus.chunks for _ in c.entities)
    return {
        "documents": len(corpus.docs),
        "chunks": len(corpus.chunks),
        "graph_nodes": len(graph.nodes),
        "graph_edges": len(graph.edges),
        "concepts": concepts,
        "relations": len(graph.edges),
        "entities": entities,
        "sensor_points": len(corpus.sensors),
        "llm": llm.MODEL if llm.llm_available() else "deterministic fallback (no credentials)",
    }


@app.get("/api/graph")
def get_graph():
    """Full graph with PageRank scores (for node sizing) and degree counts."""
    ranks = graph.rank_nodes()
    max_rank = max(ranks.values()) if ranks else 1.0
    nodes = []
    for n in graph.nodes.values():
        node = dict(n)
        node["rank"] = round(ranks.get(n["id"], 0.0) / max_rank, 4)
        node["degree"] = len(graph.neighbors(n["id"]))
        nodes.append(node)
    return {"nodes": nodes, "edges": graph.edges}


@app.get("/api/graph/path")
def graph_path(source: str, target: str):
    """Shortest relationship path between two nodes (graph reasoning)."""
    path = graph.shortest_path(source, target)
    return {"source": source, "target": target,
            "found": bool(path), "path": path}


@app.get("/api/graph/neighborhood/{node_id}")
def graph_neighborhood(node_id: str, hops: int = 2):
    """Relevance-weighted neighborhood around a node (for the details drawer)."""
    if node_id not in graph.nodes:
        raise HTTPException(404, f"No such node: {node_id}")
    scores, edges = graph.weighted_expand({node_id}, hops=hops)
    nodes = [dict(graph.nodes[n], relevance=round(s, 3))
             for n, s in scores.items()]
    nodes.sort(key=lambda n: -n["relevance"])
    return {"anchor": node_id, "nodes": nodes, "edges": edges}


@app.get("/api/telemetry")
def telemetry():
    return {
        "series": corpus.sensors[-14 * 24:],
        "prediction": predictive.analyze(corpus.sensors),
    }


@app.get("/api/pid")
def pid():
    return corpus.pid


# --------------------------------------------------------------- Asset360

@app.get("/api/assets")
def assets():
    """Every physical asset in the knowledge graph, with a live health
    verdict. Drives the Asset360 asset picker."""
    out = []
    for node_id, data in graph.nodes.items():
        if data.get("type") not in ASSET_TYPES:
            continue
        wos = [w for w in corpus.maintenance if w["equipment"] == node_id]
        health = "unknown"
        if node_id == predictive.ANCHOR_ASSET:
            pred = predictive.analyze(corpus.sensors)
            health = ("danger" if pred.get("latest_vibration", 0)
                      >= pred.get("danger_limit", 99) else
                      "alert" if pred.get("in_alert_zone") else "healthy")
        else:
            try:
                score = float(data.get("health_score", ""))
                health = "healthy" if score >= 80 else "alert" if score >= 50 else "danger"
            except (TypeError, ValueError):
                pass
        out.append({
            "id": node_id,
            "type": data.get("type"),
            "label": data.get("label", node_id),
            "work_orders": len(wos),
            "monitored": node_id == predictive.ANCHOR_ASSET,
            "health": health,
        })
    out.sort(key=lambda a: (not a["monitored"], a["id"]))
    return {"assets": out}


@app.get("/api/asset/{asset_id}")
def asset360(asset_id: str):
    """Complete digital profile of one asset: identity, live condition and
    prediction, maintenance history, spares, documents, connected
    equipment, instruments and failure modes.

    Every field is assembled from data the system ALREADY holds — the
    knowledge graph, the maintenance log, the spares list, the sensor
    series and the ingested documents. Nothing here is generated; the
    dashboard is a view over the system of record.
    """
    node = graph.nodes.get(asset_id)
    if node is None or node.get("type") not in ASSET_TYPES:
        raise HTTPException(404, f"Unknown asset: {asset_id}")

    # --- graph neighbourhood, split by relationship -------------------
    connected, measured_by, documented_by, failure_modes, other = (
        [], [], [], [], [])
    for target, rel in graph.neighbors(asset_id):
        entry = {"id": target, "rel": rel,
                 "type": graph.nodes.get(target, {}).get("type", "?"),
                 "label": graph.nodes.get(target, {}).get("label", target)}
        if rel == "CONNECTED_TO":
            connected.append(entry)
        elif rel == "MEASURED_BY":
            measured_by.append(entry)
        elif rel == "DOCUMENTED_BY":
            documented_by.append(entry)
        elif rel in ("HAS_FAILURE_MODE", "FAILS_AS"):
            failure_modes.append(entry)
        else:
            other.append(entry)

    # --- maintenance history ------------------------------------------
    history = sorted(
        (w for w in corpus.maintenance if w["equipment"] == asset_id),
        key=lambda w: w["date"], reverse=True)
    downtime = sum(float(w.get("downtime_hours") or 0) for w in history)
    modes: dict[str, int] = {}
    for w in history:
        if w.get("failure_mode"):
            modes[w["failure_mode"]] = modes.get(w["failure_mode"], 0) + 1

    # --- spares referenced by this asset's history --------------------
    part_nos = {p.strip().split(" x")[0]
                for w in history for p in (w.get("parts_used") or "").split(",")
                if p.strip()}
    spares = [s for s in corpus.spares if s["part_number"] in part_nos
              or asset_id in s.get("description", "")]
    for s in spares:
        s = s.setdefault("low_stock",
                         int(s.get("qty_on_hand", 0)) <= int(s.get("min_stock", 0)))

    # --- documents that mention this asset ----------------------------
    doc_hits: dict[str, int] = {}
    for c in corpus.chunks:
        if asset_id in c.entities:
            doc_hits[c.doc_no] = doc_hits.get(c.doc_no, 0) + 1
    documents = [{"doc_no": d, "title": corpus.docs.get(d, {}).get("title", d),
                  "chunks": n} for d, n in
                 sorted(doc_hits.items(), key=lambda x: -x[1])]

    # --- live condition + prediction (only for the monitored asset) ----
    condition = None
    if asset_id == predictive.ANCHOR_ASSET:
        pred = predictive.analyze(corpus.sensors)
        condition = {
            "prediction": pred,
            "series": corpus.sensors[-14 * 24:],
        }

    return {
        "id": asset_id,
        "type": node.get("type"),
        "label": node.get("label", asset_id),
        "properties": {k: v for k, v in node.items()
                       if k not in ("type", "label")},
        "condition": condition,
        "maintenance": {
            "work_orders": history,
            "count": len(history),
            "total_downtime_hours": round(downtime, 1),
            "failure_modes": [{"mode": m, "count": n}
                              for m, n in sorted(modes.items(),
                                                 key=lambda x: -x[1])],
        },
        "spares": spares,
        "documents": documents,
        "graph": {
            "connected_to": connected,
            "measured_by": measured_by,
            "documented_by": documented_by,
            "failure_modes": failure_modes,
            "other": other,
        },
        "pid": {"drawing": corpus.pid.get("drawing"),
                "revision": corpus.pid.get("revision")},
    }


@app.get("/api/documents")
def documents():
    """List every ingested document with its chunk/concept counts. Uploaded
    documents are deletable; seeded plant documents are not."""
    chunk_counts: dict = {}
    for c in corpus.chunks:
        chunk_counts[c.doc_no] = chunk_counts.get(c.doc_no, 0) + 1
    concept_counts: dict = {}
    for n in graph.nodes.values():
        if n.get("type") == "Concept" and n.get("source_doc"):
            concept_counts[n["source_doc"]] = concept_counts.get(n["source_doc"], 0) + 1
    docs = []
    for doc_no, meta in corpus.docs.items():
        docs.append({
            "doc_no": doc_no,
            "title": meta.get("title", doc_no),
            "type": meta.get("type", "Document"),
            "chunks": chunk_counts.get(doc_no, 0),
            "concepts": concept_counts.get(doc_no, 0),
            "uploaded": bool(meta.get("uploaded")),
        })
    docs.sort(key=lambda d: (not d["uploaded"], d["doc_no"]))  # uploaded first
    return {"count": len(docs),
            "uploaded": sum(1 for d in docs if d["uploaded"]),
            "documents": docs}


@app.delete("/api/documents/{doc_no}")
def delete_document(doc_no: str):
    """Delete an uploaded document: remove its file and rebuild the graph so its
    nodes/edges disappear. Seeded plant documents cannot be deleted."""
    resolved_doc_no, meta = _resolve_document(doc_no)
    if meta is None:
        raise HTTPException(404, f"No such document: {doc_no}")
    if not meta.get("uploaded") or not meta.get("source_file"):
        raise HTTPException(403, "Seeded plant documents cannot be deleted")
    target = (UPLOADS_DIR / Path(meta["source_file"]).name)
    if target.exists():
        target.unlink()
    _reingest()
    return {"ok": True, "deleted": resolved_doc_no,
            "documents": len(corpus.docs),
            "graph_nodes": len(graph.nodes), "graph_edges": len(graph.edges)}


@app.post("/api/reset")
def reset_chat():
    """Start a new chat: clear the server-side conversation context."""
    system.context.clear()
    return {"ok": True}


@app.post("/api/ask")
def ask(req: AskRequest):
    # Persistent conversation memory: when a conversation_id is supplied the
    # server owns the history; otherwise fall back to the client-sent history
    # (frozen legacy contract).
    conv = None
    history = req.history
    if req.conversation_id:
        conv = conversations.get(req.conversation_id)
        if conv is None:
            conv = conversations.create()
        history = conversations.history(conv["id"])
    result = system.run_case(req.query, history)
    if conv is not None:
        conversations.append(conv["id"], "user", req.query)
        conversations.append(
            conv["id"], "assistant", result.get("answer", ""),
            meta={"entities": [c.get("doc_no", "")
                               for c in result.get("citations", [])],
                  "confidence": result.get("confidence")})
        result["conversation_id"] = conv["id"]
        result["conversation_title"] = conversations.get(conv["id"])["title"]
    # Knowledge-gap detection: capture questions AXON couldn't answer well.
    verdict = result.get("verdict", "")
    if verdict.startswith("REFUSE"):
        gap = gaps.record(req.query, "No grounded evidence in the knowledge base",
                          result.get("confidence", 0.0), "HIGH",
                          suggest_sme(req.query, experts))
        result["knowledge_gap"] = gap
    elif verdict.startswith("ESCALATE"):
        gap = gaps.record(req.query, "Answer could not be fully verified against sources",
                          result.get("confidence", 0.0), "MEDIUM",
                          suggest_sme(req.query, experts))
        result["knowledge_gap"] = gap
    result["knowledge_gaps_unresolved"] = gaps.unresolved_count()
    return result


@app.post("/api/compare")
def compare(req: AskRequest):
    """Side-by-side: Vanilla RAG vs AXON GraphRAG on the same query."""
    return system.compare(req.query)


# -------------------------------------------------- conversation management

@app.get("/api/conversations")
def list_conversations(q: str = ""):
    """List (optionally search) conversations — pinned first, most recent next."""
    return {"conversations": conversations.list(q)}


@app.post("/api/conversations")
def create_conversation():
    return conversations.create()


@app.get("/api/conversations/{cid}")
def get_conversation(cid: str):
    conv = conversations.get(cid)
    if conv is None:
        raise HTTPException(404, f"No such conversation: {cid}")
    return conv


@app.patch("/api/conversations/{cid}")
def update_conversation(cid: str, req: ConversationUpdate):
    conv = None
    if req.title is not None:
        conv = conversations.rename(cid, req.title)
    if req.pinned is not None:
        conv = conversations.set_pinned(cid, req.pinned)
    if conv is None:
        raise HTTPException(404, f"No such conversation: {cid}")
    return conv


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str):
    if not conversations.delete(cid):
        raise HTTPException(404, f"No such conversation: {cid}")
    return {"ok": True, "deleted": cid}


@app.post("/api/conversations/{cid}/documents/{doc_no}")
def link_conversation_document(cid: str, doc_no: str):
    conv = conversations.link_document(cid, doc_no)
    if conv is None:
        raise HTTPException(404, f"No such conversation: {cid}")
    return {"ok": True, "documents": conv["documents"]}


@app.get("/api/knowledge-gaps")
def knowledge_gaps():
    items = gaps.list()
    return {"unresolved": len(items), "gaps": items}


class GapUpdate(BaseModel):
    owner: Optional[str] = None
    status: Optional[str] = None


@app.post("/api/knowledge-gaps/{gap_id}/update")
def update_gap(gap_id: str, req: GapUpdate):
    g = None
    if req.owner is not None:
        g = gaps.assign(gap_id, req.owner)
    if req.status is not None:
        g = gaps.set_status(gap_id, req.status)
    if g is None:
        raise HTTPException(404, f"No such gap (or invalid status): {gap_id}")
    return {"ok": True, "gap": g, "unresolved": gaps.unresolved_count()}


@app.get("/api/experts")
def list_experts():
    return {"experts": [{"name": e["name"], "role": e["role"], "area": e.get("area", "")}
                        for e in experts]}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "upload.pdf").name  # strip any path components
    allowed = supported_upload_extensions()
    if not name.lower().endswith(allowed):
        raise HTTPException(
            400, f"Unsupported file type. Supported: {', '.join(allowed)}")
    dest = UPLOADS_DIR / name
    UPLOADS_DIR.mkdir(exist_ok=True)
    dest.write_bytes(await file.read())
    try:
        _reingest()
    except Exception as e:
        dest.unlink(missing_ok=True)
        _reingest()
        raise HTTPException(422, f"Could not parse {name}: {e}")
    doc = next((d for d, m in corpus.docs.items() if m.get("title") == name or d == Path(name).stem), None)
    n_chunks = sum(1 for c in corpus.chunks if c.doc_title == name or c.doc_no == (doc or ""))
    concepts = [n for n, d in graph.nodes.items()
                if d.get("type") == "Concept" and d.get("source_doc") == doc]
    # The document is now indexed into RAG (unchanged behaviour). Additionally
    # classify it so the UI can decide whether to OFFER an Asset360 update.
    # Asset360 is never modified here — this only detects and reports.
    analysis = None
    if doc:
        try:
            analysis = maintenance.analyze(corpus, graph, doc)
        except Exception as e:  # classification must never fail an upload
            analysis = {"error": f"{type(e).__name__}: {e}"}
    document_text = "\n".join(c.text for c in corpus.chunks if c.doc_no == doc)
    asset_candidates = detect_assets(document_text) if doc else []
    return {"ok": True, "doc_no": doc, "chunks_indexed": n_chunks,
            "graph_nodes": len(graph.nodes), "graph_edges": len(graph.edges),
            "concepts": concepts,
            "linked_assets": [e["source"] for e in graph.edges
                              if e["rel"] == "DOCUMENTED_BY" and e["target"] == doc],
            "classification": (analysis or {}).get("classification"),
            "maintenance_detected": bool((analysis or {}).get("maintenance_detected")),
            "candidate_assets": (analysis or {}).get("candidate_assets", []),
            "asset_history_detected": bool(asset_candidates),
            "detected_assets": asset_candidates}


# --------------------------------------------------- Asset360 ingestion API

class ExtractRequest(BaseModel):
    doc_no: str
    asset_ids: Optional[list[str]] = None


class CommitRequest(BaseModel):
    events: list[dict]


class AssetHistoryRequest(BaseModel):
    doc_no: str


@app.post("/api/asset-history/generate")
def generate_asset_history(req: AssetHistoryRequest):
    """Create or merge Digital History artifacts after explicit approval.
    The uploaded file has already been indexed, so an artifact-generation
    error can never make the source document unavailable to RAG."""
    if req.doc_no not in corpus.docs:
        raise HTTPException(404, f"No such document: {req.doc_no}")
    try:
        result = asset_history.generate(corpus, graph, req.doc_no, _reingest)
    except Exception as exc:
        return {"ok": False, "asset_ids": [],
                "message": "Unable to generate Asset360 history automatically. "
                           f"The uploaded document is still searchable. ({type(exc).__name__}: {exc})"}
    if result.get("ok"):
        result["graph_nodes"] = len(graph.nodes)
        result["graph_edges"] = len(graph.edges)
    return result


@app.get("/api/uploads/{name}")
def serve_upload(name: str):
    """Serve an uploaded source file (PDF, image, ...) for the Asset360
    source-traceability viewer. Path components are stripped to stay inside
    the uploads directory."""
    safe = Path(name).name
    target = UPLOADS_DIR / safe
    if not target.exists() or not target.is_file():
        raise HTTPException(404, f"No such upload: {safe}")
    return FileResponse(target)


@app.post("/api/maintenance/analyze")
def maintenance_analyze(req: ExtractRequest):
    """Classify a document and report whether it carries maintenance info."""
    if req.doc_no not in corpus.docs:
        raise HTTPException(404, f"No such document: {req.doc_no}")
    return maintenance.analyze(corpus, graph, req.doc_no)


@app.post("/api/maintenance/extract")
def maintenance_extract(req: ExtractRequest):
    """Extract structured maintenance events for the preview dialog. Read-only:
    nothing is written to Asset360."""
    if req.doc_no not in corpus.docs:
        raise HTTPException(404, f"No such document: {req.doc_no}")
    return maintenance.preview(corpus, graph, req.doc_no, req.asset_ids)


@app.post("/api/maintenance/commit")
def maintenance_commit(req: CommitRequest):
    """Validate and apply confirmed events to Asset360 live (no restart)."""
    if not req.events:
        raise HTTPException(400, "No events to commit")
    result = maintenance.commit(corpus, graph, req.events, _reingest)
    if not result.get("ok"):
        return result
    result["graph_nodes"] = len(graph.nodes)
    result["graph_edges"] = len(graph.edges)
    return result


@app.get("/api/maintenance/events")
def maintenance_events():
    return {"events": maintenance.events()}


@app.get("/api/maintenance/event/{event_id}")
def maintenance_event(event_id: str):
    ev = maintenance.get_event(event_id)
    if ev is None:
        raise HTTPException(404, f"No such event: {event_id}")
    _, meta = _resolve_document(ev.get("source_document", ""))
    ev = dict(ev)
    ev["source_file"] = (meta or {}).get("source_file") or ev.get("source_document")
    return ev


if __name__ == "__main__":
    import os
    import uvicorn
    # PORT env var lets the app run alongside another instance (or under a
    # process manager) without editing code; defaults to the documented 8000.
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))

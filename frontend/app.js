/* =========================================================================
   AXON — Enterprise Knowledge Assistant (frontend)
   Vanilla JS, no build step. Talks to the FastAPI backend on the same origin.
   Sections: prefs · layout/panels · auth · conversations · documents ·
             chat/markdown · knowledge graph
   ========================================================================= */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const API = "";

/* ------------------------------------------------------------------ prefs */
const PREFS_KEY = "axon_prefs_v2";
const prefs = Object.assign(
  { railW: 288, graphW: 360, sideOpen: true, graphOpen: true,
    focus: false, layout: "force" },
  JSON.parse(localStorage.getItem(PREFS_KEY) || "{}"));
const savePrefs = () => localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 2600);
}

/* --------------------------------------------------- resizable panels */
const app = $("#app"), sidebar = $("#sidebar"), graphside = $("#graphside");

function applyPanelPrefs() {
  document.documentElement.style.setProperty("--rail-w", prefs.railW + "px");
  document.documentElement.style.setProperty("--graph-w", prefs.graphW + "px");
  sidebar.classList.toggle("collapsed", !prefs.sideOpen);
  graphside.classList.toggle("collapsed", !prefs.graphOpen);
  app.classList.toggle("focus-mode", !!prefs.focus);
  $("#tgl-focus").classList.toggle("on", !!prefs.focus);
  $("#tgl-graph").classList.toggle("on", !!prefs.graphOpen);
}

function bindSplitter(el, opts) {
  el.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    el.classList.add("active");
    el.setPointerCapture(e.pointerId);
    const move = (ev) => {
      const w = opts.width(ev.clientX);
      if (w >= opts.min && w <= opts.max) { opts.apply(w); }
    };
    const up = () => {
      el.classList.remove("active");
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      savePrefs();
      Graph.resize();
    };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
  });
}
bindSplitter($("#split-l"), {
  min: 200, max: 460,
  width: (x) => x,
  apply: (w) => { prefs.railW = w; document.documentElement.style.setProperty("--rail-w", w + "px"); },
});
bindSplitter($("#split-r"), {
  min: 260, max: 640,
  width: (x) => window.innerWidth - x,
  apply: (w) => { prefs.graphW = w; document.documentElement.style.setProperty("--graph-w", w + "px"); },
});

$("#tgl-side").onclick = () => { prefs.sideOpen = !prefs.sideOpen; savePrefs(); applyPanelPrefs(); Graph.resize(); };
$("#tgl-graph").onclick = () => { prefs.graphOpen = !prefs.graphOpen; savePrefs(); applyPanelPrefs(); Graph.resize(); };
$("#tgl-focus").onclick = () => { prefs.focus = !prefs.focus; savePrefs(); applyPanelPrefs(); Graph.resize(); };
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") { e.preventDefault(); $("#tgl-side").click(); }
  if ((e.ctrlKey || e.metaKey) && e.key === ".") { e.preventDefault(); $("#tgl-focus").click(); }
});

/* ---------------------------------------------------------- Google auth */
const AUTH_KEY = "axon_user";
function renderProfile() {
  const u = JSON.parse(localStorage.getItem(AUTH_KEY) || "null");
  const p = $("#profile");
  if (u) {
    p.hidden = false;
    $("#pimg").src = u.picture || "";
    $("#pname").textContent = u.name || "Signed in";
    $("#pemail").textContent = u.email || "";
    $("#gsi-btn").innerHTML = "";
  } else {
    p.hidden = true;
    initGsi();
  }
}
$("#signout").onclick = () => {
  localStorage.removeItem(AUTH_KEY);
  if (window.google?.accounts?.id) google.accounts.id.disableAutoSelect();
  renderProfile();
  toast("Signed out");
};
function initGsi() {
  const clientId = window.GOOGLE_CLIENT_ID
    || document.querySelector('meta[name="google-client-id"]')?.content;
  if (!clientId || !window.google?.accounts?.id) {
    // No client id configured — auth stays optional, app fully usable.
    if (!clientId) $("#gsi-btn").innerHTML = "";
    else setTimeout(initGsi, 400);
    return;
  }
  google.accounts.id.initialize({
    client_id: clientId,
    callback: (resp) => {
      try {
        const payload = JSON.parse(atob(resp.credential.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
        localStorage.setItem(AUTH_KEY, JSON.stringify({
          name: payload.name, email: payload.email, picture: payload.picture }));
        renderProfile();
        toast(`Welcome, ${payload.given_name || payload.name}`);
      } catch { toast("Sign-in failed"); }
    },
  });
  google.accounts.id.renderButton($("#gsi-btn"),
    { theme: "filled_black", size: "medium", shape: "pill", width: 240 });
}

/* ------------------------------------------------------- conversations */
const Conv = {
  list: [], active: null,

  async refresh(q = "") {
    try {
      const r = await fetch(`${API}/api/conversations?q=${encodeURIComponent(q)}`);
      const d = await r.json();
      this.list = d.conversations || [];
    } catch { this.list = []; }
    this.render();
  },

  render() {
    $("#convcount").textContent = this.list.length;
    const box = $("#convlist");
    box.innerHTML = "";
    if (!this.list.length) {
      box.innerHTML = `<div style="color:var(--faint);font-size:12px;padding:6px 8px">No conversations yet</div>`;
      return;
    }
    for (const c of this.list) {
      const el = document.createElement("div");
      el.className = "conv" + (this.active === c.id ? " active" : "");
      el.innerHTML = `
        ${c.pinned ? '<span class="pin-ic">◆</span>' : ""}
        <span class="ct" title="${esc(c.title)}">${esc(c.title)}</span>
        <span class="acts">
          <button data-a="pin" title="${c.pinned ? "Unpin" : "Pin"}">${c.pinned ? "◇" : "◆"}</button>
          <button data-a="ren" title="Rename">✎</button>
          <button data-a="del" title="Delete">✕</button>
        </span>`;
      el.onclick = (e) => {
        const a = e.target.closest("button")?.dataset.a;
        if (a === "pin") return this.pin(c);
        if (a === "ren") return this.renameInline(el, c);
        if (a === "del") return this.remove(c);
        this.open(c.id);
      };
      box.appendChild(el);
    }
  },

  async create() {
    const r = await fetch(`${API}/api/conversations`, { method: "POST" });
    const c = await r.json();
    this.active = c.id;
    Chat.clear();
    await this.refresh($("#convsearch").value);
    $("#convtitle").textContent = c.title;
    $("#q").focus();
  },

  async open(id) {
    this.active = id;
    const r = await fetch(`${API}/api/conversations/${id}`);
    if (!r.ok) return;
    const c = await r.json();
    $("#convtitle").textContent = c.title;
    Chat.clear(false);
    for (const m of c.messages || []) {
      if (m.role === "user") Chat.addUser(m.content);
      else Chat.addAssistant({ answer: m.content, confidence: m.meta?.confidence,
                               citations: [], trace: [] }, { instant: true });
    }
    this.render();
    Chat.scrollEnd();
  },

  async pin(c) {
    await fetch(`${API}/api/conversations/${c.id}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pinned: !c.pinned }) });
    this.refresh($("#convsearch").value);
  },

  renameInline(el, c) {
    const ct = el.querySelector(".ct");
    const input = document.createElement("input");
    input.className = "rn"; input.value = c.title;
    ct.replaceWith(input);
    input.focus(); input.select();
    const done = async (commit) => {
      if (commit && input.value.trim() && input.value.trim() !== c.title) {
        await fetch(`${API}/api/conversations/${c.id}`, {
          method: "PATCH", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: input.value.trim() }) });
        if (this.active === c.id) $("#convtitle").textContent = input.value.trim();
      }
      this.refresh($("#convsearch").value);
    };
    input.onkeydown = (e) => {
      if (e.key === "Enter") done(true);
      if (e.key === "Escape") done(false);
    };
    input.onblur = () => done(true);
  },

  async remove(c) {
    if (!confirm(`Delete conversation "${c.title}"?`)) return;
    await fetch(`${API}/api/conversations/${c.id}`, { method: "DELETE" });
    if (this.active === c.id) { this.active = null; Chat.clear(); $("#convtitle").textContent = "New conversation"; }
    this.refresh($("#convsearch").value);
    toast("Conversation deleted");
  },
};
$("#newchat").onclick = () => Conv.create();
$("#convsearch").oninput = debounce(() => Conv.refresh($("#convsearch").value), 250);

/* ------------------------------------------------------------ documents */
const Docs = {
  async refresh() {
    try {
      const r = await fetch(`${API}/api/documents`);
      const d = await r.json();
      $("#doccount").textContent = d.count;
      const box = $("#doclist");
      box.innerHTML = "";
      for (const doc of d.documents) {
        const el = document.createElement("div");
        el.className = "doc";
        el.innerHTML = `
          <span class="ic">▤</span>
          <span class="b">
            <span class="t">${esc(doc.title)}</span>
            <span class="m">${esc(doc.doc_no)} · ${doc.chunks} chunks · ${doc.concepts} concepts</span>
          </span>
          ${doc.uploaded ? `<button class="del" title="Delete document">✕</button>` : ""}`;
        const del = el.querySelector(".del");
        if (del) del.onclick = async () => {
          if (!confirm(`Delete "${doc.title}" and rebuild the index?`)) return;
          const r2 = await fetch(`${API}/api/documents/${encodeURIComponent(doc.doc_no)}`, { method: "DELETE" });
          if (r2.ok) { toast("Document removed — index rebuilt"); Docs.refresh(); Status.refresh(); Graph.load(); }
          else {
            let detail = "Could not delete document";
            try {
              const err = await r2.json();
              if (err.detail) detail = err.detail;
            } catch {}
            toast(detail);
          }
        };
        box.appendChild(el);
      }
    } catch { /* backend offline */ }
  },

  async upload(file) {
    const zone = $("#upzone");
    zone.classList.add("busy");
    zone.textContent = `Ingesting ${file.name}…`;
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API}/api/upload`, { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "upload failed");
      toast(`Indexed ${file.name}: ${d.chunks_indexed} chunks, ${d.concepts?.length || 0} concepts`);
      if (Conv.active && d.doc_no) {
        fetch(`${API}/api/conversations/${Conv.active}/documents/${encodeURIComponent(d.doc_no)}`, { method: "POST" });
      }
      Docs.refresh(); Status.refresh(); Graph.load();
      // The document is indexed into RAG either way. For engineering assets,
      // ask before creating/merging its Digital History; nothing is generated
      // when the user cancels. Keep the older maintenance-review path as a
      // fallback for documents whose asset tag cannot be resolved.
      if (d.asset_history_detected && d.doc_no) AssetHistory.begin(d);
      else if (d.maintenance_detected && d.doc_no) Maintenance.begin(d);
    } catch (e) {
      toast(`Upload failed: ${e.message}`);
    } finally {
      zone.classList.remove("busy");
      zone.textContent = "⇪ Add document (.pdf / .md / .txt)";
    }
  },
};
$("#upzone").onclick = () => $("#upfile").click();
$("#upzone").onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") $("#upfile").click(); };
$("#upfile").onchange = (e) => { if (e.target.files[0]) Docs.upload(e.target.files[0]); e.target.value = ""; };

/* --------------------------------------------------------------- status */
const Status = {
  async refresh() {
    try {
      const r = await fetch(`${API}/api/status`);
      const d = await r.json();
      $("#status").innerHTML =
        `<b>${d.documents}</b> docs · <b>${d.chunks}</b> chunks · ` +
        `<b>${d.graph_nodes}</b> nodes · <b>${d.graph_edges}</b> edges<br>` +
        `LLM: <b>${esc(String(d.llm).slice(0, 34))}</b>`;
    } catch {
      $("#status").textContent = "backend offline — start: uvicorn main:app";
    }
  },
};

/* --------------------------------------------------------------- chat */
const Chat = {
  history: [],   // legacy fallback when no conversation is active
  busy: false,

  clear(resetServer = true) {
    $("#lane").innerHTML = "";
    this.history = [];
    if (resetServer) fetch(`${API}/api/reset`, { method: "POST" }).catch(() => {});
    this.hello();
  },

  hello() {
    const div = document.createElement("div");
    div.className = "hello";
    div.innerHTML = `
      <h2>Ask AXON anything about your documents</h2>
      <p>Grounded, cited answers over the plant knowledge base and everything you upload.</p>
      <div>
        <span class="hint">Why is P-101 vibrating?</span>
        <span class="hint">What is the attention mechanism?</span>
        <span class="hint">Compare vanilla RAG and GraphRAG</span>
      </div>`;
    $$(".hint", div).forEach((h) => (h.onclick = () => { $("#q").value = h.textContent; Chat.send(); }));
    $("#lane").appendChild(div);
  },

  addUser(text) {
    $(".hello")?.remove();
    const m = document.createElement("div");
    m.className = "msg user";
    const b = document.createElement("div");
    b.className = "bubble-u";
    b.textContent = text;
    m.appendChild(b);
    $("#lane").appendChild(m);
    this.scrollEnd();
  },

  thinking() {
    const m = document.createElement("div");
    m.className = "msg";
    m.innerHTML = `<div class="bubble-a"><span class="thinking">AXON is reasoning <i></i><i></i><i></i></span></div>`;
    $("#lane").appendChild(m);
    this.scrollEnd();
    return m;
  },

  /* Render a completed assistant result. When instant=false the answer body
     is revealed with a streaming typewriter effect. */
  addAssistant(res, { instant = false } = {}) {
    $(".hello")?.remove();
    const m = document.createElement("div");
    m.className = "msg";
    const b = document.createElement("div");
    b.className = "bubble-a";
    m.appendChild(b);
    $("#lane").appendChild(m);

    const { body, followups } = splitFollowups(res.answer || "");
    const finish = () => {
      b.innerHTML = renderMarkdown(body, res.citations || []);
      this.decorate(b, res, followups);
      this.scrollEnd();
    };

    if (instant) { finish(); return; }

    // Simulated streaming: reveal progressively, then swap in rich markdown.
    const plain = body;
    let i = 0;
    const step = Math.max(3, Math.round(plain.length / 220));
    const tick = () => {
      i = Math.min(plain.length, i + step);
      b.innerHTML = `<p style="white-space:pre-wrap;margin:0">${esc(plain.slice(0, i))}</p><span class="caret"></span>`;
      this.scrollEnd();
      if (i < plain.length) requestAnimationFrame(tick);
      else finish();
    };
    tick();
  },

  decorate(bubble, res, followups) {
    const meta = document.createElement("div");
    meta.className = "meta";
    if (typeof res.confidence === "number") {
      const pct = Math.round(res.confidence * 100);
      const cls = res.confidence >= 0.75 ? "hi" : res.confidence >= 0.5 ? "mid" : "lo";
      meta.innerHTML += `<span class="conf ${cls}">● Confidence ${pct}%</span>`;
    }
    if (res.verdict) meta.innerHTML += `<span class="tagchip">${esc(String(res.verdict).split("—")[0].trim())}</span>`;
    if (res.answer_source) meta.innerHTML += `<span class="tagchip">${esc(res.answer_source)}</span>`;

    // Adaptive Response Engine: show what style the answer was adapted to.
    const rp = res.response_profile;
    if (rp && rp.adaptive) {
      const bits = [];
      if (rp.persona) bits.push(["persona", rp.persona]);
      if (rp.reading_level) bits.push(["level", rp.reading_level]);
      if (rp.length) bits.push(["length", rp.length]);
      if (rp.formats && rp.formats.length) bits.push(["format", rp.formats.join(", ")]);
      for (const [k, v] of bits)
        meta.innerHTML += `<span class="tagchip adapt" title="Adaptive Response Engine">${esc(k)}: ${esc(v)}</span>`;
    }

    const srcBtn = document.createElement("button");
    srcBtn.className = "toggle"; srcBtn.textContent = `Sources (${(res.citations || []).length})`;
    const trcBtn = document.createElement("button");
    trcBtn.className = "toggle"; trcBtn.textContent = "Reasoning trace";
    meta.append(srcBtn, trcBtn);
    bubble.appendChild(meta);

    const srcPanel = document.createElement("div");
    srcPanel.className = "panel";
    srcPanel.innerHTML = `<div class="cap">Evidence used</div><div class="srcs">` +
      ((res.citations || []).map((c, i) =>
        `<div class="src"><span class="n">${i + 1}</span><span><b>${esc(c.doc_no || "")}</b> ${esc(c.doc_title || c.title || "")}${c.section ? " · " + esc(c.section) : ""}</span></div>`).join("") ||
        `<div class="src">No document citations for this answer.</div>`) + `</div>`;
    bubble.appendChild(srcPanel);

    const trcPanel = document.createElement("div");
    trcPanel.className = "panel";
    trcPanel.innerHTML = `<div class="cap">Multi-agent reasoning</div>` +
      ((res.trace || []).map((t) =>
        `<div class="trace-step"><span class="st">${esc(t.agent || t.step || "")}</span><span class="sm">${esc(t.summary || t.detail || "")}</span></div>`).join("") ||
        `<div class="trace-step"><span class="sm">No trace available.</span></div>`);
    bubble.appendChild(trcPanel);

    srcBtn.onclick = () => { srcPanel.classList.toggle("open"); srcBtn.classList.toggle("on"); };
    trcBtn.onclick = () => { trcPanel.classList.toggle("open"); trcBtn.classList.toggle("on"); };

    if (res.knowledge_gap) {
      const w = document.createElement("div");
      w.className = "warn-note";
      w.textContent = `Knowledge gap recorded — suggested expert: ${res.knowledge_gap.suggested_sme || "n/a"}`;
      bubble.appendChild(w);
    }

    // Prefer server-provided structured follow-ups (robust to the model
    // omitting the section); fall back to those parsed from the answer text.
    const fups = (res.followups && res.followups.length) ? res.followups : followups;
    if (fups.length) {
      const f = document.createElement("div");
      f.className = "followups";
      for (const q of fups) {
        const chip = document.createElement("button");
        chip.className = "fu"; chip.textContent = q;
        chip.onclick = () => { $("#q").value = q; Chat.send(); };
        f.appendChild(chip);
      }
      bubble.appendChild(f);
    }

    if (res.graph_highlight) Graph.highlight(res.graph_highlight);
  },

  async send() {
    if (this.busy) return;
    const q = $("#q").value.trim();
    if (!q) return;
    this.busy = true;
    $("#send").disabled = true;
    $("#q").value = "";
    autoGrow();
    this.addUser(q);
    const spin = this.thinking();
    try {
      // Lazily create a persistent conversation on the first message so every
      // chat is saved without requiring the New Conversation button first.
      if (!Conv.active) {
        try {
          const rc = await fetch(`${API}/api/conversations`, { method: "POST" });
          if (rc.ok) Conv.active = (await rc.json()).id;
        } catch { /* offline — fall back to in-memory history */ }
      }
      const body = { query: q, history: Conv.active ? [] : this.history };
      if (Conv.active) body.conversation_id = Conv.active;
      const r = await fetch(`${API}/api/ask`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) });
      const res = await r.json();
      spin.remove();
      if (!r.ok) throw new Error(res.detail || "request failed");
      if (res.conversation_id && !Conv.active) Conv.active = res.conversation_id;
      if (res.conversation_title) {
        $("#convtitle").textContent = res.conversation_title;
        Conv.refresh($("#convsearch").value);
      }
      if (!Conv.active) {
        this.history.push({ role: "user", text: q });
        this.history.push({ role: "assistant", text: res.answer || "" });
      }
      this.addAssistant(res);
    } catch (e) {
      spin.remove();
      const m = document.createElement("div");
      m.className = "msg";
      m.innerHTML = `<div class="bubble-a"><div class="warn-note">AXON hit an error: ${esc(e.message)}. Is the backend running?</div></div>`;
      $("#lane").appendChild(m);
    } finally {
      this.busy = false;
      $("#send").disabled = false;
      $("#q").focus();
    }
  },

  scrollEnd() { const t = $("#thread"); t.scrollTop = t.scrollHeight; },
};

$("#send").onclick = () => Chat.send();
$("#q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    if (e.nativeEvent?.isComposing || e.isComposing || e.keyCode === 229) return;
    e.preventDefault();
    Chat.send();
  }
});
function autoGrow() {
  const t = $("#q");
  t.style.height = "auto";
  t.style.height = Math.min(150, t.scrollHeight) + "px";
}
$("#q").addEventListener("input", autoGrow);

/* ----------------------------------------------------- markdown mini-lib */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function splitFollowups(md) {
  const m = md.match(/#{1,4}\s*Suggested Follow-?up Questions\s*\n([\s\S]*)$/i);
  if (!m) return { body: md, followups: [] };
  const followups = m[1].split("\n")
    .map((l) => l.replace(/^\s*[-*\d.]+\s*/, "").trim())
    .filter((l) => l.length > 4).slice(0, 4);
  return { body: md.slice(0, m.index).trim(), followups };
}

function renderMarkdown(md, citations) {
  const lines = md.split("\n");
  let html = "", inUl = false, inOl = false, tableBuf = [];
  const closeLists = () => {
    if (inUl) { html += "</ul>"; inUl = false; }
    if (inOl) { html += "</ol>"; inOl = false; }
  };
  const flushTable = () => {
    if (!tableBuf.length) return;
    const rows = tableBuf.filter((r) => !/^\s*\|?[\s:|-]+\|?\s*$/.test(r));
    html += "<table>";
    rows.forEach((r, i) => {
      const cells = r.split("|").map((c) => c.trim()).filter((c, j, a) => !(c === "" && (j === 0 || j === a.length - 1)));
      html += "<tr>" + cells.map((c) => `<${i === 0 ? "th" : "td"}>${inline(c)}</${i === 0 ? "th" : "td"}>`).join("") + "</tr>";
    });
    html += "</table>";
    tableBuf = [];
  };
  const inline = (s) => {
    let x = esc(s);
    x = x.replace(/`([^`]+)`/g, "<code>$1</code>");
    x = x.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    x = x.replace(/(^|\W)\*([^*]+)\*(?=\W|$)/g, "$1<em>$2</em>");
    // [1] / [1,2] citation chips
    x = x.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, nums) =>
      nums.split(",").map((n) => `<span class="cite-n" title="${esc(citeTitle(citations, +n.trim()))}">${n.trim()}</span>`).join(""));
    return x;
  };
  for (const raw of lines) {
    const l = raw.replace(/\r$/, "");
    if (/^\s*\|.*\|\s*$/.test(l)) { closeLists(); tableBuf.push(l); continue; }
    flushTable();
    const h = l.match(/^(#{1,4})\s+(.*)/);
    if (h) { closeLists(); html += `<h4>${inline(h[2])}</h4>`; continue; }
    const ul = l.match(/^\s*[-*]\s+(.*)/);
    if (ul) { if (inOl) { html += "</ol>"; inOl = false; } if (!inUl) { html += "<ul>"; inUl = true; } html += `<li>${inline(ul[1])}</li>`; continue; }
    const ol = l.match(/^\s*\d+[.)]\s+(.*)/);
    if (ol) { if (inUl) { html += "</ul>"; inUl = false; } if (!inOl) { html += "<ol>"; inOl = true; } html += `<li>${inline(ol[1])}</li>`; continue; }
    if (!l.trim()) { closeLists(); continue; }
    closeLists();
    html += `<p>${inline(l)}</p>`;
  }
  flushTable(); closeLists();
  return html;
}
function citeTitle(citations, n) {
  const c = citations?.[n - 1];
  return c ? `${c.doc_no || ""} ${c.doc_title || c.title || ""}` : `Source ${n}`;
}
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

/* ------------------------------------------------------ knowledge graph */
const TYPE_COLORS = {
  Equipment: "#4E8AD0", Sensor: "#2FA382", Procedure: "#C08627",
  Document: "#A96BC4", Concept: "#C75F72", FailureMode: "#E25A5A",
  Standard: "#8899AA", Person: "#46B584", default: "#5F6D88",
};
const colorFor = (t) => TYPE_COLORS[t] || TYPE_COLORS.default;

const Graph = {
  nodes: [], edges: [], byId: new Map(),
  cam: { x: 0, y: 0, k: 1 },
  hiddenTypes: new Set(),
  hoverId: null, selId: null, highlightSet: null, findMatch: null,
  dragging: null, panning: false,
  canvas: null, ctx: null, mini: null, mctx: null,
  simTicks: 0, raf: null,
  _cache: null,

  async load() {
    try {
      const r = await fetch(`${API}/api/graph`);
      const d = await r.json();
      this.ingest(d);
    } catch { /* offline */ }
  },

  ingest(d) {
    const W = 900, H = 700;
    this.nodes = (d.nodes || []).map((n, i) => ({
      ...n,
      x: W / 2 + Math.cos(i * 2.4) * (120 + (i % 7) * 42),
      y: H / 2 + Math.sin(i * 2.4) * (100 + (i % 5) * 46),
      vx: 0, vy: 0,
      r: 4 + (n.rank || 0) * 9 + Math.min(4, (n.degree || 0) * 0.35),
    }));
    this.edges = d.edges || [];
    this.byId = new Map(this.nodes.map((n) => [n.id, n]));
    this.renderLegend();
    this.applyLayout(prefs.layout, true);
    this.fit();
  },

  visible(n) { return !this.hiddenTypes.has(n.type || "default"); },

  /* ---- layouts ---- */
  applyLayout(kind, silent) {
    prefs.layout = kind; savePrefs();
    $("#glayout").value = kind;
    const vis = this.nodes.filter((n) => this.visible(n));
    if (kind === "grid") {
      const cols = Math.ceil(Math.sqrt(vis.length || 1));
      vis.forEach((n, i) => {
        n.tx = 80 + (i % cols) * 96;
        n.ty = 80 + Math.floor(i / cols) * 88;
      });
      this.simTicks = 40; // ease toward targets
    } else if (kind === "radial") {
      const groups = {};
      vis.forEach((n) => (groups[n.type || "default"] ??= []).push(n));
      const types = Object.keys(groups);
      const cx = 450, cy = 350;
      types.forEach((t, gi) => {
        const ring = 90 + gi * 85;
        groups[t].forEach((n, i) => {
          const a = (i / groups[t].length) * Math.PI * 2 + gi * 0.5;
          n.tx = cx + Math.cos(a) * ring;
          n.ty = cy + Math.sin(a) * ring;
        });
      });
      this.simTicks = 40;
    } else {
      this.nodes.forEach((n) => { delete n.tx; delete n.ty; });
      this.simTicks = 240; // force iterations
    }
    if (!silent) this.kick();
  },

  /* ---- physics ---- */
  step() {
    const vis = this.nodes.filter((n) => this.visible(n));
    if (prefs.layout !== "force") {
      let moving = false;
      for (const n of vis) {
        if (n.tx == null) continue;
        n.x += (n.tx - n.x) * 0.18;
        n.y += (n.ty - n.y) * 0.18;
        if (Math.abs(n.tx - n.x) + Math.abs(n.ty - n.y) > 0.6) moving = true;
      }
      return moving;
    }
    // repulsion (sampled for perf on big graphs)
    for (let i = 0; i < vis.length; i++) {
      const a = vis[i];
      for (let j = i + 1; j < vis.length; j++) {
        const b = vis[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let d2 = dx * dx + dy * dy;
        if (d2 > 32000 || d2 === 0) continue;
        const f = 900 / d2;
        const d = Math.sqrt(d2);
        dx /= d; dy /= d;
        a.vx += dx * f; a.vy += dy * f;
        b.vx -= dx * f; b.vy -= dy * f;
      }
    }
    // springs
    for (const e of this.edges) {
      const a = this.byId.get(e.source), b = this.byId.get(e.target);
      if (!a || !b || !this.visible(a) || !this.visible(b)) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = (d - 95) * 0.004;
      a.vx += (dx / d) * f; a.vy += (dy / d) * f;
      b.vx -= (dx / d) * f; b.vy -= (dy / d) * f;
    }
    // integrate + gravity
    let energy = 0;
    for (const n of vis) {
      if (n === this.dragging) { n.vx = n.vy = 0; continue; }
      n.vx = (n.vx + (450 - n.x) * 0.0004) * 0.86;
      n.vy = (n.vy + (350 - n.y) * 0.0004) * 0.86;
      n.x += n.vx; n.y += n.vy;
      energy += Math.abs(n.vx) + Math.abs(n.vy);
    }
    return energy > 0.5;
  },

  kick() {
    if (this.raf) return;
    const loop = () => {
      const moving = this.simTicks-- > 0 && this.step();
      this.draw();
      if (moving || this.dragging) this.raf = requestAnimationFrame(loop);
      else this.raf = null;
    };
    this.raf = requestAnimationFrame(loop);
  },

  /* ---- camera helpers ---- */
  toWorld(px, py) {
    return { x: (px - this.cam.x) / this.cam.k, y: (py - this.cam.y) / this.cam.k };
  },
  zoomAt(px, py, factor) {
    const w = this.toWorld(px, py);
    this.cam.k = Math.max(0.15, Math.min(4, this.cam.k * factor));
    this.cam.x = px - w.x * this.cam.k;
    this.cam.y = py - w.y * this.cam.k;
    this.draw();
  },
  fit() {
    const vis = this.nodes.filter((n) => this.visible(n));
    if (!vis.length || !this.canvas) return;
    const xs = vis.map((n) => n.x), ys = vis.map((n) => n.y);
    const minX = Math.min(...xs) - 50, maxX = Math.max(...xs) + 50;
    const minY = Math.min(...ys) - 50, maxY = Math.max(...ys) + 50;
    const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
    this.cam.k = Math.max(0.15, Math.min(2.5, Math.min(cw / (maxX - minX), ch / (maxY - minY))));
    this.cam.x = cw / 2 - ((minX + maxX) / 2) * this.cam.k;
    this.cam.y = ch / 2 - ((minY + maxY) / 2) * this.cam.k;
    this.draw();
  },
  reset() { this.cam = { x: 0, y: 0, k: 1 }; this.highlightSet = null; this.selId = null; this.findMatch = null; $("#gfind").value = ""; this.closeDrawer(); this.fit(); },

  highlight(hl) {
    // hl: {nodes:[ids], edges?} from an answer — spotlight what was used
    const ids = Array.isArray(hl) ? hl : hl?.nodes || [];
    this.highlightSet = ids.length ? new Set(ids) : null;
    this.draw();
  },

  /* ---- rendering ---- */
  resize() {
    if (!this.canvas) return;
    const wrap = $("#gcanvas-wrap");
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = wrap.clientWidth * dpr;
    this.canvas.height = wrap.clientHeight * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.mini.width = 120 * dpr; this.mini.height = 84 * dpr;
    this.mctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  },

  draw() {
    const ctx = this.ctx;
    if (!ctx) return;
    const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
    ctx.clearRect(0, 0, cw, ch);
    ctx.save();
    ctx.translate(this.cam.x, this.cam.y);
    ctx.scale(this.cam.k, this.cam.k);

    const dim = this.highlightSet || (this.findMatch ? new Set([this.findMatch]) : null);

    // edges
    for (const e of this.edges) {
      const a = this.byId.get(e.source), b = this.byId.get(e.target);
      if (!a || !b || !this.visible(a) || !this.visible(b)) continue;
      const lit = dim && (dim.has(a.id) && dim.has(b.id));
      ctx.strokeStyle = dim ? (lit ? "rgba(240,160,60,.65)" : "rgba(60,75,105,.14)")
                            : "rgba(78,98,135,.30)";
      ctx.lineWidth = lit ? 1.6 : 0.8;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    // nodes
    for (const n of this.nodes) {
      if (!this.visible(n)) continue;
      const lit = !dim || dim.has(n.id);
      const sel = n.id === this.selId || n.id === this.hoverId || n.id === this.findMatch;
      ctx.globalAlpha = lit ? 1 : 0.18;
      ctx.fillStyle = colorFor(n.type);
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2); ctx.fill();
      if (sel) {
        ctx.strokeStyle = "#F0A03C"; ctx.lineWidth = 2 / this.cam.k;
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r + 3, 0, Math.PI * 2); ctx.stroke();
      }
      if (this.cam.k > 0.65 || sel || (n.rank || 0) > 0.55) {
        ctx.fillStyle = lit ? "rgba(231,237,247,.88)" : "rgba(231,237,247,.25)";
        ctx.font = `${10.5 / Math.max(0.8, this.cam.k)}px ui-monospace,Menlo,monospace`;
        ctx.fillText(n.label || n.id, n.x + n.r + 4, n.y + 3);
      }
      ctx.globalAlpha = 1;
    }
    ctx.restore();
    this.drawMini(cw, ch);
  },

  drawMini(cw, ch) {
    const m = this.mctx;
    m.clearRect(0, 0, 120, 84);
    const vis = this.nodes.filter((n) => this.visible(n));
    if (!vis.length) return;
    const xs = vis.map((n) => n.x), ys = vis.map((n) => n.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const sx = 110 / Math.max(1, maxX - minX), sy = 74 / Math.max(1, maxY - minY);
    const s = Math.min(sx, sy);
    const ox = 5 - minX * s, oy = 5 - minY * s;
    for (const n of vis) {
      m.fillStyle = colorFor(n.type);
      m.fillRect(n.x * s + ox, n.y * s + oy, 2, 2);
    }
    // viewport rect
    const tl = this.toWorld(0, 0), br = this.toWorld(cw, ch);
    m.strokeStyle = "rgba(240,160,60,.8)"; m.lineWidth = 1;
    m.strokeRect(tl.x * s + ox, tl.y * s + oy, (br.x - tl.x) * s, (br.y - tl.y) * s);
  },

  renderLegend() {
    const types = [...new Set(this.nodes.map((n) => n.type || "default"))];
    const box = $("#glegend");
    box.innerHTML = "";
    for (const t of types) {
      const li = document.createElement("span");
      li.className = "li" + (this.hiddenTypes.has(t) ? " off" : "");
      li.innerHTML = `<i style="background:${colorFor(t)}"></i>${esc(t)}`;
      li.onclick = () => {
        this.hiddenTypes.has(t) ? this.hiddenTypes.delete(t) : this.hiddenTypes.add(t);
        li.classList.toggle("off");
        this.draw();
      };
      box.appendChild(li);
    }
  },

  pick(px, py) {
    const w = this.toWorld(px, py);
    let best = null, bd = 1e9;
    for (const n of this.nodes) {
      if (!this.visible(n)) continue;
      const d = Math.hypot(n.x - w.x, n.y - w.y);
      if (d < n.r + 6 && d < bd) { best = n; bd = d; }
    }
    return best;
  },

  async openDrawer(n) {
    this.selId = n.id;
    $("#ndlabel").textContent = n.label || n.id;
    $("#ndtype").textContent = `${n.type || "node"} · rank ${(n.rank ?? 0).toFixed(2)} · ${n.degree ?? 0} links`;
    const rels = $("#ndrels");
    rels.innerHTML = `<div class="nr">Loading neighborhood…</div>`;
    $("#ndrawer").classList.add("open");
    this.draw();
    try {
      const r = await fetch(`${API}/api/graph/neighborhood/${encodeURIComponent(n.id)}?hops=2`);
      const d = await r.json();
      this.highlight(d.nodes.map((x) => x.id));
      rels.innerHTML = d.edges.slice(0, 24).map((e) =>
        `<div class="nr"><b>${esc(e.rel)}</b>${esc(this.byId.get(e.source)?.label || e.source)} → ${esc(this.byId.get(e.target)?.label || e.target)}</div>`,
      ).join("") || `<div class="nr">No relationships.</div>`;
    } catch {
      rels.innerHTML = `<div class="nr">Could not load neighborhood.</div>`;
    }
  },
  closeDrawer() { $("#ndrawer").classList.remove("open"); this.selId = null; },

  exportPNG() {
    const a = document.createElement("a");
    a.download = "axon-graph.png";
    a.href = this.canvas.toDataURL("image/png");
    a.click();
    toast("Graph exported as PNG");
  },
  exportJSON() {
    const blob = new Blob([JSON.stringify({ nodes: this.nodes.map(({ vx, vy, tx, ty, ...n }) => n), edges: this.edges }, null, 2)],
      { type: "application/json" });
    const a = document.createElement("a");
    a.download = "axon-graph.json";
    a.href = URL.createObjectURL(blob);
    a.click();
    URL.revokeObjectURL(a.href);
    toast("Graph exported as JSON");
  },

  bind() {
    this.canvas = $("#gcanvas"); this.ctx = this.canvas.getContext("2d");
    this.mini = $("#minimap"); this.mctx = this.mini.getContext("2d");
    new ResizeObserver(() => this.resize()).observe($("#gcanvas-wrap"));

    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      this.zoomAt(e.clientX - rect.left, e.clientY - rect.top, e.deltaY < 0 ? 1.12 : 0.89);
    }, { passive: false });

    this.canvas.addEventListener("pointerdown", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const px = e.clientX - rect.left, py = e.clientY - rect.top;
      const n = this.pick(px, py);
      this.canvas.setPointerCapture(e.pointerId);
      if (n) { this.dragging = n; this.simTicks = Math.max(this.simTicks, 30); this.kick(); }
      else { this.panning = { px, py, cx: this.cam.x, cy: this.cam.y }; this.canvas.classList.add("dragging"); }
      this._downAt = { px, py, node: n };
    });
    this.canvas.addEventListener("pointermove", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const px = e.clientX - rect.left, py = e.clientY - rect.top;
      if (this.dragging) {
        const w = this.toWorld(px, py);
        this.dragging.x = w.x; this.dragging.y = w.y;
        this.kick();
      } else if (this.panning) {
        this.cam.x = this.panning.cx + (px - this.panning.px);
        this.cam.y = this.panning.cy + (py - this.panning.py);
        this.draw();
      } else {
        const n = this.pick(px, py);
        if ((n?.id || null) !== this.hoverId) { this.hoverId = n?.id || null; this.draw(); }
        this.canvas.style.cursor = n ? "pointer" : "grab";
      }
    });
    this.canvas.addEventListener("pointerup", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const px = e.clientX - rect.left, py = e.clientY - rect.top;
      const moved = this._downAt && Math.hypot(px - this._downAt.px, py - this._downAt.py) > 5;
      if (!moved && this._downAt?.node) this.openDrawer(this._downAt.node);
      else if (!moved && !this._downAt?.node) { this.highlightSet = null; this.closeDrawer(); this.draw(); }
      this.dragging = null; this.panning = false;
      this.canvas.classList.remove("dragging");
    });

    $("#g-zi").onclick = () => this.zoomAt(this.canvas.clientWidth / 2, this.canvas.clientHeight / 2, 1.25);
    $("#g-zo").onclick = () => this.zoomAt(this.canvas.clientWidth / 2, this.canvas.clientHeight / 2, 0.8);
    $("#g-fit").onclick = () => this.fit();
    $("#g-reset").onclick = () => this.reset();
    $("#g-png").onclick = () => this.exportPNG();
    $("#g-json").onclick = () => this.exportJSON();
    $("#g-full").onclick = () => {
      graphside.classList.toggle("fullscreen");
      $("#g-full").classList.toggle("on");
      setTimeout(() => { this.resize(); this.fit(); }, 60);
    };
    $("#ndclose").onclick = () => { this.closeDrawer(); this.highlightSet = null; this.draw(); };
    $("#glayout").onchange = (e) => { this.applyLayout(e.target.value); };
    $("#gfind").oninput = debounce(() => {
      const q = $("#gfind").value.trim().toLowerCase();
      this.findMatch = null;
      if (q) {
        const n = this.nodes.find((x) => this.visible(x) &&
          ((x.label || x.id).toLowerCase().includes(q)));
        if (n) {
          this.findMatch = n.id;
          // center camera on match
          this.cam.x = this.canvas.clientWidth / 2 - n.x * this.cam.k;
          this.cam.y = this.canvas.clientHeight / 2 - n.y * this.cam.k;
        }
      }
      this.draw();
    }, 220);
  },
};

/* ------------------------------------------------------------------ boot */
window.addEventListener("load", () => {
  applyPanelPrefs();
  renderProfile();
  Graph.bind();
  Chat.hello();
  Status.refresh();
  Docs.refresh();
  Conv.refresh();
  Graph.load();
  Asset360.init();
  Maintenance.init();
  AssetHistory.init();
  $("#glayout").value = prefs.layout;
  setInterval(() => Status.refresh(), 30000);
});

/* --------------------------------------------------------------- Asset360
   A complete digital profile for one physical asset: live condition and
   remaining-useful-life prediction, maintenance history, spares, linked
   documents and P&ID neighbourhood. Every value is rendered from
   /api/asset/{id} — the backend assembles it from the knowledge graph,
   maintenance log, spares list and sensor stream. Nothing is generated
   here; this is a view over the system of record.
   ------------------------------------------------------------------- */
const Asset360 = {
  open: false,

  async init() {
    $("#tgl-asset").addEventListener("click", () => Asset360.toggle());
    $("#a3-close").addEventListener("click", () => Asset360.toggle(false));
    $("#a3-pick").addEventListener("change", (e) => Asset360.load(e.target.value));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && Asset360.open) Asset360.toggle(false);
    });
  },

  async toggle(force) {
    Asset360.open = force === undefined ? !Asset360.open : force;
    $("#asset360").hidden = !Asset360.open;
    if (Asset360.open) await Asset360.refreshAssets();
  },

  async refreshAssets(preferredId) {
    try {
      const r = await fetch(`${API}/api/assets`);
      const { assets } = await r.json();
      if (!assets.length) {
        $("#a3-body").innerHTML =
          `<div class="a3-empty">No physical assets in the knowledge graph.</div>`;
        return;
      }
      const selected = preferredId || $("#a3-pick").value;
      $("#a3-pick").innerHTML = assets.map(a =>
        `<option value="${esc(a.id)}">${esc(a.id)} — ${esc(a.type)}${
          a.monitored ? " ●" : ""}</option>`).join("");
      const active = assets.some(a => a.id === selected) ? selected : assets[0].id;
      $("#a3-pick").value = active;
      if (Asset360.open) await Asset360.load(active);
    } catch {
      $("#a3-body").innerHTML =
        `<div class="a3-empty">Could not load assets.</div>`;
    }
  },

  async load(id) {
    $("#a3-body").innerHTML = `<div class="a3-empty">Loading ${esc(id)}…</div>`;
    try {
      const r = await fetch(`${API}/api/asset/${encodeURIComponent(id)}`);
      if (!r.ok) throw new Error(String(r.status));
      Asset360.render(await r.json());
    } catch {
      $("#a3-body").innerHTML =
        `<div class="a3-empty">Could not load ${esc(id)}.</div>`;
    }
  },

  /* 14-day vibration trend with alert/danger threshold lines. */
  sparkline(series, pred) {
    const vals = series.map(p => p.vibration_mm_s);
    if (vals.length < 2) return "";
    const W = 300, H = 64, danger = pred.danger_limit, alert = pred.alert_limit;
    const lo = Math.min(...vals, 0), hi = Math.max(...vals, danger) * 1.05;
    const x = i => (i / (vals.length - 1)) * W;
    const y = v => H - ((v - lo) / (hi - lo)) * H;
    const path = vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
    return `<svg class="a3-spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <line x1="0" y1="${y(danger).toFixed(1)}" x2="${W}" y2="${y(danger).toFixed(1)}"
            stroke="var(--bad)" stroke-width="1" stroke-dasharray="4 3" opacity=".8"/>
      <line x1="0" y1="${y(alert).toFixed(1)}" x2="${W}" y2="${y(alert).toFixed(1)}"
            stroke="var(--warn)" stroke-width="1" stroke-dasharray="4 3" opacity=".8"/>
      <path d="${path}" fill="none" stroke="var(--accent)" stroke-width="1.6"/>
    </svg>`;
  },

  render(d) {
    const cards = [];
    const c = d.condition, p = c && c.prediction;
    const health = !p ? "unknown"
      : p.latest_vibration >= p.danger_limit ? "danger"
      : p.in_alert_zone ? "alert" : "healthy";

    cards.push(`<div class="a3-card a3-hero">
      <div>
        <div class="a3-id">${esc(d.id)}</div>
        <div class="a3-sub">${esc(d.label)} · ${esc(d.type)}</div>
      </div>
      <span class="a3-pill h-${health}">${health}</span>
      <span class="a3-spacer"></span>
      <div class="a3-chips">
        <span class="a3-chip">P&amp;ID <b>${esc(d.pid.drawing || "—")}</b> rev ${esc(d.pid.revision || "—")}</span>
        <span class="a3-chip">work orders <b>${d.maintenance.count}</b></span>
        <span class="a3-chip">downtime <b>${d.maintenance.total_downtime_hours} h</b></span>
        <span class="a3-chip">documents <b>${d.documents.length}</b></span>
      </div>
    </div>`);

    const props = d.properties || {};
    const profileRows = [
      ["Manufacturer", props.manufacturer], ["Model", props.model_number],
      ["Serial number", props.serial_number], ["Health score", props.health_score],
      ["Operating status", props.status], ["Location", props.location],
      ["Latest update", props.last_modified],
    ].filter(([, value]) => value != null && value !== "");
    if (profileRows.length) {
      cards.push(`<div class="a3-card"><h4>Asset profile</h4>${profileRows.map(([label, value]) =>
        `<div class="a3-kv"><span>${esc(label)}</span><span>${esc(String(value))}</span></div>`).join("")}</div>`);
    }

    if (p) {
      cards.push(`<div class="a3-card">
        <h4>Live condition — ${esc(p.signal || "vibration")}</h4>
        <div class="a3-big" style="color:var(--${health === "danger" ? "bad" : health === "alert" ? "warn" : "good"})">
          ${p.latest_vibration} <span style="font-size:12px;color:var(--dim)">mm/s</span></div>
        ${Asset360.sparkline(c.series, p)}
        <div class="a3-kv"><span>Baseline</span><span>${p.baseline_vibration} mm/s</span></div>
        <div class="a3-kv"><span>Alert / danger limit</span><span>${p.alert_limit} / ${p.danger_limit} mm/s</span></div>
        <div class="a3-kv"><span>Trend</span><span>${p.trend_mm_s_per_day > 0 ? "+" : ""}${p.trend_mm_s_per_day} mm/s per day</span></div>
        <div class="a3-kv"><span>Bearing temperature</span><span>${p.bearing_temp_recent_c ?? "—"} °C</span></div>
        <div class="a3-kv"><span>Anomaly</span><span>${p.anomaly ? "YES" : "no"}</span></div>
        <div class="a3-note">14-day trend · dashed lines are the ISO 10816 alert and danger limits.</div>
      </div>`);
      cards.push(`<div class="a3-card">
        <h4>Predicted remaining useful life</h4>
        <div class="a3-big" style="color:var(--${p.rul_days <= 3 ? "bad" : p.rul_days <= 14 ? "warn" : "good"})">
          ${p.rul_days ?? "—"} <span style="font-size:12px;color:var(--dim)">days</span></div>
        <div class="a3-note">Trend extrapolation of the vibration signal to the
          danger limit. An estimate from the sensor stream — not a guarantee.</div>
      </div>`);
    }

    if (d.maintenance.failure_modes.length) {
      Asset360._wos = d.maintenance.work_orders;
      cards.push(`<div class="a3-card">
        <h4>Failure history</h4>
        <div class="a3-chips" style="margin-bottom:9px">
          ${d.maintenance.failure_modes.map(f =>
            `<span class="a3-chip"><b>${f.count}×</b> ${esc(f.mode)}</span>`).join("")}
        </div>
        ${d.maintenance.work_orders.map((w, i) => {
          const sourced = !!(w.source_document || w.from_document);
          const conf = w.confidence != null && w.confidence !== ""
            ? Math.round(Number(w.confidence) * 100) + "%" : "";
          return `
          <div class="a3-wo ${esc(w.type)} ${sourced ? "src" : ""}" data-idx="${i}"
               ${sourced ? `title="From ${esc(w.source_document)} — click to view source"` : ""}>
            <div class="r1"><span class="wo">${esc(w.wo_number)}</span>
              <span class="dt">${esc(w.date)} · ${esc(w.type)} · ${esc(w.downtime_hours)}h</span>
              ${sourced ? `<span class="srcbadge">▤ source</span>` : ""}</div>
            <div class="r2">${esc(w.failure_mode || w.symptom || "")}${
              (w.cause || w.root_cause) ? " → <i>" + esc(w.cause || w.root_cause) + "</i>" : ""}</div>
            <div class="r2" style="color:var(--faint)">${esc(w.action || w.corrective_action || "")}</div>
            ${sourced ? `<div class="r3">${[
              w.engineer ? "👤 " + esc(w.engineer) : "",
              w.source_document ? "▤ " + esc(w.source_document) + (w.page_number ? " p." + esc(String(w.page_number)) : "") : "",
              conf ? "◈ " + conf : "",
            ].filter(Boolean).join(" &nbsp;·&nbsp; ")}</div>` : ""}
          </div>`; }).join("")}
      </div>`);
    }

    if (d.spares.length) {
      cards.push(`<div class="a3-card">
        <h4>Spares</h4>
        ${d.spares.map(s => `<div class="a3-kv">
          <span>${esc(s.part_number)} — ${esc(s.description || "")}</span>
          <span class="${Number(s.qty_on_hand) <= Number(s.min_stock) ? "a3-low" : ""}">${
            esc(s.qty_on_hand)} / min ${esc(s.min_stock)}</span></div>`).join("")}
      </div>`);
    }

    const g = d.graph;
    const grp = (title, items) => items.length
      ? `<div style="margin-bottom:8px"><div style="color:var(--faint);font-size:10.5px;margin-bottom:4px">${title}</div>
         <div class="a3-chips">${items.map(i =>
           `<span class="a3-chip"><b>${esc(i.id)}</b> ${esc(i.type)}</span>`).join("")}</div></div>` : "";
    cards.push(`<div class="a3-card">
      <h4>Connected equipment (from P&amp;ID)</h4>
      ${grp("Connected to", g.connected_to)}
      ${grp("Measured by", g.measured_by)}
      ${grp("Failure modes", g.failure_modes)}
      ${grp("Other", g.other)}
    </div>`);

    if (d.documents.length) {
      cards.push(`<div class="a3-card">
        <h4>Documents referencing this asset</h4>
        ${d.documents.map(x => `<div class="a3-kv">
          <span>${esc(x.title)}</span><span>${x.chunks} chunk${x.chunks > 1 ? "s" : ""}</span>
        </div>`).join("")}
        <div class="a3-note">Ask a question to see these cited as evidence.</div>
      </div>`);
    }

    $("#a3-body").innerHTML = cards.join("");
    // Source traceability: clicking a document-sourced work order opens the
    // original PDF at the extracted page alongside the extracted fields.
    $$("#a3-body .a3-wo.src").forEach((el) => {
      el.onclick = () => Maintenance.openSource(Asset360._wos[Number(el.dataset.idx)]);
    });
  },
};

/* ------------------------------------------------ Asset360 Digital History
   An explicit, end-to-end path for engineering-asset documents.  The source
   has already been indexed when this appears; Cancel therefore preserves the
   original upload → RAG behaviour exactly. */
const AssetHistory = {
  state: null,

  init() {
    $("#mx-x").onclick = () => this.close();
    $("#mx-backdrop").onclick = () => this.close();
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !$("#mx").hidden) this.close();
    });
  },

  open() { $("#mx-dialog").classList.remove("wide"); $("#mx").hidden = false; },
  close() {
    $("#mx").hidden = true;
    $("#mx-body").innerHTML = "";
    $("#mx-foot").innerHTML = "";
    this.state = null;
  },
  setFoot(buttons) {
    const foot = $("#mx-foot"); foot.innerHTML = "";
    buttons.forEach((b) => {
      const el = document.createElement("button");
      el.className = `mx-btn ${b.cls || ""}`; el.textContent = b.label;
      el.onclick = b.on; foot.appendChild(el);
    });
  },

  begin(d) {
    this.state = { docNo: d.doc_no, title: d.classification?.title || d.doc_no,
      classification: d.classification || {}, assets: d.detected_assets || [] };
    this.open(); this.renderPrompt();
  },

  renderPrompt() {
    const s = this.state;
    const names = s.assets.map(a => a.asset_id).filter(Boolean);
    $("#mx-title").textContent = "Generate Asset360 Digital History?";
    $("#mx-sub").textContent = s.title;
    $("#mx-body").innerHTML = `
      <div class="mx-ask">
        <h3>Generate Asset360 Digital History?</h3>
        <p>This document appears to contain engineering asset information.
          Asset360 can automatically generate an asset profile, company manual,
          maintenance, service and inspection reports, a timeline, knowledge
          graph links and a Digital Twin.</p>
      </div>
      <div class="mx-steps" style="margin-top:14px">
        <div class="mx-step done"><div class="mx-dot">✓</div><div><span class="st">Document uploaded and indexed into RAG</span>
          <span class="sm">The original document remains searchable in either choice.</span></div></div>
        <div class="mx-step active"><div class="mx-dot">●</div><div><span class="st">Engineering assets detected</span>
          <span class="sm">${names.length ? esc(names.join(", ")) : "Asset metadata detected"}</span></div></div>
      </div>`;
    this.setFoot([
      { label: "Cancel", on: () => { toast("Indexed into RAG only — no Digital History generated."); this.close(); } },
      { label: "Generate History", cls: "primary", on: () => this.generate() },
    ]);
  },

  renderProgress(active, complete = false) {
    const steps = ["Reading document", "OCR / vision processing (when needed)",
      "Classifying document", "Extracting metadata", "Identifying assets",
      "Generating company manual and reports", "Generating timeline",
      "Updating knowledge graph and asset database", "Refreshing Asset360"];
    $("#mx-title").textContent = complete ? "Asset360 history generated" : "Generating Asset360 history";
    $("#mx-body").innerHTML = `<div class="mx-steps">${steps.map((step, i) => {
      const state = complete || i < active ? "done" : i === active ? "active" : "pending";
      return `<div class="mx-step ${state}"><div class="mx-dot">${state === "done" ? "✓" : i + 1}</div>
        <div><span class="st">${esc(step)}</span></div></div>`;
    }).join("")}</div>`;
  },

  async generate() {
    this.renderProgress(0); this.setFoot([]);
    let stage = 0;
    const timer = setInterval(() => { stage = Math.min(stage + 1, 7); this.renderProgress(stage); }, 480);
    try {
      const r = await fetch(`${API}/api/asset-history/generate`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_no: this.state.docNo }),
      });
      const d = await r.json();
      if (!r.ok || !d.ok) throw new Error(d.message || d.detail || "history generation failed");
      clearInterval(timer); this.renderProgress(9, true);
      const assets = d.asset_ids || [];
      $("#mx-sub").textContent = assets.length ? `Generated: ${assets.join(", ")}` : this.state.title;
      this.setFoot([{ label: "Open Asset360", cls: "primary", on: async () => {
        this.close(); await Asset360.toggle(true);
        if (assets[0]) await Asset360.refreshAssets(assets[0]);
      } }]);
      Docs.refresh(); Status.refresh(); Graph.load();
      await Asset360.refreshAssets(assets[0]);
      toast(`Asset360 history generated for ${assets.join(", ")}.`);
    } catch (e) {
      clearInterval(timer);
      $("#mx-title").textContent = "History generation unavailable";
      $("#mx-body").innerHTML = `<div class="mx-load">Unable to generate Asset360 history automatically. The uploaded document is still searchable in RAG.<br><br>${esc(e.message)}</div>`;
      this.setFoot([{ label: "Close", on: () => this.close() }]);
    }
  },
};

/* ------------------------------------------------------- Asset360 ingestion
   The intelligent maintenance-ingestion workflow. After an upload, if the
   backend classified the document as maintenance-related, this walks the user
   through: detect → ask permission → (optionally choose assets) → extract →
   review/edit with validation → confirm → live Asset360 update. Asset360 is
   NEVER modified without an explicit Confirm here.
   ------------------------------------------------------------------------- */
const Maintenance = {
  state: null,
  editMode: false,

  init() {
    $("#mx-x").onclick = () => this.close();
    $("#mx-backdrop").onclick = () => this.close();
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !$("#mx").hidden) this.close();
    });
  },

  open(wide = false) {
    $("#mx-dialog").classList.toggle("wide", wide);
    $("#mx").hidden = false;
  },
  close() {
    $("#mx").hidden = true;
    $("#mx-body").innerHTML = "";
    $("#mx-foot").innerHTML = "";
    this.state = null; this.editMode = false;
  },

  setFoot(buttons) {
    const foot = $("#mx-foot");
    foot.innerHTML = "";
    for (const b of buttons) {
      const el = document.createElement("button");
      el.className = `mx-btn ${b.cls || ""}`;
      el.textContent = b.label;
      if (b.disabled) el.disabled = true;
      el.onclick = b.on;
      foot.appendChild(el);
    }
  },

  /* Stage 1 — detection + permission --------------------------------------- */
  begin(d) {
    this.state = {
      docNo: d.doc_no,
      title: d.classification?.title || d.doc_no,
      classification: d.classification || {},
      candidateAssets: d.candidate_assets || [],
      events: null,
    };
    this.editMode = false;
    this.open(false);
    this.renderDetect();
  },

  renderDetect() {
    const s = this.state, c = s.classification;
    const conf = Math.round((c.confidence || 0) * 100);
    $("#mx-title").textContent = "Asset360 ingestion";
    $("#mx-sub").textContent = s.title;
    $("#mx-body").innerHTML = `
      <div class="mx-steps">
        <div class="mx-step done"><div class="mx-dot">✓</div>
          <div><span class="st">Document uploaded &amp; indexed into RAG</span>
            <span class="sm">${esc(s.title)}</span></div></div>
        <div class="mx-step done"><div class="mx-dot">✓</div>
          <div><span class="st">Document classified</span>
            <span class="sm"><span class="mx-type">${esc(c.type || "Document")}</span>
              &nbsp;${conf}% confidence</span></div></div>
        <div class="mx-step active"><div class="mx-dot">●</div>
          <div><span class="st">Maintenance information detected</span>
            <span class="sm">${s.candidateAssets.length
              ? "References: " + s.candidateAssets.map(esc).join(", ")
              : "No specific asset tag detected — you can set it during review"}</span></div></div>
      </div>
      <div class="mx-ask">
        <h3>This document appears to contain maintenance information.</h3>
        <p>Would you like to update Asset360 with the events in this document?
           Choosing <b>Index only</b> keeps the current behaviour — the document
           stays searchable in RAG and Asset360 is left unchanged.</p>
      </div>`;
    this.setFoot([
      { label: "Index only", on: () => this.indexOnly() },
      { label: "Update Asset360", cls: "primary", on: () => this.startUpdate() },
    ]);
  },

  indexOnly() {
    toast("Indexed into RAG only — Asset360 left unchanged.");
    this.close();
  },

  /* Stage 2 — asset selection (bonus: multiple assets) --------------------- */
  startUpdate() {
    if ((this.state.candidateAssets || []).length > 1) {
      this.renderAssetChooser();
    } else {
      this.extract(this.state.candidateAssets.length ? this.state.candidateAssets : null);
    }
  },

  renderAssetChooser() {
    const s = this.state;
    $("#mx-title").textContent = "Choose assets";
    $("#mx-body").innerHTML = `
      <p style="color:var(--dim);font-size:13px;margin:0 0 4px">
        This document references multiple assets. Select which asset(s) should
        receive a maintenance history entry.</p>
      <div class="mx-assets" id="mx-assets">
        ${s.candidateAssets.map((a, i) => `
          <label class="mx-asset"><input type="checkbox" value="${esc(a)}" checked>
            <span class="tag">${esc(a)}</span></label>`).join("")}
      </div>`;
    this.setFoot([
      { label: "Back", on: () => this.renderDetect() },
      { label: "Extract selected", cls: "primary", on: () => {
        const picked = $$("#mx-assets input:checked").map((i) => i.value);
        if (!picked.length) { toast("Select at least one asset."); return; }
        this.extract(picked);
      } },
    ]);
  },

  /* Stage 3 — extraction --------------------------------------------------- */
  async extract(assetIds) {
    $("#mx-title").textContent = "Extracting maintenance data";
    $("#mx-sub").textContent = this.state.title;
    $("#mx-body").innerHTML = `<div class="mx-load">⏳ Running information extraction…</div>`;
    this.setFoot([]);
    try {
      const r = await fetch(`${API}/api/maintenance/extract`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_no: this.state.docNo, asset_ids: assetIds }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "extraction failed");
      this.state.events = d.events || [];
      this.editMode = false;
      if (!this.state.events.length) {
        $("#mx-body").innerHTML = `<div class="mx-load">No maintenance events could be extracted from this document.</div>`;
        this.setFoot([{ label: "Close", on: () => this.close() }]);
        return;
      }
      this.renderPreview();
    } catch (e) {
      $("#mx-body").innerHTML = `<div class="mx-load">Extraction failed: ${esc(e.message)}</div>`;
      this.setFoot([{ label: "Close", on: () => this.close() }]);
    }
  },

  /* Stage 4 — preview / edit / validate ------------------------------------ */
  FIELDS: [
    ["asset_id", "Asset", false], ["date", "Date", false],
    ["event_type", "Event", false], ["work_order", "Work order", false],
    ["failure_mode", "Failure", false], ["severity", "Severity", false],
    ["root_cause", "Cause", true], ["corrective_action", "Action", true],
    ["preventive_action", "Preventive action", true],
    ["parts_used", "Parts", false], ["downtime_hours", "Downtime (h)", false],
    ["engineer", "Engineer", false], ["cost", "Cost", false],
    ["source_document", "Source", false], ["page_number", "Page", false],
  ],

  confClass(v) { return v >= 0.85 ? "hi" : v >= 0.6 ? "mid" : "lo"; },

  renderPreview() {
    const s = this.state;
    $("#mx-title").textContent = this.editMode ? "Edit maintenance data" : "Review extracted data";
    $("#mx-sub").textContent = s.title;
    this.open(true);

    const cards = s.events.map((ev, i) => {
      const conf = Number(ev.confidence || 0);
      const findings = (ev.validation?.findings) || [];
      const fields = this.FIELDS.map(([key, label, span]) => {
        let val = ev[key];
        if (key === "parts_used") val = Array.isArray(val) ? val.join(", ") : (val || "");
        val = val == null ? "" : String(val);
        const ro = this.editMode ? "" : "readonly";
        return `<div class="mx-f ${span ? "span" : ""}">
          <label>${label}</label>
          <input data-ev="${i}" data-key="${key}" value="${esc(val)}" ${ro}
            placeholder="—">
        </div>`;
      }).join("");
      const findingHtml = findings.length
        ? `<div class="mx-findings">${findings.map((f) => `
            <div class="mx-find ${f.level}"><span class="ic">${f.level === "error" ? "✕" : "!"}</span>
              <span>${esc(f.message)}</span></div>`).join("")}</div>`
        : `<div class="mx-ok">✓ Validation passed — asset, parts, work order and date look consistent.</div>`;
      return `<div class="mx-ev">
        <div class="mx-ev-head">
          <span class="mx-ev-asset">${esc(ev.asset_id || "—")}</span>
          <span class="mx-conf ${this.confClass(conf)}">${Math.round(conf * 100)}% confidence</span>
        </div>
        <div class="mx-grid">${fields}</div>
        ${findingHtml}
      </div>`;
    }).join("");

    $("#mx-body").innerHTML = `
      <p style="color:var(--dim);font-size:12.5px;margin:0 0 12px">
        Nothing is written to Asset360 until you press <b>Confirm</b>.
        ${this.editMode ? "Fields are editable — adjust anything, then confirm."
          : "Review the extracted fields. Use <b>Edit</b> to correct anything."}</p>
      ${cards}`;
    this.setFoot([
      { label: "Cancel", cls: "danger", on: () => this.close() },
      { label: this.editMode ? "Done editing" : "Edit",
        on: () => { this.collect(); this.editMode = !this.editMode; this.renderPreview(); } },
      { label: "Confirm", cls: "primary", on: () => this.confirm() },
    ]);
  },

  collect() {
    // Pull any edits from the inputs back into state (works in both modes).
    $$("#mx-body input[data-ev]").forEach((inp) => {
      const ev = this.state.events[Number(inp.dataset.ev)];
      const key = inp.dataset.key;
      if (key === "parts_used") {
        ev[key] = inp.value.split(",").map((x) => x.trim()).filter(Boolean);
      } else if (key === "downtime_hours") {
        ev[key] = parseFloat(inp.value) || 0;
      } else {
        ev[key] = inp.value;
      }
    });
  },

  /* Stage 5 — commit → live Asset360 update -------------------------------- */
  async confirm() {
    this.collect();
    const payload = this.state.events.map((ev) => {
      const { validation, ...clean } = ev;
      return clean;
    });
    $("#mx-body").innerHTML = `<div class="mx-load">⏳ Updating Asset360 — history, timeline, spares and knowledge graph…</div>`;
    this.setFoot([]);
    try {
      const r = await fetch(`${API}/api/maintenance/commit`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ events: payload }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "commit failed");
      if (!d.ok) {
        // Blocking validation errors — surface and let the user edit.
        this.editMode = true;
        this.renderPreview();
        const box = document.createElement("div");
        box.className = "mx-findings";
        box.innerHTML = (d.findings || []).map((f) => `
          <div class="mx-find error"><span class="ic">✕</span><span>${esc(f.message)}</span></div>`).join("");
        $("#mx-body").prepend(box);
        toast(d.message || "Please resolve the validation errors.");
        return;
      }
      const assets = d.asset_ids || [];
      const spares = (d.spares_updated || []).length;
      toast(`Asset360 updated: ${d.committed} event${d.committed > 1 ? "s" : ""}` +
            (assets.length ? ` on ${assets.join(", ")}` : "") +
            (spares ? `, ${spares} spare${spares > 1 ? "s" : ""} drawn down` : ""));
      this.close();
      // Live refresh — no backend restart.
      Docs.refresh(); Status.refresh(); Graph.load();
      if (assets.length) {
        await Asset360.toggle(true);
        const pick = $("#a3-pick");
        if (pick && [...pick.options].some((o) => o.value === assets[0])) {
          pick.value = assets[0];
          await Asset360.load(assets[0]);
        }
      }
    } catch (e) {
      $("#mx-body").innerHTML = `<div class="mx-load">Update failed: ${esc(e.message)}</div>`;
      this.setFoot([{ label: "Close", on: () => this.close() }]);
    }
  },

  /* Source traceability — open the source PDF at the extracted page --------- */
  async openSource(ev) {
    this.open(true);
    $("#mx-title").textContent = "Source traceability";
    $("#mx-sub").textContent = ev.source_document || "";
    const file = ev.source_document || "";
    const page = ev.page_number || "";
    const isPdf = /\.pdf$/i.test(file);
    const url = file ? `${API}/api/uploads/${encodeURIComponent(file)}${isPdf && page ? "#page=" + encodeURIComponent(page) : ""}` : "";
    const rows = [
      ["Asset", ev.asset_id || ev.equipment], ["Date", ev.date],
      ["Event", ev.event_type || ev.type],
      ["Failure", ev.failure_mode], ["Cause", ev.root_cause || ev.cause],
      ["Action", ev.corrective_action || ev.action],
      ["Downtime", ev.downtime_hours != null ? ev.downtime_hours + " h" : ""],
      ["Engineer", ev.engineer], ["Parts", Array.isArray(ev.parts_used) ? ev.parts_used.join(", ") : ev.parts_used],
      ["Work order", ev.work_order || ev.wo_number], ["Page", page],
      ["Confidence", ev.confidence != null ? Math.round(Number(ev.confidence) * 100) + "%" : ""],
    ].filter(([, v]) => v);
    $("#mx-body").innerHTML = `
      <div class="mx-trace">
        <div>${url
          ? `<iframe src="${esc(url)}" title="Source document"></iframe>`
          : `<div class="noframe">Source document not available.</div>`}
          <div class="mx-note" style="color:var(--faint);font-size:11px;margin-top:7px">
            Original document${page ? `, opened at page ${esc(String(page))}` : ""}.
          </div>
        </div>
        <div class="fields">
          <div style="font:600 9px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--accent);margin-bottom:9px">Extracted fields</div>
          ${rows.map(([k, v]) => `<div class="mx-f"><label>${esc(k)}</label>
            <div class="val">${esc(String(v))}</div></div>`).join("")}
        </div>
      </div>`;
    this.setFoot([{ label: "Close", on: () => this.close() }]);
  },
};

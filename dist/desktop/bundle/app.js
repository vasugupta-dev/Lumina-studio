/* Lumina Studio — v2 frontend.
 * Same server API, same localStorage keys (lum-*), same features.
 * A single assistant-status line follows the real request lifecycle.
 */
"use strict";

const $ = (s) => document.querySelector(s);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function on(id, ev, fn) {
  const el = document.getElementById(id);
  if (el) el.addEventListener(ev, fn);
  return el;
}

/* ---------- API (real backend; honest offline empties when unreachable) ---------- */

async function mockApi(path, opts = {}) {
  // Backend unreachable: honest empties, never fiction. Every offline
  // surface says so; POSTs throw above and the UI shows "backend unreachable".
  await sleep(120);
  if (path === "/api/library") return {items: [], dirs: [], offline: true};
  if (path === "/api/config") return {cloud: {}};
  if (path === "/api/loaded") return {loaded: localStorage.getItem("lum-loaded") || null};
  if (path.startsWith("/api/hf-search")) return {items: [], error: "backend unreachable — offline"};
  if (path === "/api/fetch-url") return {error: "backend unreachable — offline"};
  if (path === "/api/graph/stats") return {nodes: 0, edges: 0, by_kind: {}, hubs: []};
  if (path.startsWith("/api/graph/nodes")) return {items: []};
  if (path === "/api/graph/node") return {error: "backend unreachable — offline"};
  if (path === "/api/claims") return {items: []};
  if (path === "/api/memory/stats") return {tiers: [], archived: 0};
  if (path === "/api/memory/list") return {items: []};
  if (path === "/api/bench-tiny") return {error: "backend unreachable — offline"};
  if (path === "/api/targets") return {models: [], measured: {label: "offline", cases: []}};
  if (path === "/api/status") return {formats: [], status: "offline"};
  if (path === "/api/apps") return {apps: {available: {}, count_available: 0}, operator: {}, offline: true};
  if (path === "/api/custom_mcp") return {servers: [], example: null, offline: true};
  if (path === "/api/workflows") return {examples: [], kinds: [], offline: true};
  if (path === "/api/skills") return {skills: [], builtin: 0, user: 0, offline: true};
  if (path === "/api/media/backends") return {backends: [], policy: "off", offline: true};
  if (path === "/api/media/workflows") return {workflows: [], offline: true};
  if (path === "/api/media/jobs") return {jobs: [], count: 0, offline: true};
  if (path === "/api/residency") return {loaded: false, model_id: null, policy: "off", offline: true};
  return {};
}

async function api(path, opts = {}) {
  try {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error("http " + r.status);
    return await r.json();
  } catch (e) {
    if (opts && opts.method && opts.method !== "GET") throw e;
    return mockApi(path, opts);
  }
}

/* ---------- store (same keys — user data survives) ---------- */
const store = {
  config: (() => { try { return JSON.parse(localStorage.getItem("lum-config")) || {}; }
    catch (e) { return {}; } })(),
  threads: (() => { try { return JSON.parse(localStorage.getItem("lum-threads")) || []; }
    catch (e) { return []; } })(),
  tasks: (() => { try { return JSON.parse(localStorage.getItem("lum-tasks")) || []; }
    catch (e) { return []; } })(),
  docs: (() => { try { return JSON.parse(localStorage.getItem("lum-docs")) || []; }
    catch (e) { return []; } })(),
  loaded: localStorage.getItem("lum-loaded") || null,
  save() {
    try {
      localStorage.setItem("lum-config", JSON.stringify(this.config));
      localStorage.setItem("lum-threads", JSON.stringify(this.threads));
      localStorage.setItem("lum-tasks", JSON.stringify(this.tasks));
      localStorage.setItem("lum-docs", JSON.stringify(this.docs));
      if (this.loaded) localStorage.setItem("lum-loaded", this.loaded);
    } catch (e) { /* storage full or blocked */ }
  },
};

/* ---------- assistant status (plain text line, follows requests) ---------- */
let assistantStatus = "idle";
let statusResetT = null;
const STATUS_LABELS = {idle: "", thinking: "Thinking…", working: "Working…",
  streaming: "Replying…", listening: "Listening…", waiting: "", success: "Done", error: "Something went wrong"};

function drawStatus() {
  const lab = document.getElementById("assistant-status");
  if (lab) lab.textContent = STATUS_LABELS[assistantStatus] || "";
}
function setStatus(st) {
  assistantStatus = st;
  if (statusResetT) { clearTimeout(statusResetT); statusResetT = null; }
  drawStatus();
}
function settleStatus(st, ms) {
  setStatus(st);
  if (statusResetT) clearTimeout(statusResetT);
  statusResetT = setTimeout(() => setStatus("idle"), ms);
}

/* ---------- theme ---------- */
function setTheme(t) {
  if (!["light", "dark", "oled"].includes(t)) t = "dark";
  document.body.dataset.theme = t;
  store.config.theme = t;
  store.save();
  document.querySelectorAll("#theme-seg button").forEach((b) => b.classList.toggle("on", b.dataset.theme === t));
  document.querySelectorAll(".theme-dots button").forEach((b) => b.classList.toggle("on", b.dataset.t === t));
}

/* ---------- views ---------- */
function showView(v) {
  if (!v) return;
  document.querySelectorAll(".view").forEach((x) => x.classList.remove("active"));
  const el = document.getElementById("view-" + v);
  if (el) el.classList.add("active");
  document.querySelectorAll(".nav button").forEach((x) => x.classList.toggle("active", x.dataset.view === v));
  document.querySelectorAll(".chips button").forEach((x) => x.classList.toggle("active", x.dataset.view === v));
  if (window.innerWidth <= 760) document.body.classList.remove("drawer-open");
  if (v === "graph") { renderGraphStats(); renderGraphList($("#graph-q").value || ""); }
  if (v === "claims") renderClaims();
  if (v === "arena") { refreshArenaModels(); arenaScore(); renderElo(); }
  if (v === "models") renderCatalog();
  if (v === "settings") renderMemory();
  if (v === "agent") renderAgentStatus();
  if (v === "media") {
    renderMediaResidency(); renderMediaBackends();
    renderMediaWorkflows(); renderMediaJobs();
  }
}
document.querySelectorAll(".nav button, .chips button").forEach((b) => {
  b.addEventListener("click", () => showView(b.dataset.view));
});
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); newThread(); }
});

/* ---------- threads ---------- */
let thread = null;
function newThread() {
  setStatus("idle");
  thread = {id: Date.now(), title: "New chat", msgs: []};
  store.threads.unshift(thread);
  store.save(); renderThreads(); renderMessages();
  $("#hero-hi").textContent = "Hey, let's build something";
}
function renderThreads() {
  const ul = document.getElementById("mylumina-list");
  if (!ul) return;
  const q = ($("#thread-search") || {value: ""}).value.toLowerCase();
  ul.innerHTML = "";
  store.threads.filter((t) => t.title.toLowerCase().includes(q)).slice(0, 50).forEach((t) => {
    const li = document.createElement("li");
    const n = t.msgs.filter((m) => m.role === "user").length;
    const sp = document.createElement("span");
    sp.textContent = t.title.slice(0, 48);
    const meta = document.createElement("small");
    meta.className = "hint"; meta.textContent = n + " msgs";
    li.appendChild(sp); li.appendChild(meta);
    const open = document.createElement("button");
    open.type = "button"; open.textContent = "Open";
    const dup = document.createElement("button");
    dup.type = "button"; dup.textContent = "Duplicate";
    const del = document.createElement("button");
    del.type = "button"; del.textContent = "Delete";
    open.addEventListener("click", (ev) => { ev.stopPropagation(); openThread(t); });
    dup.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const c = JSON.parse(JSON.stringify(t));
      c.id = Date.now(); c.title = t.title + " (copy)";
      store.threads.unshift(c); store.save(); renderThreads();
    });
    del.addEventListener("click", (ev) => {
      ev.stopPropagation();
      store.threads = store.threads.filter((x) => x.id !== t.id);
      if (thread && thread.id === t.id) thread = null;
      store.save(); renderThreads(); renderMessages();
    });
    const exp = document.createElement("button");
    exp.type = "button"; exp.textContent = "Export";
    exp.addEventListener("click", (ev) => { ev.stopPropagation(); exportThread(t); });
    li.appendChild(open); li.appendChild(dup); li.appendChild(exp); li.appendChild(del);
    li.style.cursor = "pointer";
    li.addEventListener("click", () => openThread(t));
    ul.appendChild(li);
  });
  if (!ul.children.length) ul.innerHTML = '<li><span class="hint">No chats yet — say hello.</span></li>';
}
function openThread(t) { thread = t; renderMessages(); showView("chat"); }
on("thread-search", "input", renderThreads);
on("thread-search-form", "submit", (e) => e.preventDefault());
on("thread-new", "click", newThread);
on("home-btn", "click", () => showView("chat"));

/* ---------- messages ---------- */
function renderRich(box, text) {
  box.innerHTML = "";
  const parts = String(text).split(/```/);
  parts.forEach((part, i) => {
    if (i % 2 === 1) {
      const pre = document.createElement("pre");
      const btn = document.createElement("button");
      btn.className = "copybtn"; btn.type = "button"; btn.textContent = "copy";
      btn.addEventListener("click", () => {
        try { navigator.clipboard.writeText(part); btn.textContent = "copied"; } catch (e) {}
        setTimeout(() => { btn.textContent = "copy"; }, 1200);
      });
      pre.appendChild(btn);
      const code = document.createElement("code");
      code.textContent = part.replace(/^\w+\n/, "");
      pre.appendChild(code);
      box.appendChild(pre);
    } else {
      const div = document.createElement("div");
      div.textContent = part;
      box.appendChild(div);
    }
  });
}
function renderMessages() {
  const box = document.getElementById("messages");
  box.innerHTML = "";
  if (!thread || !thread.msgs.length) {
    box.innerHTML = '<div class="empty">Load a model, then say hello.</div>';
    return;
  }
  thread.msgs.forEach((m) => {
    const d = document.createElement("div");
    d.className = "msg " + m.role;
    const body = document.createElement("div");
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = (m.meta || "") + " ";
    const at = new Date(m.at || Date.now());
    const stamp = document.createElement("span");
    stamp.className = "t"; stamp.textContent = at.toTimeString().slice(0, 5);
    meta.appendChild(stamp);
    if (m.role === "assistant") renderRich(body, m.text);
    else body.textContent = m.shown || m.text;
    d.appendChild(body); d.appendChild(meta);
    box.appendChild(d);
  });
  box.scrollTop = box.scrollHeight;
}

/* ---------- composer: attachments + slash + send ---------- */
const attachments = [];
function renderAttachChips() {
  const box = document.getElementById("attach-chips");
  if (!box) return;
  box.innerHTML = "";
  box.classList.toggle("hidden", !attachments.length);
  attachments.forEach((a, i) => {
    const chip = document.createElement("span");
    chip.className = "achip";
    const label = document.createElement("span");
    label.textContent = a.name + " (" + a.text.length + "ch)";
    const x = document.createElement("button");
    x.type = "button";
    x.setAttribute("aria-label", "remove " + a.name);
    x.innerHTML = '<svg class="ic"><use href="#i-x"/></svg>';
    x.addEventListener("click", () => { attachments.splice(i, 1); renderAttachChips(); });
    chip.appendChild(label); chip.appendChild(x);
    box.appendChild(chip);
  });
}
on("attach", "click", () => document.getElementById("attach-file").click());
on("attach-file", "change", async (e) => {
  for (const f of e.target.files) {
    let text = "[binary file: " + f.name + "]";
    try {
      if (f.type.startsWith("text/") || /\.(md|txt|csv|json|js|py|html)$/i.test(f.name)) {
        text = (await f.text()).slice(0, 6000);
      }
    } catch (err) { text = "[unreadable: " + f.name + "]"; }
    attachments.push({name: f.name, text});
  }
  e.target.value = "";
  renderAttachChips();
});
function syncModePick() {
  const mp = document.getElementById("mode-pick");
  if (mp) mp.value = (store.config.max_tokens || 64) > 64 ? "High" : "Instant";
}
function syncConfigUI() {
  const t = document.getElementById("cfg-temp");
  if (t) t.value = store.config.temperature || 0;
  const tv = document.getElementById("temp-v");
  if (tv) tv.textContent = store.config.temperature || 0;
  const mt = document.getElementById("cfg-maxtok");
  if (mt) mt.value = store.config.max_tokens || 64;
  syncModePick();
}
on("mode-pick", "change", (e) => {
  store.config.max_tokens = e.target.value === "High" ? 128 : 48;
  store.save(); syncConfigUI();
  notify(e.target.value === "High" ? "High mode: longer replies." : "Instant mode: short replies.");
});

const COMMANDS = [
  {cmd: "/model", hint: "load a model by id", run: (a) => loadModel(a)},
  {cmd: "/temp", hint: "set temperature 0-1.5", run: (a) => {
    store.config.temperature = Math.max(0, Math.min(1.5, +a || 0));
    store.save(); syncConfigUI(); return "temperature = " + store.config.temperature;
  }},
  {cmd: "/clear", hint: "start a new chat", run: () => { newThread(); return "new chat started"; }},
  {cmd: "/help", hint: "list commands", run: () => COMMANDS.map((c) => c.cmd + " — " + c.hint).join("\n")},
];
let slashSel = 0;
function updateSlashMenu() {
  const inp = document.getElementById("prompt");
  const menu = document.getElementById("slashmenu");
  const v = inp.value;
  if (!v.startsWith("/") || v.includes(" ")) { menu.classList.add("hidden"); return; }
  menu.innerHTML = "";
  slashSel = 0;
  COMMANDS.filter((c) => c.cmd.startsWith(v)).forEach((c, i) => {
    const b = document.createElement("button");
    b.type = "button";
    const bold = document.createElement("b"); bold.textContent = c.cmd;
    const small = document.createElement("small"); small.textContent = c.hint;
    b.appendChild(bold); b.appendChild(small);
    if (i === 0) b.classList.add("sel");
    b.addEventListener("click", () => runCommand(c, v));
    menu.appendChild(b);
  });
  menu.classList.toggle("hidden", !menu.children.length);
}
async function runCommand(c, v) {
  const arg = v.slice(c.cmd.length).trim();
  document.getElementById("slashmenu").classList.add("hidden");
  document.getElementById("prompt").value = "";
  const out = await c.run(arg);
  if (out) notify(out);
}
on("prompt", "input", (e) => {
  document.getElementById("charcount").textContent = e.target.value.length;
  updateSlashMenu();
  e.target.style.height = "auto";
  e.target.style.height = Math.min(140, e.target.scrollHeight) + "px";
});
on("prompt", "keydown", (e) => {
  const menu = document.getElementById("slashmenu");
  const open = menu && !menu.classList.contains("hidden") && menu.children.length;
  if (open && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
    e.preventDefault();
    const n = menu.children.length;
    slashSel = e.key === "ArrowDown" ? (slashSel + 1) % n : (slashSel - 1 + n) % n;
    Array.from(menu.children).forEach((c, i) => c.classList.toggle("sel", i === slashSel));
    return;
  }
  if (open && (e.key === "Enter" || e.key === "Tab")) {
    e.preventDefault();
    const btn = menu.children[Math.min(slashSel, menu.children.length - 1)];
    if (btn) btn.click();
    return;
  }
  if (e.key === "Enter" && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
    e.preventDefault();
    document.getElementById("composer").requestSubmit();
    return;
  }
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    document.getElementById("composer").requestSubmit();
  }
  if (e.key === "Escape") menu.classList.add("hidden");
});

document.getElementById("composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const inp = document.getElementById("prompt");
  const text = inp.value.trim();
  if (!text) return;
  if (!thread) newThread();
  if (thread.msgs.length === 0) {
    thread.title = text.slice(0, 30);
    $("#hero-hi").textContent = text.slice(0, 60);
  }
  const attached = attachments.splice(0, attachments.length);
  renderAttachChips();
  let full_text = text;
  if (attached.length) {
    full_text += "\n\n[attachments]\n" + attached.map((a) => "--- " + a.name + " ---\n" + a.text).join("\n");
  }
  const shown = attached.length
    ? text + "\n[" + attached.length + " attachment(s): " + attached.map((a) => a.name).join(", ").slice(0, 120) + "]"
    : text;
  thread.msgs.push({role: "user", text: full_text, shown, at: Date.now()});
  inp.value = "";
  inp.style.height = "auto";
  document.getElementById("charcount").textContent = "0";
  document.getElementById("slashmenu").classList.add("hidden");
  renderMessages(); renderThreads();
  setStatus("thinking");
  const box = document.getElementById("messages");
  const d = document.createElement("div");
  d.className = "msg assistant";
  d.innerHTML = "<div class='tdots'><span></span><span></span><span></span></div>";
  box.appendChild(d);
  let r;
  try {
    const body = {messages: thread.msgs, max_tokens: store.config.max_tokens,
                  temperature: store.config.temperature};
    if (store.config.ground) body.ground = {k: 3};
    if (store.config.skill) body.skill_auto = true;
    r = await api("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body)});
  } catch (e) {
    d.firstChild.textContent = "Backend unreachable — could not send that message.";
    thread.msgs.push({role: "assistant", text: d.firstChild.textContent, meta: "", at: Date.now()});
    store.save(); renderThreads();
    settleStatus("error", 2500);
    return;
  }
  setStatus("streaming");
  const full = (r.flair ? r.flair + "\n\n" : "") + (r.text || r.error || "(model returned no text)");
  d.firstChild.textContent = "";
  for (const ch of full) { d.firstChild.textContent += ch; await sleep(6); }
  const meta = d.querySelector(".meta");
  if (meta) {
    const bits = [];
    if (r.tokens_per_second) bits.push(r.tokens_per_second + " tok/s");
    if (r.model) bits.push(r.model);
    if (r.refused) bits.push("refused: no supporting evidence");
    if (r.grounding) {
      const g = r.grounding;
      const tot = (g.total != null) ? g.total : ((g.claims || []).length);
      const uns = g.unsupported || 0;
      bits.push(tot ? ("grounded " + (tot - uns) + "/" + tot + " claims")
                    : ("evidence: " + (g.gate || "checked")));
    }
    if (r.skill_auto && r.skill_auto.injected) bits.push("skill: " + r.skill_auto.id);
    if (r.prefill_reused != null && r.prefill_computed != null) {
      const tot2 = r.prefill_reused + r.prefill_computed;
      if (tot2) bits.push("cache " + Math.round(100 * r.prefill_reused / tot2) + "%");
    }
    meta.textContent = bits.join(" · ");
  }
  renderRich(d.firstChild, full);
  thread.msgs.push({role: "assistant", text: full, meta: meta ? meta.textContent : "", at: Date.now()});
  store.save(); renderThreads();
  settleStatus(r.error ? "error" : "success", r.error ? 2500 : 1600);
});

/* ---------- models ---------- */
let loading = false;
async function refreshLibrary() {
  let lib;
  try { lib = await api("/api/library"); }
  catch (e) { lib = {items: []}; }
  const grid = document.getElementById("lib-grid");
  const pick = document.getElementById("model-pick");
  grid.innerHTML = "";
  pick.innerHTML = '<option value="">Select a model…</option>';
  if (!lib.items || !lib.items.length) {
    grid.innerHTML = '<div class="card"><h3>No models found</h3><p class="hint">Download one in Discover, or set LUMINA_MODELS.</p></div>';
    setLoaded(store.loaded);
    return;
  }
  const maxmb = Math.max(1, ...lib.items.map((m) => m.size_mb || 0));
  lib.items.forEach((m) => {
    const o = document.createElement("option");
    o.value = m.id; o.textContent = m.id;
    pick.appendChild(o);
    const d = document.createElement("div");
    d.className = "mcard";
    d.innerHTML = "<h4></h4><span class='fmt'></span><div class='sizebar'><i></i></div><small></small><div class='row'></div>";
    d.querySelector("h4").textContent = m.id;
    d.querySelector(".fmt").textContent = m.format || "unknown";
    d.querySelector(".sizebar i").style.width = Math.max(3, Math.round(100 * (m.size_mb || 0) / maxmb)) + "%";
    d.querySelector("small").textContent = m.size_mb + " MB · " + m.path;
    const row = d.querySelector(".row");
    const b1 = document.createElement("button");
    b1.type = "button"; b1.textContent = "Load"; b1.className = "primary";
    b1.addEventListener("click", () => loadModel(m.id));
    const b2 = document.createElement("button");
    b2.type = "button"; b2.textContent = "Chat";
    b2.addEventListener("click", () => showView("chat"));
    row.appendChild(b1); row.appendChild(b2);
    grid.appendChild(d);
  });
  const st = await api("/api/loaded");
  setLoaded(st.loaded);
}
const QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"];
function setLoaded(id) {
  store.loaded = id; store.save();
  document.getElementById("load-label").textContent = id || "no model";
  document.getElementById("load-dot").classList.toggle("on", !!id);
}
async function loadModel(id) {
  if (!id) { notify("Pick a model first."); return; }
  if (loading) return;
  loading = true;
  const btn = document.getElementById("model-load");
  if (btn) btn.disabled = true;
  document.getElementById("load-label").textContent = "loading…";
  setStatus("working");
  let ok = false;
  try {
    const r = await api("/api/load", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id})});
    if (r.loaded && r.cloud) { setLoaded(r.loaded); notify("Cloud model live: " + r.loaded + "."); ok = true; }
    else if (r.loaded) { setLoaded(r.loaded); notify("Loaded " + r.loaded + " — " + r.params_m + "M params."); ok = true; }
    else notify("Load failed: " + (r.error || "unknown"));
  } catch (e) {
    notify("Load failed: backend unreachable.");
    document.getElementById("load-label").textContent = store.loaded || "no model";
  } finally {
    loading = false;
    if (btn) btn.disabled = false;
    settleStatus(ok ? "success" : "error", ok ? 1600 : 2500);
  }
}
on("model-load", "click", () => { const v = $("#model-pick").value; if (v) loadModel(v); });
on("model-unload", "click", async () => {
  try { await api("/api/unload", {method: "POST"}); } catch (e) {}
  setLoaded(null);
});
document.getElementById("hf-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#hf-q").value;
  let r;
  try { r = await api("/api/hf-search?q=" + encodeURIComponent(q)); }
  catch (err) { r = {items: []}; }
  const grid = document.getElementById("hf-grid");
  grid.innerHTML = "";
  (r.items || []).forEach((m) => {
    const d = document.createElement("div");
    d.className = "mcard";
    d.innerHTML = "<h4></h4><small></small><div class='quantrow'></div><div class='row'></div>";
    d.querySelector("h4").textContent = m.id;
    d.querySelector("small").textContent = "likes " + m.likes + " · downloads " + m.downloads + " · " + (m.tags || []).join(" ");
    const qr = d.querySelector(".quantrow");
    QUANTS.forEach((q2) => {
      const qb = document.createElement("button");
      qb.type = "button"; qb.textContent = q2;
      qb.addEventListener("click", () => startDownload(m.id));
      qr.appendChild(qb);
    });
    const b = document.createElement("button");
    b.type = "button"; b.textContent = "Download base"; b.className = "primary";
    b.addEventListener("click", () => startDownload(m.id));
    d.querySelector(".row").appendChild(b);
    grid.appendChild(d);
  });
  if (!(r.items || []).length) grid.innerHTML = "<p class='hint'>No results.</p>";
});
async function startDownload(repo) {
  if (!repo) return;
  const el = document.getElementById("dl-status");
  el.textContent = repo + ": queued…";
  try {
    await api("/api/download", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({repo})});
  } catch (e) { el.textContent = repo + ": backend unreachable"; return; }
  const tick = setInterval(async () => {
    try {
      const s = await api("/api/dl-progress?repo=" + encodeURIComponent(repo));
      el.textContent = repo + ": " + s.state + " " + (s.pct || 0) + "%";
      if (s.state === "done" || s.state === "error") { clearInterval(tick); refreshLibrary(); }
    } catch (e) { clearInterval(tick); el.textContent = repo + ": progress unavailable"; }
  }, 1500);
}
document.getElementById("dl-form").addEventListener("submit", (e) => {
  e.preventDefault();
  startDownload($("#dl-repo").value.trim());
});

/* ---------- docs ---------- */
let docFilter = "all";
document.getElementById("doc-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const v = $("#doc-input").value.trim();
  if (!v) return;
  let text = v;
  if (/^https?:\/\//.test(v)) {
    try {
      const r = await api("/api/fetch-url?url=" + encodeURIComponent(v));
      text = (r.text || "").slice(0, 4000);
    } catch (err) { notify("Fetch failed — backend unreachable."); return; }
  }
  const kind = /^https?:\/\//.test(v) ? "site" : (/[,;\n]/.test(v) ? "sheet" : "doc");
  store.docs.push({src: v.slice(0, 80), text, kind});
  store.docs.forEach((d) => { if (!d.kind) d.kind = "doc"; });
  store.save(); renderDocs();
  try {
    api("/api/rag/add", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text: text, source: v.slice(0, 80)})}).catch(() => {});
  } catch (e) {}
  $("#doc-input").value = "";
});
function renderDocs() {
  const ul = document.getElementById("doc-list");
  if (!ul) return;
  ul.innerHTML = "";
  const items = store.docs.filter((d) => docFilter === "all" || (d.kind || "doc") === docFilter);
  if (!items.length) { ul.innerHTML = "<li><span class='hint'>Nothing here yet — paste text, a URL, or CSV rows above.</span></li>"; return; }
  items.forEach((d) => {
    const idx = store.docs.indexOf(d);
    const li = document.createElement("li");
    const sp = document.createElement("span");
    sp.textContent = "[" + (d.kind || "doc") + "] " + d.src + " (" + d.text.length + " chars)";
    li.appendChild(sp);
    const b = document.createElement("button");
    b.type = "button"; b.textContent = "Remove";
    b.addEventListener("click", () => { store.docs.splice(idx, 1); store.save(); renderDocs(); });
    li.appendChild(b);
    ul.appendChild(li);
  });
}
document.querySelectorAll("#doc-filter button").forEach((b) => {
  b.addEventListener("click", () => {
    docFilter = b.dataset.f;
    document.querySelectorAll("#doc-filter button").forEach((x) => x.classList.toggle("on", x === b));
    renderDocs();
  });
});

/* ---------- slides ---------- */
const slides = [];
function renderStrip() {
  const strip = document.getElementById("slide-strip");
  strip.innerHTML = "";
  if (!slides.length) { strip.innerHTML = "<div class='hint'>No slides yet — add one above.</div>"; return; }
  slides.forEach((t, i) => {
    const d = document.createElement("div");
    const sp = document.createElement("span");
    sp.textContent = (i + 1) + ". " + t;
    const x = document.createElement("button");
    x.type = "button"; x.innerHTML = '<svg class="ic"><use href="#i-x"/></svg>';
    x.setAttribute("aria-label", "delete slide " + (i + 1));
    x.addEventListener("click", () => { slides.splice(i, 1); renderStrip(); drawSlide(Math.min(i, slides.length - 1)); });
    d.appendChild(sp); d.appendChild(x);
    strip.appendChild(d);
  });
}
function drawSlide(i = 0) {
  const c = document.getElementById("stage");
  const g = c.getContext("2d");
  g.fillStyle = "#14161a"; g.fillRect(0, 0, 640, 360);
  g.fillStyle = "#fff"; g.font = "42px sans-serif"; g.textAlign = "center";
  g.fillText(slides[i] || "No slides yet", 320, 180);
  g.fillStyle = "#6a8dff"; g.font = "20px sans-serif";
  g.fillText("Lumina Slides · " + (i + 1) + "/" + (slides.length || 1), 320, 320);
}
document.getElementById("slide-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const t = $("#slide-title").value.trim();
  if (!t) return;
  slides.push(t);
  $("#slide-title").value = "";
  renderStrip();
  drawSlide(slides.length - 1);
});
on("slides-clear", "click", () => { slides.length = 0; renderStrip(); drawSlide(); });
on("video-export", "click", () => {
  if (!slides.length) { $("#video-status").textContent = "add a slide first"; return; }
  const c = document.getElementById("stage");
  const stream = c.captureStream(30);
  const rec = new MediaRecorder(stream, {mimeType: "video/webm"});
  const chunks = [];
  rec.ondataavailable = (e) => chunks.push(e.data);
  rec.onstop = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob(chunks, {type: "video/webm"}));
    a.download = "lumina-slides.webm";
    a.click();
    $("#video-status").textContent = "exported";
  };
  let i = 0;
  drawSlide(0); rec.start();
  const iv = setInterval(() => {
    i++;
    if (i >= slides.length) { clearInterval(iv); rec.stop(); return; }
    drawSlide(i);
  }, 1500);
  $("#video-status").textContent = "recording…";
});

/* ---------- swarm / research / tasks / code / claw / work ---------- */
document.getElementById("swarm-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const p = $("#swarm-prompt").value.trim();
  if (!p) return;
  const mk = (t) => api("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({messages: [{role: "user", text: p}], max_tokens: 48, temperature: t})});
  $("#swarm-a").textContent = "running…";
  $("#swarm-b").textContent = "running…";
  try {
    const [a, b] = await Promise.all([mk(0.0), mk(0.7)]);
    $("#swarm-a").textContent = a.text || a.error || "(empty — load a model?)";
    $("#swarm-b").textContent = b.text || b.error || "(empty — load a model?)";
  } catch (e) {
    $("#swarm-a").textContent = "backend unreachable";
    $("#swarm-b").textContent = "backend unreachable";
  }
});
document.getElementById("res-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#res-q").value.trim();
  const out = document.getElementById("res-out");
  if (!q) return;
  out.innerHTML = '<div class="skel"></div><div class="skel" style="margin-top:8px"></div>';
  try {
    const h = await api("/api/hf-search?q=" + encodeURIComponent(q));
    const items = (h.items || []).slice(0, 3);
    out.innerHTML = "";
    if (!items.length) { out.innerHTML = "<p class='hint'>" + (h.error || "no sources found") + "</p>"; return; }
    for (const m of items) {
      const card = document.createElement("div");
      card.className = "card";
      const h3 = document.createElement("h3"); h3.textContent = m.id;
      const p = document.createElement("p");
      let sents = "";
      try {
        const f = await api("/api/fetch-url?url=" + encodeURIComponent("https://huggingface.co/" + m.id));
        sents = (f.text || "").split(/(?<=[.!?])\s+/).slice(0, 4).join(" ").slice(0, 600) || "(no extractable text)";
      } catch (err) { sents = "(fetch failed — offline?)"; }
      p.textContent = sents;
      card.appendChild(h3); card.appendChild(p);
      out.appendChild(card);
    }
  } catch (err) {
    out.innerHTML = "<p class='hint'>research failed: backend unreachable</p>";
  }
});
function renderTasks() {
  const ul = document.getElementById("task-list");
  if (!ul) return;
  ul.innerHTML = "";
  if (!store.tasks.length) { ul.innerHTML = "<li><span class='hint'>No saved prompts yet.</span></li>"; return; }
  store.tasks.forEach((t, i) => {
    const li = document.createElement("li");
    const sp = document.createElement("span");
    sp.textContent = t;
    li.appendChild(sp);
    const b = document.createElement("button");
    b.type = "button"; b.textContent = "Send to chat";
    b.addEventListener("click", () => { showView("chat"); $("#prompt").value = t; $("#prompt").focus(); });
    const del = document.createElement("button");
    del.type = "button"; del.textContent = "Delete";
    del.addEventListener("click", () => { store.tasks.splice(i, 1); store.save(); renderTasks(); });
    li.appendChild(b); li.appendChild(del);
    ul.appendChild(li);
  });
}
document.getElementById("task-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const v = $("#task-text").value.trim();
  if (!v) return;
  store.tasks.push(v); store.save(); renderTasks();
  $("#task-text").value = "";
});
document.getElementById("code-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#code-q").value.trim();
  if (!q) return;
  $("#code-out").textContent = "running…";
  try {
    const r = await api("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({messages: [{role: "system", content: "You are a senior code reviewer. Be concise."}, {role: "user", text: q}],
        max_tokens: 64, temperature: 0})});
    $("#code-out").textContent = r.text || r.error || "(empty — load a model?)";
  } catch (e2) { $("#code-out").textContent = "backend unreachable"; }
});
document.getElementById("term-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const c = $("#term-cmd").value.trim();
  if (!c) return;
  $("#term-out").textContent = "running…";
  try {
    const r = await api("/api/exec", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({cmd: c})});
    $("#term-out").textContent = r.out || r.error || "(empty)";
  } catch (e2) { $("#term-out").textContent = "backend unreachable"; }
});
document.getElementById("convert-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  $("#convert-cmd").textContent =
    "python3 scripts/lumina_convert.py --input \"" + fd.get("input") + "\" --output \"" + fd.get("output") + "\" --rung " + fd.get("rung");
});
on("convert-run", "click", async () => {
  const fd = new FormData(document.getElementById("convert-form"));
  const st = $("#convert-status");
  const box = document.getElementById("convert-result");
  const btn = document.getElementById("convert-run");
  btn.disabled = true;
  st.textContent = "converting with the real codec…";
  setStatus("working");
  try {
    const r = await api("/api/convert", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(Object.assign({input: fd.get("input"), output: fd.get("output"), rung: fd.get("rung")},
        (() => { const mf = new FormData(document.getElementById("moe-form"));
                 const e = +mf.get("experts") || 0;
                 return e > 0 ? {moe: {experts: e, top_k: +mf.get("top_k") || 2}} : {}; })()))});
    if (r.ok) {
      st.textContent = "done in " + r.seconds + "s";
      box.classList.remove("hidden");
      box.innerHTML = "";
      const rows = [
        ["input format", r.input_format], ["output", r.mb + " MB"],
        ["effective bpp", r.effective_bpp], ["params", r.params_m + "M (fp32 " + r.fp32_mb + " MB)"],
        ["matrices", r.quantized + "/" + r.tensors + " quantized, " + r.escaped + " escaped"],
        ["calibrated", r.calibrated + " tensors"], ["weight error", "mean " + r.mean_relW + " · max " + r.max_relW],
        ["time", r.seconds + "s"]];
      rows.forEach(([k, v]) => {
        const d = document.createElement("div");
        d.className = "mono";
        d.textContent = k + ": " + v;
        box.appendChild(d);
      });
      if (r.virtual_moe) {
        const vm = document.createElement("div");
        vm.className = "mono";
        vm.textContent = "virtual MoE: " + JSON.stringify(r.virtual_moe.rel_func_err)
          + " (" + r.virtual_moe.signatures + ")";
        box.appendChild(vm);
      }
      settleStatus("success", 1600);
      refreshLibrary();
    } else {
      st.textContent = r.error || "conversion failed";
      box.classList.add("hidden");
      settleStatus("error", 2200);
    }
  } catch (e) {
    st.textContent = "conversion failed: backend unreachable";
    settleStatus("error", 2200);
  } finally { btn.disabled = false; }
});
document.getElementById("inspect-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const box = document.getElementById("inspect-result");
  box.classList.remove("hidden");
  box.innerHTML = "";
  const say = (t) => {
    const d = document.createElement("div");
    d.className = "mono"; d.textContent = t;
    box.appendChild(d);
  };
  say("inspecting…");
  try {
    const r = await api("/api/inspect", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({path: fd.get("path")})});
    box.innerHTML = "";
    if (r.error) { say("error: " + r.error); return; }
    const rows = [
      ["path", r.path], ["format", r.format], ["size", r.mb + " MB (" + r.bytes + " bytes)"]];
    if (r.rung) rows.push(["rung", r.rung], ["matrices", r.matrices],
      ["recorded mean relW", r.recorded_mean_relW], ["model", r.model]);
    if (r.source_dir) rows.push(["converted from", r.source_dir]);
    if (r.tensors) rows.push(["tensors", r.tensors], ["params", r.params_m + "M (fp32 " + r.fp32_mb + " MB)"]);
    if (r.note) rows.push(["note", r.note]);
    rows.forEach(([k, v]) => say(k + ": " + v));
  } catch (err) {
    box.innerHTML = "";
    say("inspect failed: backend unreachable");
  }
});
document.getElementById("hw-run").addEventListener("click", async () => {
  const box = document.getElementById("hw-out");
  box.classList.remove("hidden");
  box.innerHTML = "";
  const say = (t) => {
    const d = document.createElement("div");
    d.className = "mono"; d.textContent = t;
    box.appendChild(d);
  };
  say("probing…");
  try {
    const r = await api("/api/hardware");
    box.innerHTML = "";
    const inv = r.inventory || {};
    say("CPUs: " + inv.cpu_count + " · free RAM: " + inv.mem_available_mb + " MB");
    const tn = (inv.torch && inv.torch.version) || "absent";
    const cn = (inv.torch && inv.torch.cuda) ? ((inv.cuda && inv.cuda.name) || "yes") : "no";
    say("torch: " + tn + " · CUDA: " + cn);
    say("thread policy: BLAS=" + r.thread_policy.blas_threads
      + " pool=" + r.thread_policy.pool_workers);
    const q = r.qwen38_recipe || {};
    say("Qwen3.8 recipe (" + (q.hardware || "") + "):");
    Object.entries(q.scenarios || {}).forEach(([k, v]) => {
      say("  " + k + ": " + Math.round(v.tok_per_s) + " tok/s ("
        + (v.weight_gb_active + v.extra_gb_per_token).toFixed(2) + " GB/tok)");
    });
    const vd = document.createElement("div");
    vd.className = "hint"; vd.textContent = q.verdict || "";
    box.appendChild(vd);
  } catch (err) {
    box.innerHTML = "";
    say("hardware probe failed: backend unreachable");
  }
});
document.getElementById("bench-run").addEventListener("click", async () => {
  const btn = document.getElementById("bench-run");
  btn.disabled = true;
  $("#bench-out").textContent = "running…";
  try {
    const r = await api("/api/bench-tiny");
    if (r.error) { $("#bench-out").textContent = "bench failed: " + r.error; }
    else $("#bench-out").textContent = "tiny proxy: " + r.tokens_per_second + " tok/s in " + r.seconds + "s";
    const t = await api("/api/targets");
    const bt = document.getElementById("bench-targets");
    bt.innerHTML = "";
    (t.models || []).forEach((m) => {
      const card = document.createElement("div");
      card.className = "card";
      const h3 = document.createElement("h3"); h3.textContent = m.label;
      const dv = document.createElement("div");
      dv.className = "mono"; dv.textContent = "target " + m.target_tps + " tok/s";
      card.appendChild(h3); card.appendChild(dv);
      if (m.note) {
        const nt = document.createElement("div");
        nt.className = "hint"; nt.textContent = m.note;
        card.appendChild(nt);
      }
      bt.appendChild(card);
    });
    if (t.measured && t.measured.cases) {
      const mc = document.createElement("div");
      mc.className = "card";
      const h3 = document.createElement("h3"); h3.textContent = "Measured (proxy rig)";
      mc.appendChild(h3);
      t.measured.cases.slice(0, 4).forEach((c) => {
        const dv = document.createElement("div");
        dv.className = "mono"; dv.textContent = c.label + ": " + c.warm_tok_per_s + " tok/s";
        mc.appendChild(dv);
      });
      const src = document.createElement("div");
      src.className = "hint"; src.textContent = t.measured.source || "";
      mc.appendChild(src);
      bt.appendChild(mc);
    }
  } catch (err) {
    $("#bench-out").textContent = "bench failed: backend unreachable";
  } finally { btn.disabled = false; }
});

/* ---------- arena ---------- */
const arenaState = {busy: false, text: ""};
async function refreshArenaModels() {
  let lib;
  try { lib = await api("/api/library"); } catch (e) { lib = {items: []}; }
  const a = document.getElementById("arena-a"), b = document.getElementById("arena-b");
  if (!a || !b) return;
  const opts = (lib.items || []).map((m) => "<option value=\"" + m.id + "\">" + m.id + " (" + m.format + ")</option>").join("");
  a.innerHTML = opts; b.innerHTML = opts;
  if ((lib.items || []).length > 1) b.selectedIndex = 1;
}
async function arenaRunOne(id, outEl, metaEl) {
  let r0;
  try {
    r0 = await api("/api/load", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id})});
  } catch (e) { outEl.textContent = "backend unreachable"; return null; }
  if (!r0.loaded && !r0.cloud) { outEl.textContent = r0.error || "load failed"; return null; }
  let r;
  try {
    r = await api("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({messages: [{role: "user", text: arenaState.text}], max_tokens: 64, temperature: 0})});
  } catch (e) { outEl.textContent = "backend unreachable"; return null; }
  const txt = r.text || r.error || "(model returned no text)";
  outEl.textContent = txt;
  metaEl.textContent = [r.tokens_per_second ? r.tokens_per_second + " tok/s" : "",
    r.wall_s ? r.wall_s + "s" : "", "model: " + (r.model || id)].filter(Boolean).join(" · ");
  return txt;
}
async function arenaRun() {
  if (arenaState.busy) return;
  const q = $("#arena-q").value.trim();
  if (!q) return;
  arenaState.busy = true;
  arenaState.text = q;
  const btn = document.getElementById("arena-run");
  btn.disabled = true; btn.textContent = "Running A…";
  $("#arena-out-a").textContent = "loading model A…";
  $("#arena-out-b").textContent = "waiting…";
  ["vote-a", "vote-tie", "vote-b"].forEach((id) => document.getElementById(id).disabled = true);
  setStatus("working");
  let okA = false, okB = false;
  try {
    okA = (await arenaRunOne($("#arena-a").value, $("#arena-out-a"), $("#arena-meta-a"))) !== null;
    btn.textContent = "Running B…";
    okB = (await arenaRunOne($("#arena-b").value, $("#arena-out-b"), $("#arena-meta-b"))) !== null;
    if (okA && okB) {
      ["vote-a", "vote-tie", "vote-b"].forEach((id) => document.getElementById(id).disabled = false);
      settleStatus("success", 1600);
    } else {
      $("#arena-score").textContent = "one or both sources failed to load — pick two loadable models";
      settleStatus("error", 2200);
    }
  } catch (e) {
    $("#arena-out-b").textContent = "backend unreachable";
    settleStatus("error", 2200);
  } finally {
    arenaState.busy = false; btn.disabled = false; btn.textContent = "Run both";
  }
}
function arenaScore() {
  let sc;
  try { sc = JSON.parse(localStorage.getItem("lum-arena") || "{}"); } catch (e) { sc = {}; }
  const el = document.getElementById("arena-score");
  if (el) el.textContent = Object.entries(sc).map(([k, v]) => k + ": " + v + " win(s)").join(" · ") || "no votes yet";
}
function arenaVote(winner) {
  if (winner >= 0) {
    try {
      api("/api/arena/elo", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({a: $("#arena-a").value, b: $("#arena-b").value,
                              outcome: winner === 0 ? "a" : "b"})}).then(renderElo).catch(() => {});
    } catch (e) {}
    let sc;
    try { sc = JSON.parse(localStorage.getItem("lum-arena") || "{}"); } catch (e) { sc = {}; }
    const k = winner === 0 ? $("#arena-a").value : $("#arena-b").value;
    sc[k] = (sc[k] || 0) + 1;
    try { localStorage.setItem("lum-arena", JSON.stringify(sc)); } catch (e) {}
  }
  ["vote-a", "vote-tie", "vote-b"].forEach((id) => document.getElementById(id).disabled = true);
  arenaScore();
}
on("arena-run", "click", arenaRun);
document.getElementById("arena-form").addEventListener("submit", (e) => { e.preventDefault(); arenaRun(); });
on("vote-a", "click", () => arenaVote(0));
on("vote-b", "click", () => arenaVote(1));
on("vote-tie", "click", () => {
  arenaVote(-1);
  try {
    api("/api/arena/elo", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({a: $("#arena-a").value, b: $("#arena-b").value,
                            outcome: "tie"})}).then(renderElo).catch(() => {});
  } catch (e) {}
});

/* ---------- graph ---------- */
async function renderGraphStats() {
  const el = document.getElementById("graph-stats");
  if (!el) return;
  let st;
  try { st = await api("/api/graph/stats"); } catch (e) { st = {nodes: 0, edges: 0}; }
  el.innerHTML = "";
  [["nodes", st.nodes], ["edges", st.edges], ["python", (st.by_kind || {}).py || 0], ["js", (st.by_kind || {}).js || 0]].forEach(([k, v]) => {
    const d = document.createElement("div");
    d.className = "mcard";
    d.innerHTML = "<h4></h4><small></small>";
    d.querySelector("h4").textContent = String(v);
    d.querySelector("small").textContent = k;
    el.appendChild(d);
  });
}
async function renderGraphList(q) {
  const ul = document.getElementById("graph-list");
  if (!ul) return;
  let r;
  try { r = await api("/api/graph/nodes?q=" + encodeURIComponent(q || "")); } catch (e) { r = {items: []}; }
  ul.innerHTML = "";
  (r.items || []).forEach((n) => {
    const li = document.createElement("li");
    li.className = "graph-node";
    li.innerHTML = "<span></span><small class='hint'></small>";
    li.firstChild.textContent = n.id;
    li.lastChild.textContent = n.kind + " · " + n.lines + " lines";
    li.addEventListener("click", async () => {
      let d;
      try { d = await api("/api/graph/node?id=" + encodeURIComponent(n.id)); } catch (e) { return; }
      if (!d || !d.node) return;
      const box = document.getElementById("graph-detail");
      box.classList.remove("hidden");
      box.innerHTML = "";
      const h3 = document.createElement("h3"); h3.textContent = d.node.id;
      const p1 = document.createElement("p"); p1.className = "hint";
      p1.textContent = "imports: " + (d.imports.join(", ") || "(none)");
      const p2 = document.createElement("p"); p2.className = "hint";
      p2.textContent = "imported by: " + (d.imported_by.join(", ") || "(none)");
      box.appendChild(h3); box.appendChild(p1); box.appendChild(p2);
    });
    ul.appendChild(li);
  });
  if (!(r.items || []).length) ul.innerHTML = "<li><span class='hint'>No files match.</span></li>";
}
document.getElementById("graph-search-form").addEventListener("submit", (e) => {
  e.preventDefault(); renderGraphList($("#graph-q").value);
});
on("graph-q", "input", () => renderGraphList($("#graph-q").value));

/* ---------- claims ---------- */
async function renderClaims() {
  const el = document.getElementById("claims-list");
  if (!el) return;
  let r;
  try { r = await api("/api/claims"); } catch (e) { r = {items: []}; }
  el.innerHTML = "";
  (r.items || []).forEach((c) => {
    const d = document.createElement("div");
    d.className = "card claim-card " + (c.verified ? "verified" : "partial");
    const chip = document.createElement("span");
    chip.className = "claim-chip";
    chip.textContent = c.verified ? "VERIFIED" : "PARTIAL / DEFERRED";
    const h3 = document.createElement("h3");
    h3.textContent = c.section;
    const p = document.createElement("p");
    p.style.fontSize = "13px";
    p.textContent = c.text;
    d.appendChild(chip); d.appendChild(h3); d.appendChild(p);
    el.appendChild(d);
  });
  if (!(r.items || []).length) el.innerHTML = "<p class='hint'>No claims found.</p>";
}

/* ---------- memory ---------- */
async function renderMemory() {
  const statsEl = document.getElementById("memory-stats");
  const listEl = document.getElementById("memory-list");
  if (!statsEl) return;
  let st, li;
  try { st = await api("/api/memory/stats"); li = await api("/api/memory/list"); }
  catch (e) { st = {tiers: []}; li = {items: []}; }
  statsEl.innerHTML = "";
  (st.tiers || []).forEach((t) => {
    const d = document.createElement("div");
    d.className = "mcard";
    d.innerHTML = "<h4></h4><small></small>";
    d.querySelector("h4").textContent = t.name;
    d.querySelector("small").textContent = t.where + " — " + t.holds;
    statsEl.appendChild(d);
  });
  listEl.innerHTML = "";
  (li.items || []).forEach((t) => {
    const row = document.createElement("li");
    const sp = document.createElement("span");
    sp.textContent = t.title + " (" + t.msgs + " msgs)";
    const restore = document.createElement("button");
    restore.type = "button"; restore.textContent = "Restore";
    restore.addEventListener("click", async () => {
      try {
        const r = await api("/api/memory/restore", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id: t.id})});
        if (r.thread) {
          const th = r.thread;
          th.id = Date.now();
          store.threads.unshift(th);
          thread = th; store.save();
          renderThreads(); renderMessages();
          notify("Thread restored from archive.");
        }
      } catch (e) { notify("Restore failed: backend unreachable."); }
    });
    const forget = document.createElement("button");
    forget.type = "button"; forget.textContent = "Forget";
    forget.addEventListener("click", async () => {
      try {
        await api("/api/memory/forget", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id: t.id})});
        renderMemory();
      } catch (e) { notify("Forget failed."); }
    });
    row.appendChild(sp); row.appendChild(restore); row.appendChild(forget);
    listEl.appendChild(row);
  });
  if (!(li.items || []).length) listEl.innerHTML = "<li><span class='hint'>Archive is empty.</span></li>";
}
on("mem-refresh", "click", renderMemory);
on("mem-archive", "click", async () => {
  if (!thread || !thread.msgs.length) { notify("Nothing to archive — open a chat first."); return; }
  try {
    const r = await api("/api/memory/archive", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({thread})});
    if (r.archived) {
      store.threads = store.threads.filter((t) => t.id !== thread.id);
      thread = store.threads[0] || null;
      store.save(); renderThreads(); renderMessages(); renderMemory();
      notify("Archived to server memory.");
    } else notify("Archive failed: " + (r.error || "unknown"));
  } catch (e) { notify("Archive failed: backend unreachable."); }
});

/* ---------- cloud ---------- */
const CLOUD_PRESETS = {openai: "https://api.openai.com/v1", openrouter: "https://openrouter.ai/api/v1", custom: ""};
async function loadCloudForm() {
  let cfg;
  try { cfg = await api("/api/config"); } catch (e) { return; }
  const cl = (cfg && cfg.cloud) || {};
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
  const en = document.getElementById("cloud-enabled");
  if (en) en.checked = Boolean(cl.enabled);
  set("cloud-provider", cl.provider || "custom");
  set("cloud-url", cl.base_url || "");
  set("cloud-model", cl.model || "");
  const key = document.getElementById("cloud-key");
  if (key) { key.value = ""; key.placeholder = cl.key_saved ? "(saved — leave blank to keep)" : "sk-…"; }
}
on("cloud-provider", "change", (e) => {
  const url = document.getElementById("cloud-url");
  if (url && CLOUD_PRESETS[e.target.value] && !url.value) url.value = CLOUD_PRESETS[e.target.value];
});
on("cloud-save", "click", async () => {
  const st = document.getElementById("cloud-status");
  const payload = {cloud: {
    enabled: document.getElementById("cloud-enabled").checked,
    provider: document.getElementById("cloud-provider").value,
    base_url: document.getElementById("cloud-url").value.trim(),
    model: document.getElementById("cloud-model").value.trim(),
  }};
  const key = document.getElementById("cloud-key").value;
  if (key) payload.cloud.api_key = key;
  try {
    await api("/api/config", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    if (st) st.textContent = "Saved.";
    document.getElementById("cloud-key").value = "";
    refreshLibrary(); loadCloudForm();
  } catch (e) { if (st) st.textContent = "Save failed: backend unreachable."; }
});
on("cloud-test", "click", async () => {
  const st = document.getElementById("cloud-status");
  if (st) st.textContent = "Testing…";
  try {
    const r = await api("/api/cloud-test", {method: "POST"});
    if (st) st.textContent = r.ok ? ("Connection OK — " + (r.text || "")).slice(0, 160) : ("Test failed: " + (r.error || "unknown"));
  } catch (e) { if (st) st.textContent = "Test failed: backend unreachable."; }
});

/* ---------- toast ---------- */
function notify(t, ms = 4200) {
  const b = document.getElementById("toast");
  if (!b) return;
  b.textContent = t;
  b.classList.remove("hidden");
  clearTimeout(notify._t);
  notify._t = setTimeout(() => b.classList.add("hidden"), ms);
}
on("plan-pill", "click", async () => {
  try {
    const st = await api("/api/status");
    notify("Backend live. Formats: " + (st.formats || []).join(", ") + ".");
  } catch (e) { notify("Backend unreachable — running on mock data."); }
});
on("drawer-btn", "click", () => document.body.classList.toggle("drawer-open"));
on("scrim", "click", () => document.body.classList.remove("drawer-open"));

/* ---------- settings ---------- */
document.querySelectorAll("#setting-tabs button").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll("#setting-tabs button").forEach((x) => x.classList.toggle("on", x === b));
    document.querySelectorAll("[data-setting]").forEach((c) => c.classList.toggle("hidden", c.dataset.setting !== b.dataset.tab));
    if (b.dataset.tab === "memory") renderMemory();
    if (b.dataset.tab === "mcp") renderMcp("");
    if (b.dataset.tab === "computer") { renderOperator(); renderApps(); }
    if (b.dataset.tab === "automation") { renderCustomMcp(); }
    if (b.dataset.tab === "advanced") renderLogs();
  });
});
document.querySelectorAll("#theme-seg button").forEach((b) => {
  b.addEventListener("click", () => setTheme(b.dataset.theme));
});
for (const [id, k] of [["cfg-system", "system"], ["cfg-temp", "temperature"], ["cfg-maxtok", "max_tokens"], ["cfg-fun", "fun"]]) {
  on(id, "change", async () => {
    const el = document.getElementById(id);
    store.config[k] = el.type === "checkbox" ? el.checked : (el.type === "number" || el.type === "range" ? +el.value : el.value);
    if (id === "cfg-temp") document.getElementById("temp-v").textContent = el.value;
    store.save();
    try {
      await api("/api/config", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(store.config)});
    } catch (e) { /* offline: local only */ }
  });
}



/* ---------- Model library (curated catalog + fit + one-click download) --- */
async function renderCatalog() {
  const grid = document.getElementById("catalog-grid");
  if (!grid) return;
  let cat;
  try { cat = await api("/api/catalog"); }
  catch (err) { grid.innerHTML = "<div class='card'>library offline</div>"; return; }
  grid.innerHTML = "";
  (cat.catalog || []).forEach((m) => {
    const d = document.createElement("div");
    d.className = "mcard";
    d.innerHTML = "<h4></h4><span class='fmt'></span><div class='row'></div><small></small><div class='quantrow'></div>";
    d.querySelector("h4").textContent = m.id;
    const fit = { "fits-fp16": ["fits fp16", "ok"],
                  "needs-conversion": ["convert to int4 first", "warn"],
                  "too-big": ["too big here", "bad"],
                  "unknown": ["fit unknown", ""] }[m.fit] || ["?", ""];
    d.querySelector(".fmt").textContent = m.params_b + "B - " + fit[0];
    d.querySelector(".fmt").style.color =
      m.fit === "fits-fp16" ? "var(--ok)" : (m.fit === "too-big" ? "var(--warn)" : "");
    d.querySelector("small").textContent =
      m.blurb + " - " + m.license + " - tok: " + String(m.tokenizer).split(" ")[0]
      + (m.measured_tps ? " - measured here: " + m.measured_tps + " tok/s" : "");
    const row = d.querySelector(".row");
    if (m.downloaded) {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = "Load"; b.className = "primary";
      b.addEventListener("click", () => { loadModel(m.id); showView("chat"); });
      row.appendChild(b);
    } else {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = "Download"; b.className = "primary";
      b.addEventListener("click", () => { startDownload(m.repo); b.disabled = true;
        b.textContent = "downloading..."; });
      row.appendChild(b);
    }
    const cv = document.createElement("button");
    cv.type = "button"; cv.textContent = "Convert...";
    cv.addEventListener("click", () => {
      showView("work");
      document.querySelector('#convert-form input[name="input"]').value = m.id + "/model.safetensors";
      document.querySelector('#convert-form input[name="output"]').value = m.id + "_int4.lumina";
    });
    d.querySelector(".quantrow").appendChild(cv);
    grid.appendChild(d);
  });
  const note = document.createElement("p");
  note.className = "hint";
  note.textContent = "free RAM: " + (cat.available_mb || "?") + "MB - fit is an fp16 "
    + "estimate; Convert... prefills the Work view for the int4 path (~4x smaller).";
  grid.appendChild(note);
}

/* ---------- export a thread as markdown ---------- */
function exportThread(t) {
  const lines = ["# " + t.title, ""];
  (t.msgs || []).forEach((m) => {
    lines.push("**" + m.role + "**:");
    lines.push("");
    lines.push(String(m.shown || m.text));
    lines.push("");
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n")], {type: "text/markdown"}));
  a.download = (t.title || "chat").replace(/[^a-z0-9]+/gi, "_").slice(0, 40) + ".md";
  a.click();
}

/* ---------- doctor + plan + draft training (Work view) ---------- */
document.getElementById("doctor-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const id = new FormData(e.target).get("id");
  const box = document.getElementById("doctor-out");
  box.classList.remove("hidden");
  box.innerHTML = "";
  const say = (t, cls) => {
    const d = document.createElement("div");
    d.className = cls || "mono"; d.textContent = t;
    box.appendChild(d);
  };
  say("diagnosing " + id + "…");
  try {
    const d = await api("/api/doctor?id=" + encodeURIComponent(id));
    box.innerHTML = "";
    say("doctor: " + (d.ok ? "ALL CHECKS PASS" : "PROBLEMS FOUND"));
    (d.checks || []).forEach((c) => {
      say((c.ok ? " ok  " : " FAIL ") + c.check + (c.note ? " — " + c.note : ""),
          c.ok ? "hint" : "mono");
    });
    const p = await api("/api/plan?id=" + encodeURIComponent(id));
    say("plan: weights " + p.weights_mb + "MB · working set ~" + p.working_set_mb
      + "MB · min RAM " + p.min_ram_mb + "MB · free " + p.available_mb + "MB"
      + (p.fits_now ? " · FITS" : " · DOES NOT FIT"));
    if (p.advice) say(p.advice, "hint");
  } catch (err) {
    box.innerHTML = "";
    say("doctor failed: backend unreachable");
  }
});
on("train-draft", "click", async () => {
  const st = $("#train-status");
  const btn = document.getElementById("train-draft");
  btn.disabled = true;
  st.textContent = "training draft heads…";
  setStatus("working");
  try {
    const r = await api("/api/train-draft", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify({})});
    if (r.ok) {
      st.textContent = "trained: " + (r.heads || []).join(", ");
      settleStatus("success", 1600);
    } else {
      st.textContent = r.error || "training failed";
      settleStatus("error", 2200);
    }
  } catch (err) {
    st.textContent = "backend unreachable";
    settleStatus("error", 2200);
  } finally { btn.disabled = false; }
});

/* ---------- RAG search (Docs view) ---------- */
document.getElementById("rag-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#rag-q").value.trim();
  if (!q) return;
  const out = document.getElementById("rag-out");
  out.innerHTML = "";
  const say = (t, cls) => {
    const d = document.createElement("div");
    d.className = cls || "hint"; d.textContent = t;
    out.appendChild(d);
  };
  say("searching…");
  try {
    const r = await api("/api/rag/query?q=" + encodeURIComponent(q) + "&k=4");
    out.innerHTML = "";
    if (!(r.hits || []).length) { say("no chunks matched — attach docs above first."); return; }
    r.hits.forEach((h) => {
      const li = document.createElement("div");
      li.className = "mono";
      li.textContent = "[" + h.source + "] bm25=" + h.bm25 + " :: " + h.text.slice(0, 160) + "…";
      out.appendChild(li);
    });
  } catch (err) {
    out.innerHTML = "";
    say("search failed: backend unreachable");
  }
});

/* ---------- Arena ELO ---------- */
async function renderElo() {
  const out = document.getElementById("elo-out");
  if (!out) return;
  try {
    const r = await api("/api/arena/elo");
    out.innerHTML = "";
    (r.ratings || []).forEach((x) => {
      const d = document.createElement("div");
      d.className = "mono";
      d.textContent = x.elo + "  " + x.model;
      out.appendChild(d);
    });
    if (!(r.ratings || []).length)
      out.innerHTML = "<div class='hint'>no votes yet — every vote updates the server-side ELO</div>";
  } catch (err) { /* offline */ }
}


/* ---------- request log (Settings → advanced) ---------- */
async function renderLogs() {
  const out = document.getElementById("logs-out");
  if (!out) return;
  try {
    const r = await api("/api/logs");
    out.innerHTML = "";
    (r.items || []).slice().reverse().forEach((l) => {
      const d = document.createElement("div");
      d.className = "mono";
      d.textContent = l.t + "  " + l.method + "  " + l.path;
      out.appendChild(d);
    });
    if (!(r.items || []).length)
      out.innerHTML = "<div class='hint'>no API requests yet</div>";
  } catch (err) { /* offline */ }
}
on("logs-refresh", "click", renderLogs);

/* ---------- MCP registry (Settings tab) ---------- */
async function renderMcp(q) {
  const grid = document.getElementById("mcp-grid");
  const stats = document.getElementById("mcp-stats");
  if (!grid) return;
  let r;
  try { r = await api("/api/mcp/registry?q=" + encodeURIComponent(q || "")); }
  catch (err) { grid.innerHTML = "<div class='hint'>registry offline</div>"; return; }
  stats.textContent = r.total + " configs · " + r.curated_servers + " curated servers · "
    + r.preconfigured_variants + " preconfigured variants · running: "
    + (r.running.length ? r.running.join(", ") : "none");
  grid.innerHTML = "";
  (r.items || []).forEach((m) => {
    const li = document.createElement("li");
    const nm = document.createElement("span");
    nm.textContent = m.id + (m.variant ? "  [" + m.variant + "]" : "");
    const meta = document.createElement("small");
    meta.className = "hint"; meta.textContent = m.desc.slice(0, 90);
    li.appendChild(nm); li.appendChild(meta);
    if ((r.running || []).includes(m.id)) {
      const stop = document.createElement("button");
      stop.type = "button"; stop.textContent = "Stop";
      stop.addEventListener("click", async () => {
        await api("/api/mcp/stop", {method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({id: m.id})}).catch(() => {});
        renderMcp(q);
      });
      const tools = document.createElement("button");
      tools.type = "button"; tools.textContent = "Tools";
      tools.addEventListener("click", async () => {
        const t = await api("/api/mcp/tools", {method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({id: m.id})}).catch(() => ({error: "offline"}));
        alert(t.tools ? t.tools.map((x) => x.name).join("\n") : (t.error || "no tools"));
      });
      li.appendChild(tools); li.appendChild(stop);
    } else {
      const go = document.createElement("button");
      go.type = "button"; go.textContent = "Start";
      go.addEventListener("click", async () => {
        const rr = await api("/api/mcp/start", {method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({id: m.id})}).catch(() => ({error: "offline"}));
        if (rr.error) alert("start failed: " + rr.error);
        renderMcp(q);
      });
      li.appendChild(go);
    }
    grid.appendChild(li);
  });
}
document.getElementById("mcp-search-form").addEventListener("submit", (e) => {
  e.preventDefault(); renderMcp($("#mcp-q").value);
});

/* ---------- skill library (chat strip) ---------- */
async function renderSkills() {
  const strip = document.getElementById("skills-strip");
  if (!strip) return;
  let r;
  try { r = await api("/api/skills"); }
  catch (err) { return; }
  strip.innerHTML = "";
  const lbl = document.createElement("span");
  lbl.className = "hint"; lbl.textContent = "Skills:";
  strip.appendChild(lbl);
  (r.skills || []).slice(0, 14).forEach((sk) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = sk.id;
    b.title = sk.desc;
    b.addEventListener("click", () => {
      const inp = prompt("Input for '" + sk.id + "':\n" + sk.desc);
      if (inp == null) return;
      const run = confirm("Run '" + sk.id + "' through the loaded model? Cancel = just fill the composer.");
      const promptText = sk.prompt.replace("{input}", inp);
      if (run) {
        runSkill(sk.id, inp);
      } else {
        $("#prompt").value = promptText;
        $("#prompt").dispatchEvent(new Event("input"));
      }
    });
    strip.appendChild(b);
  });
}
async function runSkill(id, input) {
  setStatus("thinking");
  try {
    const r = await api("/api/skills/run", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({skill: id, input: input, max_tokens: 128})});
    if (r.error) { notify(r.error); settleStatus("error", 2200); return; }
    if (!thread) newThread();
    thread.msgs.push({role: "user", text: "[skill: " + id + "]\n" + input, at: Date.now()});
    thread.msgs.push({role: "assistant", text: r.text, meta: r.tokens_per_second + " tok/s",
                      at: Date.now()});
    store.save(); renderMessages(); renderThreads();
    settleStatus("success", 1600);
  } catch (err) {
    notify("skill failed: backend unreachable");
    settleStatus("error", 2200);
  }
}

/* ---------- computer tab: operator mode, files, browser, apps ---------- */
function opRender(st) {
  const cb = document.getElementById("op-full");
  const out = document.getElementById("op-status");
  const o = (st && st.operator) || {};
  if (cb) cb.checked = !!o.full_access;
  if (out) {
    out.textContent = "operator: " + (o.full_access ? "ON" : "off")
      + " · source: " + (o.source || "cfg")
      + " · bind: " + (o.host || "?")
      + (o.guard ? " · " + o.guard : "")
      + (o.blocked ? " · blocked: " + o.blocked : "")
      + (o.can_enable === false ? " · cannot enable here" : "");
  }
}
async function renderOperator() {
  let cfg = {};
  try { cfg = await api("/api/config"); } catch (e) { return; }
  opRender(cfg);
  if (document.getElementById("op-cwd")) {
    document.getElementById("op-cwd").placeholder = cfg.cwd || "/home/you";
  }
}
on("op-full", "change", async (e) => {
  const el = e.target;
  try {
    const r = await api("/api/config", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({full_access: el.checked})});
    opRender(r);
    notify(el.checked ? "Operator mode ON — actions are audited." : "Operator mode off.");
  } catch (err) { notify("toggle failed (backend unreachable)"); el.checked = !el.checked; }
});
on("op-cwd-save", "click", async () => {
  const v = document.getElementById("op-cwd").value.trim();
  if (!v) return;
  try {
    const r = await api("/api/config", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify({cwd: v})});
    notify("cwd: " + (r.cwd || v));
  } catch (e) { notify("save failed"); }
});
async function renderAudit() {
  const out = document.getElementById("op-audit-out");
  if (!out) return;
  out.innerHTML = "";
  try {
    const r = await api("/api/apps", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify({op: "audit", n: 40})});
    (r.events || []).slice().reverse().forEach((e) => {
      const li = document.createElement("li");
      const s = document.createElement("span");
      s.className = "mono";
      s.textContent = new Date(e.at * 1000).toLocaleTimeString() + "  " + e.op + "  "
        + e.target.slice(0, 70) + (e.ok ? "" : "  [FAILED]");
      li.appendChild(s); out.appendChild(li);
    });
    if (!(r.events || []).length) out.innerHTML = "<li><span class='hint'>nothing audited yet — operator actions land here</span></li>";
  } catch (err) { out.innerHTML = "<li><span class='hint'>backend unreachable</span></li>"; }
}
on("op-audit", "click", renderAudit);

let fsCur = "";
async function fsList(path) {
  const out = document.getElementById("fs-out");
  if (!out) return;
  out.innerHTML = "<li class='hint'>reading…</li>";
  try {
    const r = await api("/api/fs", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({op: "list", path: path || "~"})});
    if (r.error) { out.innerHTML = ""; const li = document.createElement("li");
      li.innerHTML = "<span class='hint'></span>"; li.firstChild.textContent = r.error; out.appendChild(li); return; }
    fsCur = r.path;
    document.getElementById("fs-path").value = r.path;
    out.innerHTML = "";
    (r.items || []).forEach((it) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button"; b.textContent = (it.dir ? "▸ " : "· ") + it.name;
      b.style.flex = "1"; b.style.textAlign = "left";
      const meta = document.createElement("small"); meta.className = "hint";
      meta.textContent = it.dir ? "dir" : ((it.size / 1024).toFixed(1) + " KB");
      b.addEventListener("click", () => {
        if (it.dir) fsList(r.path + "/" + it.name); else fsView(r.path + "/" + it.name);
      });
      li.appendChild(b); li.appendChild(meta); out.appendChild(li);
    });
    if (!(r.items || []).length) out.innerHTML = "<li><span class='hint'>empty directory</span></li>";
  } catch (err) { out.innerHTML = ""; notify("files: backend unreachable or operator mode off"); }
}
async function fsView(path) {
  const pre = document.getElementById("fs-view");
  try {
    const r = await api("/api/fs", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({op: "read", path: path, max_bytes: 60000})});
    pre.classList.remove("hidden");
    pre.textContent = r.error ? r.error : (path + (r.truncated ? "  (truncated)" : "") + "\n\n" + r.text);
  } catch (e) { notify("read failed"); }
}
on("fs-list", "click", () => fsList(document.getElementById("fs-path").value));
on("fs-home", "click", () => fsList("~"));
on("fs-up", "click", () => fsList(fsCur.replace(/\/[^/]+\/?$/, "") || "/"));
on("fs-search", "click", async () => {
  const q = document.getElementById("fs-search-q").value.trim();
  const out = document.getElementById("fs-out");
  try {
    const r = await api("/api/fs", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({op: "search", path: fsCur || "~", query: q, limit: 100})});
    out.innerHTML = "";
    (r.hits || []).forEach((h) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button"; b.textContent = h.path; b.style.flex = "1"; b.style.textAlign = "left";
      b.addEventListener("click", () => { h.dir ? fsList(h.path) : fsView(h.path); });
      li.appendChild(b); out.appendChild(li);
    });
    if (!(r.hits || []).length) out.innerHTML = "<li><span class='hint'>no matches</span></li>";
  } catch (e) { notify("search failed"); }
});

async function browserOp(op, extra) {
  const st = document.getElementById("br-status");
  try {
    const r = await api("/api/browser", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(Object.assign({op: op}, extra || {}))});
    if (r.error) { st.textContent = r.error; return r; }
    st.textContent = op + " ok · " + (r.url || "") + (r.title ? " · " + r.title : "")
      + (r.bytes ? " · " + r.bytes + " bytes" : "");
    return r;
  } catch (e) { st.textContent = "browser: backend unreachable or operator mode off"; return {}; }
}
on("br-go", "click", () => browserOp("navigate", {url: document.getElementById("br-url").value.trim()}));
on("br-text", "click", async () => {
  const r = await browserOp("text");
  const pre = document.getElementById("br-text-out");
  pre.textContent = (r.text || r.error || "").slice(0, 4000);
});
on("br-shot", "click", async () => {
  const r = await browserOp("screenshot");
  const img = document.getElementById("br-shot-img");
  if (r.data_url) { img.src = r.data_url; img.hidden = false; }
  else if (r.url) { img.src = r.url; img.hidden = false; }
});
on("br-close", "click", () => browserOp("close"));
on("br-fill", "click", () => browserOp("fill", {selector: document.getElementById("br-sel").value.trim(),
  value: document.getElementById("br-val").value}));
on("br-type", "click", () => browserOp("type_human", {selector: document.getElementById("br-sel").value.trim(),
  value: document.getElementById("br-val").value, delay_ms: 35}));
on("br-click", "click", () => browserOp("click", {selector: document.getElementById("br-sel").value.trim(), settle_ms: 400}));
on("br-enter", "click", () => browserOp("press", {key: "Enter", settle_ms: 400}));
async function renderApps() {
  const out = document.getElementById("apps-out");
  try {
    const r = await api("/api/apps");
    const a = (r.apps && r.apps.available) || {};
    out.textContent = Object.keys(a).length
      ? "installed here: " + Object.keys(a).join(", ")
      : (r.error || "no allowlisted app found on this machine");
  } catch (e) { out.textContent = "backend unreachable"; }
}
on("apps-refresh", "click", renderApps);
on("app-launch", "click", async () => {
  const name = document.getElementById("app-name").value.trim();
  if (!name) return;
  try {
    const r = await api("/api/apps", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({op: "launch", app: name})});
    notify(r.error || ("launched " + r.launched + " (pid " + r.pid + ")"));
  } catch (e) { notify("launch failed (operator off or unreachable)"); }
});

/* ---------- automation tab: custom MCP, skills, workflows ---------- */
const CM_EXAMPLE_DEFAULT = [{
  name: "echo",
  description: "Return the arguments you pass as JSON.",
  handler: {kind: "echo"},
  input_schema: {type: "object", properties: {text: {type: "string"}}, required: []}
}];
async function cmLoadExample() {
  try {
    const r = await api("/api/custom_mcp");
    const ex = r.example || {id: "my-tools", name: "My tools", desc: "", tools: CM_EXAMPLE_DEFAULT};
    document.getElementById("cm-id").value = ex.id || "my-tools";
    document.getElementById("cm-name").value = ex.name || "My tools";
    document.getElementById("cm-desc").value = ex.desc || "";
    document.getElementById("cm-tools").value = JSON.stringify(ex.tools || CM_EXAMPLE_DEFAULT, null, 1);
    document.getElementById("cm-status").textContent = "example loaded — save to generate the server file";
  } catch (e) {
    document.getElementById("cm-tools").value = JSON.stringify(CM_EXAMPLE_DEFAULT, null, 1);
  }
}
on("cm-load-example", "click", cmLoadExample);
on("cm-save", "click", async () => {
  const st = document.getElementById("cm-status");
  let tools;
  try { tools = JSON.parse(document.getElementById("cm-tools").value || "[]"); }
  catch (e) { st.textContent = "tools JSON is not valid JSON: " + e.message; return; }
  try {
    const r = await api("/api/custom_mcp", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({op: "save", server: {
        id: document.getElementById("cm-id").value.trim(),
        name: document.getElementById("cm-name").value.trim(),
        desc: document.getElementById("cm-desc").value.trim(),
        tools: tools}})});
    st.textContent = r.error ? ("save failed: " + r.error)
      : ("saved " + r.saved + " · " + r.bytes + " bytes · tools: " + (r.tools || []).join(", ")
         + " — start it from the MCPs tab (id custom/" + r.saved + ")");
    if (!r.error) renderCustomMcp();
  } catch (e) { st.textContent = "save failed: backend unreachable"; }
});
on("cm-test", "click", async () => {
  const st = document.getElementById("cm-status");
  const pre = document.getElementById("cm-test-out");
  const id = document.getElementById("cm-id").value.trim();
  pre.textContent = "starting the generated server and speaking real JSON-RPC…";
  try {
    const r = await api("/api/custom_mcp", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({op: "test", id: id})});
    pre.textContent = JSON.stringify(r, null, 1).slice(0, 4000);
    st.textContent = r.error ? ("test failed: " + r.error) : "round-trip ok";
  } catch (e) { pre.textContent = "test failed: backend unreachable"; }
});
async function renderCustomMcp() {
  const out = document.getElementById("cm-servers");
  if (!out) return;
  out.innerHTML = "";
  try {
    const r = await api("/api/custom_mcp");
    (r.servers || []).forEach((s) => {
      const li = document.createElement("li");
      const span = document.createElement("span");
      span.textContent = "custom/" + s.id + " — " + s.name + "  ·  "
        + (s.tools || []).map((t) => t.name).join(", ");
      const test = document.createElement("button");
      test.type = "button"; test.textContent = "Test";
      test.addEventListener("click", async () => {
        const rr = await api("/api/custom_mcp", {method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({op: "test", id: s.id})}).catch(() => ({error: "offline"}));
        document.getElementById("cm-test-out").textContent = JSON.stringify(rr, null, 1).slice(0, 3000);
      });
      const del = document.createElement("button");
      del.type = "button"; del.textContent = "Delete";
      del.addEventListener("click", async () => {
        if (!confirm("Delete custom MCP " + s.id + "?")) return;
        await api("/api/custom_mcp", {method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({op: "delete", id: s.id})}).catch(() => {});
        renderCustomMcp();
      });
      li.appendChild(span); li.appendChild(test); li.appendChild(del);
      out.appendChild(li);
    });
  } catch (e) { /* offline */ }
}
on("cm-list", "click", renderCustomMcp);
on("sk-save", "click", async () => {
  const out = document.getElementById("sk-out");
  try {
    const r = await api("/api/skills/save", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({id: document.getElementById("sk-id").value.trim(),
        desc: document.getElementById("sk-desc").value.trim(),
        prompt: document.getElementById("sk-prompt").value})});
    notify(r.error || ("saved skill " + r.saved + (r.has_input_var ? "" : " — note: no {input} in the prompt")));
    if (!r.error) { renderSkills(); }
    out.innerHTML = "";
  } catch (e) { notify("save failed: backend unreachable"); }
});
on("sk-list", "click", async () => {
  const out = document.getElementById("sk-out");
  out.innerHTML = "";
  try {
    const r = await api("/api/skills");
    (r.skills || []).forEach((s) => {
      const li = document.createElement("li");
      const span = document.createElement("span");
      span.textContent = s.id + (s.user ? "  [user]" : "") + " — " + (s.desc || "");
      li.appendChild(span);
      if (s.user) {
        const del = document.createElement("button");
        del.type = "button"; del.textContent = "Delete";
        del.addEventListener("click", async () => {
          await api("/api/skills/delete", {method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({id: s.id})}).catch(() => {});
          on_sk_list_again();
        });
        li.appendChild(del);
      }
      out.appendChild(li);
    });
  } catch (e) { /* offline */ }
});
function on_sk_list_again() { document.getElementById("sk-list").click(); }
on("wf-example", "click", async () => {
  try {
    const r = await api("/api/workflows");
    const ex = (r.examples || [])[0];
    document.getElementById("wf-steps").value = JSON.stringify(ex ? ex.steps :
      [{kind: "echo", text: "hello from a workflow"}], null, 1);
  } catch (e) {}
});
on("wf-run", "click", async () => {
  const pre = document.getElementById("wf-out");
  let steps;
  try { steps = JSON.parse(document.getElementById("wf-steps").value || "[]"); }
  catch (e) { pre.textContent = "steps JSON invalid: " + e.message; return; }
  pre.textContent = "running…";
  try {
    const r = await api("/api/workflow/run", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({steps: steps})});
    pre.textContent = JSON.stringify(r, null, 1).slice(0, 6000);
  } catch (e) { pre.textContent = "run failed: backend unreachable"; }
});

/* ---------- init ---------- */
(async function init() {
  setTheme(store.config.theme || "dark");
  document.querySelectorAll(".theme-dots button").forEach((b) => {
    b.addEventListener("click", () => setTheme(b.dataset.t));
  });
  document.getElementById("cfg-system").value = store.config.system || "";
  syncConfigUI();
  document.getElementById("cfg-fun").checked = !!store.config.fun;
  renderThreads(); renderDocs(); renderTasks(); renderStrip(); renderAttachChips();
  renderElo(); renderSkills();
  loadCloudForm();
  drawStatus();
  try { await refreshLibrary(); } catch (e) { /* mock fallback inside */ }
  try { await refreshArenaModels(); } catch (e) {}
  renderGraphStats(); renderGraphList(""); renderClaims();
  drawSlide();
  try {
    const st = await api("/api/status");
    document.getElementById("status-record").textContent =
      (st && st.status) ? st.status.slice(0, 1500) : "backend unreachable — running on local mock data.";
  } catch (e) {
    document.getElementById("status-record").textContent = "backend unreachable — running on local mock data.";
  }
  if (!store.threads.length) newThread(); else { thread = store.threads[0]; renderMessages(); }
})();

/* ---------- agent view (builtin loop / OpenAI Agents SDK) ---------- */
let agentWarmKicked = false;
async function renderAgentStatus(warm) {
  const el = $("#agent-status");
  if (!el) return;
  try {
    const st = await api("/api/agent/sdk");
    if (st && st.ready) {
      el.textContent = "openai-agents " + st.version + " · ready"
        + (st.import_s ? " · import " + st.import_s + "s" : "");
      return;
    }
    if (st && st.disabled) { el.textContent = "SDK disabled by env (LUMINA_AGENTS_DISABLE) · builtin loop"; return; }
    if (st && st.installed === false) { el.textContent = "openai-agents not installed · builtin loop only"; return; }
    if (st && (warm || !agentWarmKicked)) {
      agentWarmKicked = true;
      el.textContent = "warming up: importing openai-agents (one-time)…";
      await api("/api/agent/sdk/warmup", {method: "POST",
        headers: {"Content-Type": "application/json"}, body: "{}"});
      setTimeout(() => renderAgentStatus(true), 6000);
      return;
    }
    el.textContent = st ? ("openai-agents · " + (st.state || "unknown")) : "backend unreachable";
  } catch (e) { el.textContent = "backend unreachable"; }
}
document.getElementById("agent-warm").addEventListener("click", () => renderAgentStatus(true));
document.getElementById("agent-engine").addEventListener("change", (e) => {
  api("/api/config", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({agent_engine: e.target.value})}).catch(() => {});
});
document.getElementById("agent-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const p = $("#agent-prompt").value.trim();
  if (!p) return;
  const out = $("#agent-steps");
  out.textContent = "running… (each step is a real model call)";
  try {
    const r = await api("/api/agent", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({messages: [{role: "user", content: p}],
                            engine: $("#agent-engine").value, max_steps: 3})});
    const lines = [];
    if (r.engine) lines.push("engine: " + r.engine + (r.engine_reason ? " (" + r.engine_reason + ")" : ""));
    if (r.model_target) lines.push("model: " + (r.model_target.model || r.model_target.model_id || "?")
      + " [" + (r.model_target.kind || "?") + "]");
    if (r.sdk && r.sdk.version) lines.push("sdk: openai-agents " + r.sdk.version);
    for (const s of (r.steps || [])) {
      if (s.type === "tool") lines.push("tool: " + ((s.call && s.call.tool) || "?") + " "
        + JSON.stringify((s.call && s.call.args) || {}) + "\n  -> " + String(s.result).slice(0, 400));
      else if (s.type === "final") lines.push("final: " + s.text);
      else if (s.type === "bad-call") lines.push("bad call: " + s.text);
    }
    if (r.error) lines.push("error: " + r.error);
    out.textContent = lines.join("\n") || "(no steps)";
  } catch (e) { out.textContent = "backend unreachable"; }
});

/* ---------- media generation view (declared backends, AI-callable) ---------- */

function mediaFileUrl(name) {
  return "/api/media/file?name=" + encodeURIComponent(name || "");
}
function mediaExt(name) {
  const m = String(name || "").toLowerCase().match(/\.([a-z0-9]+)$/);
  return m ? m[1] : "";
}
function mediaPreview(f) {
  const src = f.url || mediaFileUrl(f.name);
  const ext = mediaExt(f.name || f.url);
  let el;
  if (["png", "jpg", "jpeg", "webp", "gif"].includes(ext)) {
    el = document.createElement("img");
    el.className = "media-thumb"; el.alt = "";
  } else if (["mp4", "webm"].includes(ext)) {
    el = document.createElement("video");
    el.className = "media-thumb"; el.controls = true; el.preload = "metadata";
  } else if (["wav", "mp3", "ogg", "flac"].includes(ext)) {
    el = document.createElement("audio");
    el.controls = true; el.preload = "metadata";
  } else {
    el = document.createElement("a");
    el.textContent = f.name || "download";
    el.target = "_blank"; el.rel = "noopener";
  }
  el.src = src;
  return el;
}
async function renderMediaResidency() {
  const el = $("#media-residency");
  if (!el) return;
  try {
    const r = await api("/api/residency");
    if (r.offline) { el.textContent = "backend unreachable"; return; }
    const parts = [];
    parts.push(r.loaded
      ? ("LLM loaded: " + (r.model_id || "?")
         + (r.weights_mb != null ? " (" + Math.round(r.weights_mb) + " MB weights)" : ""))
      : "LLM not loaded");
    if (r.rss_mb != null) parts.push("RSS " + Math.round(r.rss_mb) + " MB");
    if (r.available_mb != null) parts.push("free " + Math.round(r.available_mb) + " MB");
    parts.push("policy: " + (r.policy || "off"));
    const pr = r.partial_residency || {};
    parts.push("partial residency: " + (pr.supported
      ? "supported"
      : "not supported (loader reads every tensor eagerly — deferred)"));
    el.textContent = parts.join(" · ");
  } catch (e) { el.textContent = "backend unreachable"; }
}
async function renderMediaBackends() {
  const el = $("#media-backends");
  if (!el) return;
  el.textContent = "";
  const note = (txt) => {
    const li = document.createElement("li");
    li.textContent = txt;
    el.appendChild(li);
  };
  try {
    const r = await api("/api/media/backends");
    const bs = (r && r.backends) || [];
    if (!bs.length) { note(r.offline ? "backend unreachable" : "no backends declared"); return; }
    for (const b of bs) {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = b.id + " · " + ((b.caps || []).join(", ") || "?")
        + " · " + (b.kind || "?");
      const st = document.createElement("span");
      st.className = "hint";
      st.textContent = b.configured
        ? (b.reachable === false ? "configured, unreachable" : "ready")
        : "not configured";
      if (b.error) st.textContent += " — " + b.error;
      else if (b.note) st.textContent += " — " + b.note;
      li.appendChild(name); li.appendChild(st);
      el.appendChild(li);
    }
  } catch (e) { note("backend unreachable"); }
}
async function renderMediaWorkflows() {
  const sel = $("#media-workflow");
  if (!sel) return;
  try {
    const r = await api("/api/media/workflows");
    const ws = (r && r.workflows) || [];
    const cur = sel.value;
    sel.textContent = "";
    const auto = document.createElement("option");
    auto.value = ""; auto.textContent = "workflow: auto";
    sel.appendChild(auto);
    for (const w of ws) {
      const o = document.createElement("option");
      o.value = w.id;
      o.textContent = w.id + " (" + (w.capability || "?") + ")";
      sel.appendChild(o);
    }
    sel.value = cur;
  } catch (e) { /* keep auto */ }
}
function mediaJobLine(j) {
  let line = j.id + " · " + j.capability + " · " + j.status;
  if (j.backend) line += " · " + j.backend;
  if (j.seconds != null) line += " · " + j.seconds + "s";
  if (j.residency) {
    line += " · residency: " + (j.residency.policy_used || "?");
    if (j.residency.freed_mb) line += " (freed " + Math.round(j.residency.freed_mb) + " MB)";
    if (j.residency.reloaded) line += " · LLM reloaded";
    if (j.residency.reload_error) line += " · reload failed: " + j.residency.reload_error;
  }
  return line;
}
async function renderMediaJobs() {
  const el = $("#media-jobs");
  if (!el) return;
  try {
    const r = await api("/api/media/jobs");
    const js = (r && r.jobs) || [];
    el.textContent = "";
    if (!js.length) {
      const d = document.createElement("div");
      d.className = "hint";
      d.textContent = r.offline ? "backend unreachable" : "no jobs yet";
      el.appendChild(d);
      return;
    }
    for (const j of js.slice(0, 12)) {
      const card = document.createElement("div");
      card.className = "card";
      const h = document.createElement("h3");
      h.textContent = mediaJobLine(j);
      card.appendChild(h);
      if (j.prompt) {
        const p = document.createElement("div");
        p.className = "hint";
        p.textContent = String(j.prompt).slice(0, 140);
        card.appendChild(p);
      }
      if (j.error) {
        const e2 = document.createElement("pre");
        e2.className = "mono";
        e2.textContent = "error: " + j.error;
        card.appendChild(e2);
      }
      for (const f of (j.files || [])) {
        const box = document.createElement("div");
        box.appendChild(mediaPreview(f));
        if (f.name) {
          const cap = document.createElement("div");
          cap.className = "hint";
          cap.textContent = f.name
            + (f.bytes != null ? " · " + Math.round(f.bytes / 1024) + " KB" : "")
            + (f.w && f.h ? " · " + f.w + "x" + f.h : "");
          box.appendChild(cap);
        }
        card.appendChild(box);
      }
      el.appendChild(card);
    }
  } catch (e) { el.textContent = "backend unreachable"; }
}
async function pollMediaJob(id, out, tries) {
  tries = tries || 0;
  if (tries > 120) { out.textContent += "\ngave up polling after " + tries + " tries"; return; }
  await sleep(1500);
  try {
    const r = await api("/api/media/jobs?id=" + encodeURIComponent(id));
    const j = r && r.job;
    if (!j) return;
    if (j.status === "done" || j.status === "error") {
      out.textContent += "\n" + mediaJobLine(j) + (j.error ? "\nerror: " + j.error : "");
      renderMediaJobs(); renderMediaResidency();
      return;
    }
    pollMediaJob(id, out, tries + 1);
  } catch (e) { /* quiet — the jobs list still shows the truth */ }
}
document.getElementById("media-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const out = $("#media-status");
  const prompt = $("#media-prompt").value.trim();
  if (!prompt) { out.textContent = "prompt required"; return; }
  const body = {capability: $("#media-cap").value, prompt: prompt, wait: true};
  const neg = $("#media-neg").value.trim(); if (neg) body.negative = neg;
  const wf = $("#media-workflow").value; if (wf) body.workflow_id = wf;
  body.width = parseInt($("#media-w").value || "1024", 10);
  body.height = parseInt($("#media-h").value || "1024", 10);
  body.steps = parseInt($("#media-steps").value || "20", 10);
  const seed = $("#media-seed").value.trim();
  if (seed) body.seed = parseInt(seed, 10);
  const pol = $("#media-policy").value; if (pol) body.policy = pol;
  out.textContent = "running… (the residency policy may unload the LLM first)";
  try {
    const r = await api("/api/media/generate", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    if (r.error) {
      out.textContent = "error: " + r.error
        + (r.residency ? "\nresidency: " + JSON.stringify(r.residency) : "");
    } else {
      out.textContent = r.text || "done";
      if (r.job && (r.job.status === "queued" || r.job.status === "running")) {
        out.textContent += "\njob still running — polling…";
        pollMediaJob(r.job.id, out, 0);
        return;
      }
    }
    renderMediaJobs(); renderMediaResidency();
  } catch (e2) { out.textContent = "backend unreachable"; }
});

/* ---------- composer toggles (grounding + auto skill fetch) ---------- */
function renderChatToggles() {
  const g = document.getElementById("tg-ground");
  const s = document.getElementById("tg-skill");
  if (g) {
    g.classList.toggle("on", !!store.config.ground);
    g.setAttribute("aria-pressed", String(!!store.config.ground));
  }
  if (s) {
    s.classList.toggle("on", !!store.config.skill);
    s.setAttribute("aria-pressed", String(!!store.config.skill));
  }
}
if (document.getElementById("tg-ground")) {
  document.getElementById("tg-ground").addEventListener("click", () => {
    store.config.ground = !store.config.ground;
    store.save();
    renderChatToggles();
  });
}
if (document.getElementById("tg-skill")) {
  document.getElementById("tg-skill").addEventListener("click", () => {
    store.config.skill = !store.config.skill;
    store.save();
    renderChatToggles();
  });
}
renderChatToggles();

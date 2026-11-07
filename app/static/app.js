"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));

const LABEL_CLASS = {
  "included": "l-pos", "not excluded": "l-pos", "not included": "l-neg", "excluded": "l-neg",
  "not enough information": "l-nei", "not applicable": "l-na", "invalid": "l-invalid",
};
const STATUS = { likely_eligible: "Likely eligible", likely_ineligible: "Likely not eligible", needs_review: "Needs review" };
const DECODER_NAME = { kv: "KV cache (shared prefix)", nocache: "No cache", spec_lookup: "Speculative, prompt lookup", spec_draft: "Speculative, draft model" };

let state = { result: null, examples: [], gold: null, filter: "all", selected: null, health: null };

// ------------------------------------------------------------------ api
async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  let res;
  try { res = await fetch(path, opts); }
  catch (e) { throw new Error("Cannot reach the server. Is `python -m app` still running?"); }
  const type = res.headers.get("content-type") || "";
  const data = type.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) {
    const msg = (data && (data.error || data.detail)) || `Request failed (${res.status})`;
    throw new Error(Array.isArray(msg) ? msg.map((m) => m.msg).join("; ") : msg);
  }
  return data;
}

// ------------------------------------------------------------------ status
async function pollHealth() {
  const pill = $("#status");
  try {
    const h = await api("/api/health");
    state.health = h;
    if (h.status === "ready") {
      const m = h.model || {};
      pill.className = "pill pill-ok";
      pill.textContent = `Ready · ${(m.n_params / 1e9).toFixed(2)}B params · ${h.config.adapter ? "LoRA" : "zero-shot"} · ${h.device}`;
      pill.title = `${h.config.base_model}${h.config.adapter ? " + " + h.config.adapter : ""}\nloaded in ${fmt(h.load_seconds, 1)} s\ndraft model: ${h.draft_status}`;
      return;
    }
    if (h.status === "error") { pill.className = "pill pill-bad"; pill.textContent = "Model failed to load"; pill.title = h.error; return; }
    pill.className = "pill pill-wait"; pill.textContent = "Loading model…"; pill.title = "Loading weights into memory (the first run also downloads ~3 GB).";
  } catch (e) { pill.className = "pill pill-bad"; pill.textContent = "Server offline"; }
  setTimeout(pollHealth, 2000);
}

// ------------------------------------------------------------------ tabs
$$(".tab").forEach((b) => b.addEventListener("click", () => {
  $$(".tab").forEach((t) => { t.classList.toggle("active", t === b); t.setAttribute("aria-selected", t === b); });
  $$(".tabpanel").forEach((p) => (p.hidden = p.id !== `tab-${b.dataset.tab}`));
  if (b.dataset.tab === "bench") loadBenchmarks();
}));

// ------------------------------------------------------------------ input preview
let parseTimer;
function schedulePreview() { clearTimeout(parseTimer); parseTimer = setTimeout(preview, 400); }
async function preview() {
  const note = $("#note").value, crit = $("#criteria").value;
  $("#note-meta").textContent = ""; $("#crit-meta").textContent = "";
  if (!note.trim() && !crit.trim()) return;
  try {
    const p = await api("/api/parse", { note: note || "x", criteria_text: crit || "Inclusion Criteria:\n- x" });
    if (note.trim()) $("#note-meta").textContent = `${p.sentences.length} sentences detected`;
    if (crit.trim()) {
      const inc = p.criteria.filter((c) => c.criterion_type === "inclusion").length;
      $("#crit-meta").textContent = `${inc} inclusion, ${p.criteria.length - inc} exclusion criteria`;
    }
  } catch (e) { if (crit.trim()) $("#crit-meta").textContent = e.message; }
}
$("#note").addEventListener("input", schedulePreview);
$("#criteria").addEventListener("input", schedulePreview);
$("#decoder").addEventListener("change", () => { $("#k-wrap").hidden = !$("#decoder").value.startsWith("spec"); });

// ------------------------------------------------------------------ examples
async function loadExamples() {
  try {
    const { examples } = await api("/api/examples");
    state.examples = examples;
    $("#example").innerHTML = `<option value="">Load an example…</option>` +
      examples.map((e, i) => `<option value="${i}">${esc(e.title)}</option>`).join("");
  } catch (e) { /* examples are optional */ }
}
$("#example").addEventListener("change", (ev) => {
  const ex = state.examples[ev.target.value];
  if (!ex) return;
  $("#note").value = ex.note; $("#criteria").value = ex.criteria;
  state.gold = ex.gold ? Object.fromEntries(ex.gold.map((g) => [g.criterion_text.trim(), g])) : null;
  preview();
});
$("#clear").addEventListener("click", () => {
  $("#note").value = ""; $("#criteria").value = ""; $("#example").value = ""; state.gold = null;
  $("#note-meta").textContent = ""; $("#crit-meta").textContent = ""; showError(null);
});

// ------------------------------------------------------------------ screening
function showError(msg) { const el = $("#form-error"); el.hidden = !msg; el.textContent = msg || ""; }

$("#form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  showError(null);
  const note = $("#note").value, crit = $("#criteria").value;
  if (!note.trim()) return showError("Please enter a patient note.");
  if (!crit.trim()) return showError("Please enter at least one criterion under 'Inclusion Criteria:' or 'Exclusion Criteria:'.");
  if (state.health && state.health.status !== "ready") return showError("The model is still loading. Wait for the green status pill.");
  const typed = $("#criteria").value;
  if (state.gold && !Object.keys(state.gold).every((t) => typed.includes(t))) state.gold = null;  // edited example
  const btn = $("#run"); btn.disabled = true;
  $("#empty").hidden = true; $("#out").hidden = true; $("#progress").hidden = false;
  const n = (crit.match(/\n\s*[-*•\d]/g) || []).length || "the";
  $("#progress-text").textContent = `Screening ${n} criteria with ${DECODER_NAME[$("#decoder").value]}… on a CPU this takes a few seconds per criterion.`;
  try {
    const res = await api("/api/screen", { note, criteria_text: crit, decoder: $("#decoder").value, k: Number($("#k").value) || 4 });
    state.result = res; state.filter = "all"; state.selected = null;
    render();
  } catch (e) {
    showError(e.message); $("#empty").hidden = false;
  } finally { btn.disabled = false; $("#progress").hidden = true; }
});

function render() {
  const r = state.result; if (!r) return;
  $("#out").hidden = false;
  const o = r.overall;
  const gold = state.gold;
  let goldLine = "";
  if (gold) {
    const matched = r.results.filter((x) => gold[x.criterion_text.trim()]);
    const agree = matched.filter((x) => gold[x.criterion_text.trim()].label === x.label).length;
    if (matched.length) goldLine = `<p style="margin-top:6px">Agreement with the expert annotation (validation split): <b>${agree}/${matched.length}</b></p>`;
  }
  $("#overall").className = `card banner ${o.status}`;
  $("#overall").innerHTML = `<p class="big">${STATUS[o.status] || o.status}</p><p>${esc(o.summary)}</p>${goldLine}
    <div class="counts">${Object.entries(o.counts).map(([l, c]) => `<span class="chip ${LABEL_CLASS[l] || "l-na"}">${esc(l)}: ${c}</span>`).join("")}</div>`;
  $("#warnings").innerHTML = (r.warnings || []).map((w) => `<div class="warn">${esc(w)}</div>`).join("");

  const groups = [["all", "All"], ["inclusion", "Inclusion"], ["exclusion", "Exclusion"], ["flag", "Rules out / undecided"]];
  $("#filters").innerHTML = groups.map(([k, t]) => `<button type="button" class="btn small ${state.filter === k ? "on" : ""}" data-f="${k}">${t}</button>`).join("");
  $$("#filters button").forEach((b) => b.addEventListener("click", () => { state.filter = b.dataset.f; render(); }));

  const rows = r.results.filter((x) => state.filter === "all" || x.criterion_type === state.filter ||
    (state.filter === "flag" && ["not included", "excluded", "not enough information", "invalid"].includes(x.label)));
  $("#crit-list").innerHTML = rows.map((x) => {
    const g = gold && gold[x.criterion_text.trim()];
    const goldHtml = g ? `<span class="${g.label === x.label ? "gold-ok" : "gold-bad"}">${g.label === x.label ? "✓ matches expert" : "✗ expert: " + esc(g.label)}</span>` : "";
    const spec = x.speculative ? `<span>accepted ${x.speculative.accepted}/${x.speculative.drafted} drafted</span>` : "";
    const dropped = x.dropped_sentence_ids.length ? `<span>dropped sentences: ${x.dropped_sentence_ids.join(", ")}</span>` : "";
    return `<li class="crit ${state.selected === x.index ? "sel" : ""}" tabindex="0" data-i="${x.index}">
      <div class="head"><div><div class="type">${esc(x.criterion_type)} #${x.index + 1}</div><div class="text">${esc(x.criterion_text)}</div></div>
        <span class="chip ${LABEL_CLASS[x.label] || "l-na"}">${esc(x.label)}</span></div>
      <div>${x.evidence.length ? x.evidence.map((e) => `<span class="ev">${e}</span>`).join("") : `<span class="hint">no evidence cited</span>`}</div>
      <p class="expl">${esc(x.explanation)}</p>
      <div class="meta">${goldHtml}<span>${fmt(x.latency_ms, 0)} ms · ${x.new_tokens} tokens</span>${spec}${dropped}
        <button type="button" class="linkbtn" data-cmp="${x.index}">compare decoders</button></div></li>`;
  }).join("") || `<li class="hint">No criteria in this filter.</li>`;
  $$(".crit").forEach((li) => {
    const pick = () => { state.selected = Number(li.dataset.i); render(); };
    li.addEventListener("click", (e) => { if (!e.target.dataset.cmp) pick(); });
    li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
  });
  $$("[data-cmp]").forEach((b) => b.addEventListener("click", () => compare(Number(b.dataset.cmp))));

  const hl = new Set(state.selected !== null ? r.results[state.selected].evidence : r.results.flatMap((x) => x.evidence));
  $("#note-view").innerHTML = r.sentences.map((s) => `<li value="${s.id}" class="${hl.has(s.id) ? "hl" : ""}">${esc(s.text)}</li>`).join("");
  const t = r.timing;
  $("#timing").textContent = `${DECODER_NAME[r.decoder]}${r.k ? ` (k=${r.k})` : ""} · ${r.results.length} criteria in ${fmt(t.total_s, 2)} s ` +
    `(${fmt(t.per_criterion_ms, 0)} ms each, ${fmt(t.tokens_per_s, 1)} generated tokens/s)` +
    (t.shared_prefix_tokens ? ` · ${t.shared_prefix_tokens} shared prompt tokens encoded once` : "");
}

// ------------------------------------------------------------------ live decoder comparison
async function compare(i) {
  const r = state.result, c = r.results[i];
  const box = $("#compare"); box.hidden = false;
  const withDraft = state.health && state.health.draft_status !== "not_configured";
  $("#compare-body").innerHTML = `<p class="hint">Running criterion #${i + 1} through every decoder…</p>`;
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
  try {
    const res = await api("/api/compare", { note: $("#note").value, criterion_type: c.criterion_type, criterion_text: c.criterion_text,
      k: Number($("#k").value) || 4, include_draft: withDraft && $("#decoder").value === "spec_draft" });
    const base = res.rows.find((x) => x.decoder === "kv");
    $("#compare-body").innerHTML = `<p class="hint">Criterion #${i + 1}: ${esc(c.criterion_text)}</p><div class="tbl-wrap"><table class="tbl">
      <tr><th>Decoder</th><th class="num">Latency (ms)</th><th class="num">Tokens</th><th class="num">Tokens/s</th><th class="num">vs KV</th><th>Accepted</th><th>Same output</th></tr>
      ${res.rows.map((x) => `<tr><td>${DECODER_NAME[x.decoder]}</td><td class="num">${fmt(x.latency_ms, 0)}</td><td class="num">${x.new_tokens}</td>
        <td class="num">${fmt(x.tokens_per_s, 1)}</td><td class="num">${base ? "×" + fmt(base.latency_ms / x.latency_ms, 2) : "–"}</td>
        <td>${x.speculative ? `${x.speculative.accepted}/${x.speculative.drafted}` : "–"}</td><td>${x.same_output_as_first ? "✓" : "✗"}</td></tr>`).join("")}
      </table></div><p class="hint">Single-criterion timing without the shared-prefix reuse; "×" above 1 means faster than KV cache. All decoders are greedy, so the outputs should match.</p>`;
  } catch (e) { $("#compare-body").innerHTML = `<div class="error">${esc(e.message)}</div>`; }
}

// ------------------------------------------------------------------ export
$$("[data-export]").forEach((b) => b.addEventListener("click", async () => {
  const r = state.result; if (!r) return;
  const fmtName = b.dataset.export, stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
  const meta = { Decoder: DECODER_NAME[r.decoder], Model: state.health ? `${state.health.config.base_model} + ${state.health.config.adapter || "no adapter"}` : "" };
  try {
    if (fmtName === "json") return download(`screening-${stamp}.json`, JSON.stringify(r, null, 2), "application/json");
    const text = await api("/api/export", { result: r, format: fmtName, meta });
    if (fmtName === "markdown") return download(`screening-${stamp}.md`, text, "text/markdown");
    const w = window.open("", "_blank");
    if (w) { w.document.write(text); w.document.close(); } else download(`screening-${stamp}.html`, text, "text/html");
  } catch (e) { alertBox(e.message); }
}));
function download(name, text, type) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name;
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}
function alertBox(msg) { $("#warnings").insertAdjacentHTML("afterbegin", `<div class="error">${esc(msg)}</div>`); }

// ------------------------------------------------------------------ benchmarks
let benchLoaded = false;
async function loadBenchmarks() {
  if (benchLoaded) return;
  const body = $("#bench-body");
  try {
    const b = await api("/api/benchmarks");
    benchLoaded = true;
    body.innerHTML = [evalCard(b), kvCard(b.kv), attnCard(b.attention), specCard(b.speculative), figCard(b.figures)].join("");
  } catch (e) { body.innerHTML = `<div class="card error">${esc(e.message)}</div>`; }
}
function evalCard(b) {
  if (!b.evaluation.length) return `<div class="card"><h2>Evaluation</h2><p class="hint">No evaluation results found in results/.</p></div>`;
  const best = Math.max(...b.evaluation.filter((r) => r.split === "test").map((r) => r.macro_f1));
  const row = (r) => `<tr><td>${esc(r.model)}</td><td>${r.split}</td><td class="num">${r.n}</td><td class="num">${fmt(r.accuracy)}</td>
    <td class="num ${r.split === "test" && r.macro_f1 === best ? "best" : ""}">${fmt(r.macro_f1)}</td><td class="num">${fmt(r.inclusion_accuracy)}</td>
    <td class="num">${fmt(r.exclusion_accuracy)}</td><td class="num">${fmt(r.evidence_micro_f1)}</td></tr>`;
  return `<div class="card"><h2>Criterion-level evaluation (greedy decoding, fixed patient split)</h2><div class="tbl-wrap"><table class="tbl">
    <tr><th>Model</th><th>Split</th><th class="num">n</th><th class="num">Accuracy</th><th class="num">Macro-F1</th><th class="num">Incl. acc</th><th class="num">Excl. acc</th><th class="num">Evidence micro-F1</th></tr>
    ${b.evaluation.map(row).join("")}</table></div>
    <p class="hint">Final model chosen on validation macro-F1 (${b.final_model ? esc(b.final_model.final) : "–"}); the test split was used only for final reporting.</p></div>`;
}
function kvCard(kv) {
  if (!kv) return "";
  const ag = kv.task_agreement_fp32 || kv.task_agreement_fp16;
  return `<div class="card"><h2>Task 3: KV-cached vs no-cache decoding <span class="hint">${esc(kv.device || "")}, fp16</span></h2>
    <p>Task prompts: KV = no-cache on <b>${kv.task_agreement_fp16.nocache_eq_kv}/${kv.task_agreement_fp16.n}</b> (fp16)${kv.task_agreement_fp32 ? `, <b>${ag.nocache_eq_kv}/${ag.n}</b> (fp32)` : ""}; KV = HF generate on <b>${kv.task_agreement_fp16.kv_eq_hf_generate}/${kv.task_agreement_fp16.n}</b>.</p>
    <div class="tbl-wrap"><table class="tbl"><tr><th class="num">Input</th><th class="num">Output</th><th class="num">No cache (ms)</th><th class="num">KV (ms)</th><th class="num">KV prefill (ms)</th><th class="num">No cache tok/s</th><th class="num">KV tok/s</th><th class="num">Speedup</th><th>Same tokens</th></tr>
    ${kv.grid.map((r) => `<tr><td class="num">${r.input_len}</td><td class="num">${r.output_len}</td><td class="num">${fmt(r.no_cache_ms, 0)}</td><td class="num">${fmt(r.kv_ms, 0)}</td>
      <td class="num">${fmt(r.kv_prefill_ms, 1)}</td><td class="num">${fmt(r.no_cache_tok_s, 1)}</td><td class="num">${fmt(r.kv_tok_s, 1)}</td><td class="num">×${fmt(r.speedup, 2)}</td>
      <td>${r.tokens_equal ? "✓" : `✗ from token ${r.first_mismatch} (fp16 near-tie)`}</td></tr>`).join("")}</table></div></div>`;
}
function attnCard(a) {
  if (!a) return "";
  return `<div class="card"><h2>Task 4: tiled online-softmax attention <span class="hint">${esc(a.device || "")}</span></h2>
    <p>Correct within tolerance on <b>${a.n_pass}/${a.n}</b> configurations (worst |Δ| fp32 ${fmt(a.worst.float32, 7)}, fp16 ${fmt(a.worst.float16, 5)}).
    ${Object.entries(a.model_end_to_end || {}).map(([dt, v]) => `Whole model on tiled attention (${dt}): same greedy tokens ${v.tokens_equal}/${v.n}.`).join(" ")}</p>
    <div class="tbl-wrap"><table class="tbl"><tr><th class="num">Seq len</th><th class="num">Reference ms</th><th class="num">Tiled ms</th><th class="num">Fused SDPA ms</th><th class="num">Reference MB</th><th class="num">Tiled MB</th><th class="num">Fused MB</th></tr>
    ${a.performance.map((r) => `<tr><td class="num">${r.seq_len}</td><td class="num">${fmt(r.reference.ms, 2)}</td><td class="num">${fmt(r.tiled_128.ms, 2)}</td><td class="num">${fmt(r.torch_sdpa_fused.ms, 2)}</td>
      <td class="num">${fmt(r.reference.peak_extra_mem_mb, 0)}</td><td class="num">${fmt(r.tiled_128.peak_extra_mem_mb, 0)}</td><td class="num">${fmt(r.torch_sdpa_fused.peak_extra_mem_mb, 0)}</td></tr>`).join("")}</table></div>
    <p class="hint">The tiled kernel never stores the full score matrix, so its memory grows linearly instead of quadratically; it is slower than the reference because it is a Python loop over tiles, while fused SDPA does the same algorithm in one CUDA kernel.</p></div>`;
}
function specCard(s) {
  if (!s) return "";
  const cm = s.cost_model;
  const costs = Object.entries(s.draft_costs || {}).filter(([, v]) => v.c !== undefined && v.c !== null)
    .map(([d, v]) => `${esc(d)}: one draft step = ${fmt(v.c, 2)} × a target step`).join(" · ");
  return `<div class="card"><h2>Task 5: speculative decoding <span class="hint">${esc(s.device || "")}, fp16</span></h2>
    <p>Target greedy baseline: ${fmt(s.baseline.task.tokens_per_s, 1)} tok/s (task prompts), ${fmt(s.baseline.summary.tokens_per_s, 1)} tok/s (note summaries).
    ${cm ? `Cost model: target step ${fmt(cm.target_step_ms, 1)} ms. ` : ""}${costs}</p>
    <div class="tbl-wrap"><table class="tbl"><tr><th>Drafter</th><th class="num">k</th><th>Prompts</th><th>Same as greedy</th><th class="num">Acceptance</th><th class="num">Tok/target pass</th><th class="num">Tok/s</th><th class="num">Speedup</th><th class="num">Predicted</th></tr>
    ${s.rows.map((r) => `<tr><td>${esc(r.draft)}</td><td class="num">${r.k}</td><td>${r.prompts}</td><td>${r.equal}</td><td class="num">${fmt(r.acceptance, 2)}</td>
      <td class="num">${fmt(r.tok_per_target_pass, 2)}</td><td class="num">${fmt(r.tok_s, 1)}</td><td class="num ${r.speedup > 1 ? "best" : ""}">×${fmt(r.speedup, 2)}</td>
      <td class="num">${r.predicted_speedup ? "×" + fmt(r.predicted_speedup, 2) : "–"}</td></tr>`).join("")}</table></div>
    <p class="hint">Predicted = E(α,k) / (v_k + k·c): expected tokens per iteration over the cost of one iteration in target-step units. A draft whose step costs close to a target step cannot win at batch size 1, however high its acceptance.</p></div>`;
}
function figCard(figs) {
  if (!figs || !figs.length) return "";
  const pick = figs.filter((f) => /benchmark|compare_baseline_test|cm_ft_v2_test_greedy_overall|training_curve_lora_qwen25_1p5b_v2/.test(f));
  return `<div class="card"><h2>Figures</h2><div class="figs">${(pick.length ? pick : figs).map((f) =>
    `<figure><a href="/figures/${encodeURIComponent(f)}" target="_blank" rel="noopener"><img loading="lazy" src="/figures/${encodeURIComponent(f)}" alt="${esc(f)}"></a><figcaption>${esc(f.replace(".png", "").replace(/_/g, " "))}</figcaption></figure>`).join("")}</div></div>`;
}

// ------------------------------------------------------------------ start
pollHealth();
loadExamples();

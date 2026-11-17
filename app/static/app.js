/* 相机陷阱连拍重复帧核查台 —— 前端逻辑 */
"use strict";

const DECISION_META = {
  keep:   { label: "保留", cls: "keep" },
  omit:   { label: "建议省略", cls: "omit" },
  review: { label: "待核", cls: "review" },
};
const LABEL_META = {
  duplicate: { text: "重复", cls: "dup" },
  different: { text: "内容变化", cls: "diff" },
  unknown:   { text: "证据不足/灰区", cls: "unk" },
};

const state = {
  sessionId: null,
  version: 0,
  nFrames: 0,
  detections: [],
  images: [],
  currentFrame: 0,
  report: null,
  jobId: null,
  pollTimer: null,
};

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

/* ---------------------------------------------------------------- API */

async function api(path, options = {}) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  return resp.json();
}

function showError(targetId, err) {
  const el = $(targetId);
  el.textContent = `错误：${err.message}`;
  el.classList.remove("muted");
  el.style.color = "#ff9d97";
}

/* ------------------------------------------------------------ 会话载入 */

function resetSuggestion(bannerText) {
  state.report = null;
  state.jobId = null;
  if (state.pollTimer) { clearTimeout(state.pollTimer); state.pollTimer = null; }
  const banner = $("result-banner");
  banner.className = "banner stale";
  banner.textContent = bannerText;
  $("verdict-summary").innerHTML = "";
  $("verdict-list").innerHTML = "";
  $("pair-list").innerHTML = "";
  $("frame-verdict").className = "muted";
  $("frame-verdict").textContent = "当前帧尚无建议。";
}

function loadSession(payload) {
  // 切换会话：立即撤下旧建议，旧任务的迟到结果之后会被会话/版本校验拦截
  resetSuggestion("已切换会话，旧建议已撤下。");
  state.sessionId = payload.session_id;
  state.version = payload.version;
  state.nFrames = payload.n_frames;
  state.detections = payload.detections;
  state.currentFrame = 0;

  state.images = [];
  for (let i = 0; i < state.nFrames; i++) {
    const img = new Image();
    img.src = `/api/sessions/${state.sessionId}/frames/${i}`;
    img.onload = () => { if (i === state.currentFrame) draw(); };
    state.images.push(img);
  }

  const info = $("session-info");
  info.classList.add("muted");
  info.style.color = "";
  info.innerHTML =
    `会话 <b>${payload.session_id}</b>（输入版本 v${payload.version}）：` +
    `${payload.n_frames} 帧、${payload.detections.length} 个检测框。` +
    (payload.description ? `<br>${payload.description}` : "");

  $("analyze-btn").disabled = false;
  $("no-session").style.display = "none";
  renderFrameNav();
  draw();
}

async function uploadSession() {
  const imgs = $("img-input").files;
  const jsons = $("json-input").files;
  if (imgs.length === 0 || imgs.length > 6) {
    alert("请选择 1~6 张帧图片。");
    return;
  }
  if (jsons.length !== 1) {
    alert("请选择 1 个检测 JSON 文件。");
    return;
  }
  const fd = new FormData();
  for (const f of imgs) fd.append("images", f);
  fd.append("detections", jsons[0]);
  try {
    const payload = await api("/api/sessions", { method: "POST", body: fd });
    loadSession(payload);
  } catch (err) {
    showError("session-info", err);
  }
}

async function loadExample() {
  try {
    loadSession(await api("/api/example", { method: "POST" }));
  } catch (err) {
    showError("session-info", err);
  }
}

/* ------------------------------------------------------------ 画面渲染 */

function detsInFrame(f) {
  return state.detections.filter((d) => d.frame === f);
}

function verdictOf(f) {
  if (!state.report) return null;
  return state.report.verdicts.find((v) => v.frame === f) || null;
}

function draw() {
  const img = state.images[state.currentFrame];
  if (!img || !img.complete || img.naturalWidth === 0) return;
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);

  for (const d of detsInFrame(state.currentFrame)) {
    const [x, y, w, h] = d.rect;
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffe600";
    ctx.strokeRect(x, y, w, h);
    ctx.font = "bold 12px sans-serif";
    const label = `#${d.id}`;
    const tw = ctx.measureText(label).width;
    const ly = y >= 16 ? y - 16 : y + h + 2;
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(x, ly, tw + 6, 15);
    ctx.fillStyle = "#ffe600";
    ctx.fillText(label, x + 3, ly + 12);
  }
  renderFrameVerdict();
}

function renderFrameNav() {
  const n = state.nFrames || 0;
  $("frame-indicator").textContent =
    n === 0 ? "— / —" : `第 ${state.currentFrame + 1} / ${n} 帧（本帧 ${detsInFrame(state.currentFrame).length} 框）`;
  $("prev-frame").disabled = n === 0 || state.currentFrame === 0;
  $("next-frame").disabled = n === 0 || state.currentFrame >= n - 1;
}

function renderFrameVerdict() {
  const el = $("frame-verdict");
  const v = verdictOf(state.currentFrame);
  if (!v) {
    el.className = "muted";
    el.textContent = "当前帧尚无建议。";
    return;
  }
  const meta = DECISION_META[v.decision];
  el.className = `frame-verdict ${meta.cls}`;
  const dup = v.duplicate_of !== null && v.duplicate_of !== undefined
    ? `（与最早帧 帧 ${v.duplicate_of} 重复）` : "";
  el.innerHTML = `<span class="badge ${meta.cls}">${meta.label}</span> ${dup} ${v.reason}`;
}

function switchFrame(delta) {
  const next = state.currentFrame + delta;
  if (next < 0 || next >= state.nFrames) return;
  state.currentFrame = next;
  renderFrameNav();
  draw();
}

/* ------------------------------------------------------------ 分析任务 */

async function analyze() {
  if (!state.sessionId) return;
  const status = $("analyze-status");
  status.className = "muted small";
  status.style.color = "";
  resetSuggestion("分析进行中，旧建议已撤下。");
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/analyze`, { method: "POST" });
    state.jobId = resp.job_id;
    status.textContent = `分析中（输入版本 v${resp.version}）…`;
    if (state.pollTimer) clearTimeout(state.pollTimer);
    pollJob(resp.job_id, resp.session_id, resp.version);
  } catch (err) {
    showError("analyze-status", err);
  }
}

async function pollJob(jobId, sessionId, jobVersion) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    // 会话已切换：旧任务结果一律丢弃
    if (jobId !== state.jobId || sessionId !== state.sessionId) return;
    if (job.status === "running") {
      state.pollTimer = setTimeout(() => pollJob(jobId, sessionId, jobVersion), 400);
      return;
    }
    const status = $("analyze-status");
    if (job.status === "error") {
      status.style.color = "#ff9d97";
      status.textContent = job.error || "分析失败。";
      return;
    }
    // 迟到结果双重校验：任务必须属于当前会话且版本一致
    if (job.session_id !== state.sessionId || job.version !== state.version) {
      status.style.color = "#ffd887";
      status.textContent =
        `任务完成但基于旧输入（v${job.version}，当前 v${state.version}），结果已作废，未覆盖当前会话。`;
      const banner = $("result-banner");
      banner.className = "banner stale";
      banner.textContent = "迟到的旧结果已自动丢弃。";
      return;
    }
    const result = job.result;
    if (result && result.status === "failed") {
      status.style.color = "#ff9d97";
      status.textContent = result.message;
      return;
    }
    status.className = "muted small";
    status.textContent = `分析完成（输入版本 v${job.version}）。建议仅作核查参考，原图与检测框未被修改。`;
    state.report = result;
    renderReport();
    draw();
  } catch (err) {
    showError("analyze-status", err);
  }
}

/* ------------------------------------------------------------ 结果渲染 */

function fmt(v, digits = 3) {
  return v === null || v === undefined ? "—" : Number(v).toFixed(digits);
}

function regionBlock(title, r) {
  if (!r) return "";
  const head = r.sufficient
    ? `${title}：证据充分`
    : `${title}：<span class="weak">证据不足</span>`;
  const detail = r.disjoint
    ? "检测框无交集（位置变化强证据）"
    : `RMSE ${fmt(r.rmse)} · 相关 ${fmt(r.corr, 4)} · 姿态 ${fmt(r.hog_dist, 4)} · 能量 ${fmt(r.energy)} · ${r.pixels}px`;
  const reason = r.reason ? `<div class="region-reason">${r.reason}</div>` : "";
  return `<div class="region ${r.sufficient ? "" : "insufficient"}">
    <div>${head}</div><div class="metric">${detail}</div>${reason}
  </div>`;
}

function renderReport() {
  const r = state.report;
  const banner = $("result-banner");
  if (!r) return;

  const counts = { keep: 0, omit: 0, review: 0 };
  for (const v of r.verdicts) counts[v.decision] = (counts[v.decision] || 0) + 1;
  banner.className = "banner ok";
  banner.textContent =
    `分析完成：${counts.keep} 帧保留、${counts.omit} 帧建议省略、${counts.review} 帧待核。`;

  $("verdict-summary").innerHTML =
    `<div class="summary-line">
       <span class="chip keep">保留 ${counts.keep}</span>
       <span class="chip omit">省略 ${counts.omit}</span>
       <span class="chip review">待核 ${counts.review}</span>
     </div>
     <div class="muted small">自动省略仅在背景与动物区域都有充分可辨证据、且重复组内每两帧均相符时发生；其余情形一律保守处理。</div>`;

  const vlist = $("verdict-list");
  vlist.innerHTML = "";
  r.verdicts.forEach((v) => {
    const meta = DECISION_META[v.decision];
    const row = document.createElement("div");
    row.className = `verdict-row ${meta.cls}`;
    row.innerHTML =
      `<span class="badge ${meta.cls}">${meta.label}</span>` +
      `<span class="frame-name">帧 ${v.frame}</span>` +
      (v.duplicate_of !== null && v.duplicate_of !== undefined
        ? `<span class="muted small">重复于最早帧 帧 ${v.duplicate_of}</span>` : "") +
      `<span class="reason-text">${v.reason}</span>`;
    row.addEventListener("click", () => {
      state.currentFrame = v.frame;
      renderFrameNav();
      draw();
    });
    vlist.appendChild(row);
  });

  const plist = $("pair-list");
  plist.innerHTML = "";
  r.pairs.forEach((p) => {
    const meta = LABEL_META[p.label];
    const div = document.createElement("div");
    div.className = "pair-card";
    div.innerHTML =
      `<div class="pair-head">
         <span class="pair-name">帧 ${p.a} ↔ 帧 ${p.b}</span>
         <span class="pair-label ${meta.cls}">${meta.text}</span>
       </div>
       <div class="pair-regions">${regionBlock("背景", p.background)}${regionBlock("动物", p.animal)}</div>
       <div class="pair-reason">${p.reason}</div>`;
    plist.appendChild(div);
  });
}

/* ------------------------------------------------------------ 事件绑定 */

$("upload-btn").addEventListener("click", uploadSession);
$("example-btn").addEventListener("click", loadExample);
$("prev-frame").addEventListener("click", () => switchFrame(-1));
$("next-frame").addEventListener("click", () => switchFrame(1));
$("analyze-btn").addEventListener("click", analyze);
document.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT") return;
  if (ev.key === "ArrowLeft") switchFrame(-1);
  if (ev.key === "ArrowRight") switchFrame(1);
});

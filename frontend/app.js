/**
 * VoxPrompt — agentic console.
 *
 * Pipeline: capture audio -> POST /transcribe -> POST /analyze -> POST /build.
 * Every step is inspectable and editable before the next one runs, because the
 * models paraphrase and mishear and the user is the authority on what they meant.
 *
 * Sessions are cached in localStorage. That is per-browser and never reaches
 * the server, which keeps the "nothing leaves your machine" promise intact.
 */
"use strict";

const MAX_SECONDS = 300;      // 5 minutes, per the product brief.
const WARN_SECONDS = 270;
const WAVE_BARS = 56;
const MAX_VOCABULARY_CHARS = 600;   // Mirrors backend/config.py.
const SESSION_KEY = "voxprompt.sessions.v1";
const MAX_SESSIONS = 30;

/** Rough speed factor per model size, measured on a CPU-only Intel Mac. */
const MODEL_SPEED = {
  tiny: { rtf: 0.06, note: "fastest, least accurate" },
  base: { rtf: 0.10, note: "fast, struggles with jargon" },
  small: { rtf: 0.40, note: "recommended — much better on names" },
  medium: { rtf: 1.20, note: "slowest, marginal gain on CPU" },
};

/**
 * Field order, labels, and whether each holds a list.
 * List fields are edited as one item per line.
 */
const FIELDS = {
  goal: { label: "Goal", list: false },
  audience: { label: "Audience", list: false },
  context: { label: "Context", list: false },
  constraints: { label: "Constraints", list: true },
  examples: { label: "Examples", list: true },
  output_format: { label: "Output format", list: false },
  success_criteria: { label: "Success criteria", list: true },
};

/**
 * Mirrors NULL_STRINGS in backend/analyze.py.
 *
 * The backend decides what is missing on arrival; once the user starts
 * editing, the browser has to make the same judgement live. A test asserts
 * these two lists stay identical.
 */
const NULL_STRINGS = new Set([
  "null", "none", "nil", "n/a", "na", "-", "--", "",
  "unknown", "unspecified", "not specified", "not mentioned",
  "not stated", "not provided", "not applicable", "no information",
]);

/** Mirrors is_empty() in backend/analyze.py. */
function isEmptyValue(value) {
  if (value === null || value === undefined) return true;
  if (Array.isArray(value)) return value.every(isEmptyValue);
  return NULL_STRINGS.has(String(value).trim().toLowerCase().replace(/\.+$/, ""));
}

const el = (id) => document.getElementById(id);

const ui = {
  sessionList: el("sessionList"), sessionEmpty: el("sessionEmpty"),
  newSessionBtn: el("newSessionBtn"),
  statEngine: el("statEngine"), statWhisper: el("statWhisper"), statLatency: el("statLatency"),
  sandboxLine: el("sandboxLine"), statusDot: el("statusDot"), statusLabel: el("statusLabel"),
  stream: el("stream"), banner: el("banner"),
  audioCard: el("audioCard"), wave: el("wave"), audioName: el("audioName"), audioTime: el("audioTime"),
  workflow: el("workflow"), workflowPill: el("workflowPill"), workflowNote: el("workflowNote"),
  resultCard: el("resultCard"), resultMeta: el("resultMeta"), transcript: el("transcript"),
  charCount: el("charCount"), copyBtn: el("copyBtn"), analyzeBtn: el("analyzeBtn"),
  analyzeCard: el("analyzeCard"), analyzeMeta: el("analyzeMeta"), analyzeResult: el("analyzeResult"),
  chipRow: el("chipRow"), fieldList: el("fieldList"), fieldStatus: el("fieldStatus"),
  copyJsonBtn: el("copyJsonBtn"), buildBtn: el("buildBtn"), rawJson: el("rawJson"),
  buildCard: el("buildCard"), buildMeta: el("buildMeta"), buildResult: el("buildResult"),
  promptOutput: el("promptOutput"), promptStats: el("promptStats"),
  refineBtn: el("refineBtn"), copyPromptBtn: el("copyPromptBtn"),
  statusCard: el("statusCard"), spinner: el("spinner"),
  statusText: el("statusText"), statusHint: el("statusHint"),
  fileInput: el("fileInput"), fileName: el("fileName"), recorderHint: el("recorderHint"),
  recordBtn: el("recordBtn"), recordLabel: el("recordLabel"), meterFill: el("meterFill"),
  timer: el("timer"), processBtn: el("processBtn"),
  modelSelect: el("modelSelect"), modelHint: el("modelHint"), targetModel: el("targetModel"),
  vocabulary: el("vocabulary"), vocabHint: el("vocabHint"),
  cacheSize: el("cacheSize"), clearCacheBtn: el("clearCacheBtn"),
};

/** Recorder state, grouped so it is obvious what a reset clears. */
const rec = {
  recorder: null, chunks: [], stream: null,
  audioCtx: null, analyser: null, meterRaf: null,
  timerId: null, startedAt: 0, levels: [],
};

/** The audio waiting to be processed, and the session it belongs to. */
let pending = null;              // { blob, filename, seconds, levels }
let warmModels = new Set();
let current = newSession();

// ---------------------------------------------------------------------------
// Session cache (localStorage)
// ---------------------------------------------------------------------------

function newSession() {
  return {
    id: `s-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    title: "Untitled session",
    created: Date.now(),
    audioName: "", seconds: 0, levels: [],
    transcript: "", transcriptMeta: "",
    extraction: null, analyzeMeta: "",
    prompt: "", promptStats: "", buildMeta: "",
  };
}

/**
 * Read the cache.
 *
 * Every access is guarded: localStorage throws in private mode and in some
 * embedded webviews, and a broken cache must never take the app down with it.
 */
function loadSessions() {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveSessions(sessions) {
  try {
    localStorage.setItem(SESSION_KEY, JSON.stringify(sessions.slice(0, MAX_SESSIONS)));
    return true;
  } catch {
    // Most likely the quota: waveforms and prompts add up.
    return false;
  }
}

/** Persist the in-progress session, newest first. */
function persistCurrent() {
  if (!current.transcript) return;   // Nothing worth remembering yet.

  current.title = deriveTitle(current);
  const sessions = loadSessions().filter((s) => s.id !== current.id);
  sessions.unshift(current);

  if (!saveSessions(sessions)) {
    showBanner("Could not save this session — the browser cache is full.", "warn");
  }
  renderSessions();
  renderCacheSize();
}

/** A readable name: the extracted goal if there is one, else the first words. */
function deriveTitle(session) {
  const goal = session.extraction?.goal;
  const source = (goal && String(goal).trim()) || session.transcript;
  if (!source) return "Untitled session";

  const words = source.trim().split(/\s+/).slice(0, 6).join(" ");
  return words.length > 46 ? `${words.slice(0, 46)}…` : words;
}

function renderSessions() {
  const sessions = loadSessions();
  ui.sessionList.innerHTML = "";
  ui.sessionEmpty.hidden = sessions.length > 0;

  for (const session of sessions) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "session-item" + (session.id === current.id ? " active" : "");
    button.innerHTML =
      '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" ' +
      'stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>';

    const title = document.createElement("span");
    title.className = "session-title";
    title.textContent = session.title;
    button.append(title);

    button.addEventListener("click", () => restoreSession(session.id));
    item.append(button);
    ui.sessionList.append(item);
  }
}

function renderCacheSize() {
  let bytes = 0;
  try {
    bytes = new Blob([localStorage.getItem(SESSION_KEY) || ""]).size;
  } catch {
    bytes = 0;
  }
  ui.cacheSize.textContent =
    bytes > 1_048_576 ? `${(bytes / 1_048_576).toFixed(1)} MB` : `${Math.ceil(bytes / 1024)} KB`;
}

/** Reopen a cached session, restoring every stage that had been reached. */
function restoreSession(id) {
  const session = loadSessions().find((s) => s.id === id);
  if (!session) return;

  current = session;
  clearBanner();
  hideStatus();
  pending = null;
  ui.processBtn.disabled = true;

  if (session.levels?.length) {
    drawWave(session.levels);
    ui.audioName.textContent = session.audioName || "recording";
    ui.audioTime.textContent = formatTime(session.seconds || 0);
    ui.audioCard.hidden = false;
  } else {
    ui.audioCard.hidden = true;
  }

  ui.workflow.hidden = false;
  setWorkflow("done", "Restored from the local session cache.");

  ui.transcript.value = session.transcript || "";
  ui.resultMeta.textContent = session.transcriptMeta || "";
  ui.resultCard.hidden = !session.transcript;
  updateCharCount();

  if (session.extraction) {
    renderFields(session.extraction, []);
    ui.analyzeMeta.textContent = session.analyzeMeta || "";
    ui.analyzeResult.hidden = false;
    ui.analyzeCard.hidden = false;
  } else {
    ui.analyzeCard.hidden = true;
    ui.analyzeResult.hidden = true;
  }

  if (session.prompt) {
    ui.promptOutput.textContent = session.prompt;
    ui.promptStats.textContent = session.promptStats || "";
    ui.buildMeta.textContent = session.buildMeta || "";
    ui.buildMeta.classList.remove("stale");
    ui.buildResult.hidden = false;
    ui.buildCard.hidden = false;
  } else {
    ui.buildCard.hidden = true;
    ui.buildResult.hidden = true;
  }

  renderSessions();
  ui.stream.scrollTop = 0;
}

function startNewSession() {
  current = newSession();
  pending = null;
  clearBanner();
  hideStatus();

  ui.audioCard.hidden = true;
  ui.workflow.hidden = true;
  ui.resultCard.hidden = true;
  ui.analyzeCard.hidden = true;
  ui.analyzeResult.hidden = true;
  ui.buildCard.hidden = true;
  ui.buildResult.hidden = true;
  ui.transcript.value = "";
  ui.processBtn.disabled = true;
  ui.recordLabel.textContent = "Local recorder idle";
  ui.timer.textContent = "00:00";
  ui.fileName.textContent = "mp3, wav, m4a, webm";

  renderSessions();
}

// ---------------------------------------------------------------------------
// Small UI helpers
// ---------------------------------------------------------------------------

function showBanner(message, kind = "error") {
  ui.banner.className = `banner ${kind}`;
  ui.banner.innerHTML = message;
  ui.banner.hidden = false;
}

function clearBanner() { ui.banner.hidden = true; }

function formatTime(totalSeconds) {
  const m = Math.floor(totalSeconds / 60);
  const s = Math.floor(totalSeconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function setStatus(text, hint = "") {
  ui.statusCard.hidden = false;
  ui.statusText.textContent = text;
  ui.statusHint.textContent = hint;
}

function hideStatus() { ui.statusCard.hidden = true; }

function setWorkflow(state, note) {
  const label = { run: "RUNNING", done: "COMPLETED", fail: "FAILED" }[state] || "RUNNING";
  ui.workflowPill.textContent = label;
  ui.workflowPill.className = `pill ${state === "done" ? "done" : state === "fail" ? "fail" : ""}`;
  ui.workflowNote.textContent = note;
}

/** Render a level array as waveform bars. */
function drawWave(levels) {
  ui.wave.innerHTML = "";
  if (!levels?.length) return;

  const peak = Math.max(...levels, 0.01);
  for (const level of levels) {
    const bar = document.createElement("span");
    const height = Math.max(8, Math.round((level / peak) * 100));
    bar.style.height = `${height}%`;
    if (height > 55) bar.classList.add("hot");
    ui.wave.append(bar);
  }
}

// ---------------------------------------------------------------------------
// Level meter
// ---------------------------------------------------------------------------

function startMeter(stream) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) return;   // Meter is decorative; recording still works.

  rec.audioCtx = new AudioCtx();
  const source = rec.audioCtx.createMediaStreamSource(stream);
  rec.analyser = rec.audioCtx.createAnalyser();
  rec.analyser.fftSize = 1024;
  source.connect(rec.analyser);

  const samples = new Uint8Array(rec.analyser.fftSize);
  let lastCapture = 0;

  const tick = (now) => {
    rec.analyser.getByteTimeDomainData(samples);
    let sumSquares = 0;
    for (const sample of samples) {
      const centred = (sample - 128) / 128;
      sumSquares += centred * centred;
    }
    const rms = Math.sqrt(sumSquares / samples.length);
    const level = Math.min(100, rms * 280);

    ui.meterFill.style.width = `${level}%`;
    ui.meterFill.style.background = level > 80 ? "var(--warn)" : "var(--ok)";

    // Sample periodically to build the waveform shown after recording.
    if (now - lastCapture > 120) {
      rec.levels.push(rms);
      lastCapture = now;
    }
    rec.meterRaf = requestAnimationFrame(tick);
  };
  rec.meterRaf = requestAnimationFrame(tick);
}

function stopMeter() {
  if (rec.meterRaf) cancelAnimationFrame(rec.meterRaf);
  rec.meterRaf = null;
  if (rec.audioCtx) rec.audioCtx.close().catch(() => {});
  rec.audioCtx = null;
  rec.analyser = null;
  ui.meterFill.style.width = "0%";
}

/** Reduce the captured levels to a fixed number of bars. */
function summariseLevels(levels) {
  if (!levels.length) return [];
  const out = [];
  const bucket = levels.length / WAVE_BARS;
  for (let i = 0; i < WAVE_BARS; i += 1) {
    const slice = levels.slice(Math.floor(i * bucket), Math.max(Math.floor((i + 1) * bucket), 1));
    out.push(slice.length ? Math.max(...slice) : 0);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

function pickMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",           // Safari
    "audio/ogg;codecs=opus",
  ];
  return candidates.find((t) => MediaRecorder.isTypeSupported(t)) || "";
}

function extensionFor(mimeType) {
  if (mimeType.includes("mp4")) return "m4a";
  if (mimeType.includes("ogg")) return "ogg";
  return "webm";
}

function startTimer() {
  rec.startedAt = Date.now();
  ui.timer.classList.add("active");

  rec.timerId = setInterval(() => {
    const elapsed = (Date.now() - rec.startedAt) / 1000;
    ui.timer.textContent = formatTime(elapsed);
    if (elapsed >= MAX_SECONDS) {
      showBanner("Reached the 5 minute limit — stopping the recording.", "warn");
      stopRecording();
    } else if (elapsed >= WARN_SECONDS) {
      ui.recordLabel.textContent = `Recording — ${Math.ceil(MAX_SECONDS - elapsed)}s left`;
    }
  }, 200);
}

function stopTimer() {
  if (rec.timerId) clearInterval(rec.timerId);
  rec.timerId = null;
  ui.timer.classList.remove("active");
}

async function startRecording() {
  clearBanner();

  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    showBanner(
      "This browser cannot record audio. Use Chrome, Firefox, Edge or Safari — " +
      "or upload an audio file instead."
    );
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    // The distinction matters: one is fixable in settings, one is not.
    const denied = err.name === "NotAllowedError" || err.name === "SecurityError";
    showBanner(
      denied
        ? "Microphone access was denied. Allow it in your browser's site settings " +
          "(and in System Settings › Privacy &amp; Security › Microphone), then reload."
        : `Could not open the microphone: ${err.name}. Is another app using it?`
    );
    return;
  }

  const mimeType = pickMimeType();
  try {
    rec.recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  } catch (err) {
    stream.getTracks().forEach((t) => t.stop());
    showBanner(`Could not start the recorder: ${err.message}`);
    return;
  }

  rec.stream = stream;
  rec.chunks = [];
  rec.levels = [];

  rec.recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size > 0) rec.chunks.push(event.data);
  });

  rec.recorder.addEventListener("error", (event) => {
    showBanner(`Recording error: ${event.error?.message || "unknown"}`);
    cleanupStream();
    resetRecordButton();
  });

  rec.recorder.addEventListener("stop", () => {
    const seconds = (Date.now() - rec.startedAt) / 1000;
    const type = rec.recorder.mimeType || mimeType || "audio/webm";
    const blob = new Blob(rec.chunks, { type });
    const levels = summariseLevels(rec.levels);

    cleanupStream();
    resetRecordButton();

    if (blob.size === 0) {
      showBanner("The recording came out empty. Check your microphone and try again.");
      return;
    }
    stageAudio(blob, `recording.${extensionFor(type)}`, seconds, levels);
  });

  rec.recorder.start(250);   // Flush chunks regularly so nothing is lost.
  startTimer();
  startMeter(stream);

  ui.recordBtn.classList.add("recording");
  ui.recordBtn.setAttribute("aria-label", "Stop recording");
  ui.recordLabel.textContent = "Local recorder active";
  ui.recorderHint.textContent = "Recording — click the button again to stop.";
}

function stopRecording() {
  if (rec.recorder && rec.recorder.state !== "inactive") rec.recorder.stop();
}

function cleanupStream() {
  stopTimer();
  stopMeter();
  if (rec.stream) rec.stream.getTracks().forEach((t) => t.stop());
  rec.stream = null;
}

function resetRecordButton() {
  ui.recordBtn.classList.remove("recording");
  ui.recordBtn.setAttribute("aria-label", "Start recording");
  ui.recordLabel.textContent = "Local recorder idle";
  ui.recorderHint.textContent = "Click record to start local input streaming.";
}

/** Hold audio until the user presses Process, matching the composer flow. */
function stageAudio(blob, filename, seconds, levels) {
  pending = { blob, filename, seconds, levels };

  current.audioName = filename;
  current.seconds = seconds;
  current.levels = levels;

  drawWave(levels);
  ui.audioName.textContent = filename;
  ui.audioTime.textContent = formatTime(seconds);
  ui.audioCard.hidden = false;

  ui.processBtn.disabled = false;
  ui.recordLabel.textContent = "Ready to process";
  ui.recorderHint.textContent = "Press Process audio to transcribe locally.";
}

// ---------------------------------------------------------------------------
// Step 1 — transcription
// ---------------------------------------------------------------------------

async function processAudio() {
  if (!pending) return;

  clearBanner();
  ui.processBtn.disabled = true;
  ui.recordBtn.disabled = true;
  ui.workflow.hidden = false;
  setWorkflow("run", "Whisper is processing the audio locally…");

  const chosen = ui.modelSelect.value || "base";
  const rtf = MODEL_SPEED[chosen]?.rtf ?? 0.1;
  const estimate = Math.max(2, Math.round((pending.seconds || 30) * rtf));
  setStatus(
    `Transcribing with ${chosen}…`,
    warmModels.has(chosen)
      ? `Roughly ${estimate}s.`
      : `Roughly ${estimate}s once the model is ready. First use of ${chosen} loads it first.`
  );

  const form = new FormData();
  form.append("audio", pending.blob, pending.filename);
  form.append("model", chosen);
  if (ui.vocabulary.value.trim()) form.append("vocabulary", ui.vocabulary.value.trim());

  try {
    const response = await fetch("/transcribe", { method: "POST", body: form });
    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      setWorkflow("fail", "Transcription failed.");
      showBanner(payload?.detail || `Transcription failed (${response.status}).`);
      return;
    }
    showTranscript(payload);
  } catch (err) {
    setWorkflow("fail", "Could not reach the backend.");
    showBanner(`Could not reach the backend: ${err.message}. Is the server still running?`);
  } finally {
    hideStatus();
    ui.recordBtn.disabled = false;
    ui.processBtn.disabled = !pending;
  }
}

function showTranscript(payload) {
  ui.transcript.value = payload.text;
  ui.resultMeta.textContent = `${payload.text.length} chars`;
  ui.resultCard.hidden = false;

  // A new transcript invalidates whatever was derived from the old one.
  ui.analyzeCard.hidden = true;
  ui.analyzeResult.hidden = true;
  ui.buildCard.hidden = true;
  ui.buildResult.hidden = true;

  setWorkflow(
    "done",
    `Transcribed ${payload.duration.toFixed(1)}s of audio in ${payload.elapsed_s}s ` +
    `with ${payload.model}.`
  );
  ui.statLatency.textContent = `${payload.elapsed_s}s`;

  current.transcript = payload.text;
  current.transcriptMeta = ui.resultMeta.textContent;
  current.extraction = null;
  current.prompt = "";
  persistCurrent();

  updateCharCount();
  ui.transcript.focus();
  checkHealth();   // Refresh which models are warm.
}

function updateCharCount() {
  const text = ui.transcript.value;
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  ui.charCount.textContent = `${words} words · ${text.length} characters`;
}

// ---------------------------------------------------------------------------
// Step 2 — variable extraction
// ---------------------------------------------------------------------------

function buildFieldInput(name, value, question) {
  const spec = FIELDS[name];
  const input = document.createElement("textarea");

  input.className = "field-input";
  input.dataset.field = name;
  input.id = `field-${name}`;
  input.rows = spec.list ? 3 : 2;
  input.value = spec.list ? (value || []).join("\n") : (value ?? "");
  input.placeholder = question || (spec.list ? "One per line" : `Add ${spec.label.toLowerCase()}…`);

  input.addEventListener("input", () => {
    autoGrow(input);
    refreshAnalysisState();
  });
  return input;
}

function autoGrow(input) {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 260)}px`;
}

/** Read the edited fields back out of the DOM. The source of truth for /build. */
function readExtraction() {
  const extraction = {};
  for (const [name, spec] of Object.entries(FIELDS)) {
    const input = ui.fieldList.querySelector(`[data-field="${name}"]`);
    const raw = input ? input.value : "";

    if (spec.list) {
      extraction[name] = raw
        .split("\n")
        .map((line) => line.replace(/^[-*•]\s*/, "").trim())
        .filter((line) => !isEmptyValue(line));
    } else {
      extraction[name] = isEmptyValue(raw) ? null : raw.trim();
    }
  }
  return extraction;
}

/** Re-evaluate what is still missing after an edit, and update the UI. */
function refreshAnalysisState() {
  const extraction = readExtraction();
  let filled = 0;

  ui.chipRow.innerHTML = "";
  for (const [name, spec] of Object.entries(FIELDS)) {
    const empty = isEmptyValue(extraction[name]);
    if (!empty) filled += 1;

    const row = ui.fieldList.querySelector(`[data-row="${name}"]`);
    if (row) row.classList.toggle("is-missing", empty);

    const chip = document.createElement("span");
    chip.className = `chip ${empty ? "empty" : "filled"}`;
    chip.textContent = `${empty ? "○" : "✓"} [${name}]`;
    ui.chipRow.append(chip);
  }

  const total = Object.keys(FIELDS).length;
  ui.fieldStatus.textContent =
    filled === total
      ? `All ${total} fields filled`
      : `${filled} of ${total} filled · blanks are fine, they are simply left out`;

  ui.rawJson.textContent = JSON.stringify(extraction, null, 2);

  // A prompt built before this edit no longer matches the fields above.
  if (!ui.buildResult.hidden) {
    ui.buildMeta.textContent = "out of date — rebuild";
    ui.buildMeta.classList.add("stale");
  }

  current.extraction = extraction;
  return extraction;
}

function renderFields(extraction, missing) {
  const questionFor = new Map((missing || []).map((m) => [m.field, m.question]));

  ui.fieldList.innerHTML = "";
  for (const [name, spec] of Object.entries(FIELDS)) {
    const row = document.createElement("li");
    row.dataset.row = name;

    const label = document.createElement("label");
    label.className = "field-name";
    label.textContent = spec.label;
    label.htmlFor = `field-${name}`;

    row.append(label, buildFieldInput(name, extraction[name], questionFor.get(name)));
    ui.fieldList.append(row);
  }

  refreshAnalysisState();
  ui.fieldList.querySelectorAll(".field-input").forEach(autoGrow);
}

async function analyzeTranscript() {
  const transcript = ui.transcript.value.trim();
  if (!transcript) {
    showBanner("There is no transcript to analyse yet.", "warn");
    return;
  }

  // Re-analysing replaces every field, so do not silently bin typed answers.
  if (!ui.analyzeResult.hidden && !window.confirm(
    "Re-extracting replaces all variables and discards your edits. Continue?"
  )) return;

  clearBanner();
  ui.analyzeBtn.disabled = true;
  setWorkflow("run", "Extracting structured variables with the local model…");
  setStatus(
    "Extracting variables…",
    "The local model reads the whole transcript. One to two minutes for a long recording."
  );

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript }),
    });
    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      setWorkflow("fail", "Extraction failed.");
      showBanner(payload?.detail || `Extraction failed (${response.status}).`);
      return;
    }

    ui.analyzeCard.hidden = false;
    renderFields(payload.extraction, payload.missing);
    ui.analyzeMeta.textContent = `${payload.elapsed_s}s · ${payload.model}`;
    ui.analyzeResult.hidden = false;
    setWorkflow("done", `Structured into ${Object.keys(FIELDS).length} variables.`);

    current.transcript = transcript;
    current.analyzeMeta = ui.analyzeMeta.textContent;
    persistCurrent();
  } catch (err) {
    setWorkflow("fail", "Could not reach the backend.");
    showBanner(`Could not reach the backend: ${err.message}`);
  } finally {
    hideStatus();
    ui.analyzeBtn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Step 3 — prompt assembly
// ---------------------------------------------------------------------------

async function buildPrompt() {
  clearBanner();
  ui.buildBtn.disabled = true;

  try {
    const response = await fetch("/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ extraction: readExtraction() }),
    });
    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      showBanner(payload?.detail || `Could not build the prompt (${response.status}).`);
      return;
    }

    ui.buildCard.hidden = false;
    ui.promptOutput.textContent = payload.prompt;
    ui.promptStats.textContent =
      `~${payload.estimated_tokens} tokens · ${payload.characters} characters ` +
      `· ${payload.sections.length} sections`;
    ui.buildMeta.textContent = payload.sections.join(", ");
    ui.buildMeta.classList.remove("stale");
    ui.buildResult.hidden = false;
    ui.buildBtn.textContent = "Rebuild prompt";

    current.prompt = payload.prompt;
    current.promptStats = ui.promptStats.textContent;
    current.buildMeta = ui.buildMeta.textContent;
    persistCurrent();

    ui.buildCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    showBanner(`Could not reach the backend: ${err.message}`);
  } finally {
    ui.buildBtn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Clipboard
// ---------------------------------------------------------------------------

async function copyText(text, button, label) {
  try {
    await navigator.clipboard.writeText(text);
    const original = button.textContent;
    button.textContent = "Copied";
    setTimeout(() => (button.textContent = original), 1400);
  } catch {
    // The clipboard API needs a secure context; say so rather than failing mutely.
    showBanner(`Could not copy the ${label} automatically. Select it and press Cmd+C.`, "warn");
  }
}

// ---------------------------------------------------------------------------
// Settings + health
// ---------------------------------------------------------------------------

function populateModels(health = null) {
  const options = health?.available_models?.length
    ? health.available_models
    : Object.keys(MODEL_SPEED);

  warmModels = new Set(health?.loaded_models ?? []);
  const previous = ui.modelSelect.value;

  ui.modelSelect.innerHTML = "";
  for (const name of options) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = warmModels.has(name) ? `${name} (ready)` : name;
    ui.modelSelect.append(option);
  }

  const fallback = options.includes("small") ? "small" : options[0];
  ui.modelSelect.value = options.includes(previous) ? previous : fallback;
  updateModelHint();
}

function updateModelHint() {
  const info = MODEL_SPEED[ui.modelSelect.value];
  if (!info) return;
  ui.modelHint.textContent = `${info.note} · ~${Math.round(info.rtf * 300)}s per 5 min of audio`;
}

/**
 * Warn about the LLM before someone records five minutes for nothing.
 *
 * A reachable Ollama without the configured model fails only at the extraction
 * step, so the two cases are reported separately and specifically.
 */
function reportLlmState(health) {
  if (!health.ffmpeg) {
    showBanner(
      "<strong>ffmpeg is not installed.</strong> Transcription will fail until you run " +
      "<code>brew install ffmpeg</code> and restart the server."
    );
    return;
  }
  if (!health.ollama) {
    showBanner(
      "<strong>Ollama is not running</strong>, so variable extraction will fail. " +
      "Start it with <code>ollama serve</code>. Recording still works.",
      "warn"
    );
  } else if (!health.ollama_model_available) {
    showBanner(
      `<strong>Model <code>${health.ollama_model}</code> is not pulled.</strong> ` +
      `Extraction will fail until you run <code>ollama pull ${health.ollama_model}</code>.`,
      "warn"
    );
  }
}

function setSandboxStatus(health) {
  const problems = [];
  if (!health.ffmpeg) problems.push("ffmpeg missing");
  if (!health.ollama) problems.push("Ollama offline");
  else if (!health.ollama_model_available) problems.push("model not pulled");

  const ok = problems.length === 0;
  ui.statusDot.className = `dot-status ${ok ? "ok" : health.ffmpeg ? "warn" : "bad"}`;
  ui.statusLabel.textContent = ok ? "Sandbox secure" : problems.join(", ");
  ui.sandboxLine.textContent = health.model_warming
    ? `Loading ${health.whisper_model}… everything runs on this machine.`
    : `Whisper (${health.whisper_model}) and ${health.ollama_model}, running entirely offline.`;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const health = await response.json();

    populateModels(health);
    setSandboxStatus(health);
    reportLlmState(health);

    ui.targetModel.textContent = health.ollama_model || "not configured";
    // Real values, not decoration: this is what is actually running.
    ui.statEngine.textContent = "CTranslate2 int8 · CPU";
    ui.statWhisper.textContent = health.model_warming
      ? `${health.whisper_model} (loading)`
      : health.whisper_model;

    if (health.model_warming) setTimeout(checkHealth, 5000);
  } catch {
    ui.statusDot.className = "dot-status bad";
    ui.statusLabel.textContent = "Backend unreachable";
    ui.statEngine.textContent = "offline";
  }
}

/**
 * Warn when the hint stops looking like a word list.
 *
 * The hint shares Whisper's 224-token window with the audio context, so prose
 * here biases the decoder instead of sharpening rare words. The backend
 * truncates regardless; this makes the limit visible rather than silent.
 */
function updateVocabHint() {
  const value = ui.vocabulary.value;
  const tooLong = value.length > MAX_VOCABULARY_CHARS;
  const looksLikeProse = value.split(/\s+/).filter(Boolean).length > 40;

  ui.vocabHint.classList.toggle("over-limit", tooLong || looksLikeProse);

  if (tooLong) {
    ui.vocabHint.innerHTML =
      `Too long — only the first ${MAX_VOCABULARY_CHARS} characters are used ` +
      `(${value.length} entered). Keep it to names and jargon.`;
  } else if (looksLikeProse) {
    ui.vocabHint.innerHTML =
      "This looks like prose. Use only the <strong>terms</strong> Whisper mishears " +
      "— sentences here can bias the transcript.";
  } else {
    ui.vocabHint.innerHTML =
      "Custom technical keywords or spelling targets to bias the Whisper engine. " +
      "Comma-separated <strong>terms</strong>, not sentences.";
  }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

ui.recordBtn.addEventListener("click", () => {
  if (rec.recorder && rec.recorder.state === "recording") stopRecording();
  else startRecording();
});

ui.processBtn.addEventListener("click", processAudio);
ui.analyzeBtn.addEventListener("click", analyzeTranscript);
ui.buildBtn.addEventListener("click", buildPrompt);

ui.fileInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

  ui.fileName.textContent = file.name;
  // Uploads carry no level data, so the waveform stays empty until transcribed.
  stageAudio(file, file.name, 0, []);
  ui.audioTime.textContent = `${(file.size / 1_048_576).toFixed(1)} MB`;
  event.target.value = "";   // Allow re-picking the same file.
});

ui.transcript.addEventListener("input", () => {
  updateCharCount();
  current.transcript = ui.transcript.value;
});

ui.refineBtn.addEventListener("click", () => {
  ui.analyzeCard.scrollIntoView({ behavior: "smooth", block: "start" });
  ui.fieldList.querySelector(".field-input")?.focus();
});

ui.copyBtn.addEventListener("click", () => copyText(ui.transcript.value, ui.copyBtn, "transcript"));
ui.copyJsonBtn.addEventListener("click", () =>
  copyText(JSON.stringify(readExtraction(), null, 2), ui.copyJsonBtn, "JSON"));
ui.copyPromptBtn.addEventListener("click", () =>
  copyText(ui.promptOutput.textContent, ui.copyPromptBtn, "prompt"));

ui.modelSelect.addEventListener("change", updateModelHint);
ui.vocabulary.addEventListener("input", updateVocabHint);
ui.newSessionBtn.addEventListener("click", startNewSession);

ui.clearCacheBtn.addEventListener("click", () => {
  if (!window.confirm("Delete every saved session from this browser? This cannot be undone.")) return;
  try {
    localStorage.removeItem(SESSION_KEY);
  } catch {
    /* Nothing to clear if storage is unavailable. */
  }
  startNewSession();
  renderCacheSize();
});

// Populate from the built-in list first so the controls work immediately,
// then let /health refine them.
populateModels();
renderSessions();
renderCacheSize();
checkHealth();

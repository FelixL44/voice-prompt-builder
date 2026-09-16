/**
 * Voice Prompt Builder -- recording UI (v0.1).
 *
 * Records with MediaRecorder, shows a timer and a live level meter, posts the
 * blob to /transcribe, and drops the result into an editable textarea.
 */
"use strict";

const MAX_SECONDS = 300; // 5 minutes, per the product brief.
const WARN_SECONDS = 270; // Turn the timer amber for the last 30s.

const el = {
  banner: document.getElementById("banner"),
  recordBtn: document.getElementById("recordBtn"),
  recordLabel: document.getElementById("recordLabel"),
  recorderHint: document.getElementById("recorderHint"),
  timer: document.getElementById("timer"),
  meterFill: document.getElementById("meterFill"),
  fileInput: document.getElementById("fileInput"),
  fileName: document.getElementById("fileName"),
  statusCard: document.getElementById("statusCard"),
  statusText: document.getElementById("statusText"),
  statusHint: document.getElementById("statusHint"),
  spinner: document.getElementById("spinner"),
  resultCard: document.getElementById("resultCard"),
  resultMeta: document.getElementById("resultMeta"),
  transcript: document.getElementById("transcript"),
  charCount: document.getElementById("charCount"),
  copyBtn: document.getElementById("copyBtn"),
  healthInfo: document.getElementById("healthInfo"),
  modelSelect: document.getElementById("modelSelect"),
  modelHint: document.getElementById("modelHint"),
  vocabulary: document.getElementById("vocabulary"),
  vocabHint: document.getElementById("vocabHint"),
  analyzeCard: document.getElementById("analyzeCard"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  analyzeMeta: document.getElementById("analyzeMeta"),
  analyzeResult: document.getElementById("analyzeResult"),
  fieldList: document.getElementById("fieldList"),
  rawJson: document.getElementById("rawJson"),
};

/** Field order and labels for the structure view. */
const FIELD_LABELS = {
  goal: "Goal",
  audience: "Audience",
  context: "Context",
  constraints: "Constraints",
  examples: "Examples",
  output_format: "Output format",
  success_criteria: "Success criteria",
};

/** Mirrors MAX_VOCABULARY_CHARS in backend/config.py. */
const MAX_VOCABULARY_CHARS = 600;

/** Rough speed factor per model size, measured on a CPU-only Intel Mac. */
const MODEL_SPEED = {
  tiny: { rtf: 0.06, note: "fastest, least accurate" },
  base: { rtf: 0.10, note: "fast, struggles with jargon" },
  small: { rtf: 0.40, note: "recommended \u2014 much better on names" },
  medium: { rtf: 1.20, note: "slowest, marginal gain on CPU" },
};

/** Mutable recording state, grouped so it is obvious what gets reset. */
const state = {
  recorder: null,
  chunks: [],
  stream: null,
  audioCtx: null,
  analyser: null,
  meterRaf: null,
  timerId: null,
  startedAt: 0,
  busy: false,
};

/** Duration of the most recent recording, used to estimate transcription time. */
let lastDurationSeconds = 0;

// ---------------------------------------------------------------------------
// Small UI helpers
// ---------------------------------------------------------------------------

function showBanner(message, kind = "error") {
  el.banner.className = `banner ${kind}`;
  el.banner.innerHTML = message;
  el.banner.hidden = false;
}

function clearBanner() {
  el.banner.hidden = true;
}

function formatTime(totalSeconds) {
  const m = Math.floor(totalSeconds / 60);
  const s = Math.floor(totalSeconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function setStatus(text, hint = "") {
  el.statusCard.hidden = false;
  el.spinner.hidden = false;
  el.statusText.textContent = text;
  el.statusHint.textContent = hint;
}

function hideStatus() {
  el.statusCard.hidden = true;
}

/** Disable inputs while a transcription is in flight. */
function setBusy(busy) {
  state.busy = busy;
  el.recordBtn.disabled = busy;
  el.fileInput.disabled = busy;
}

// ---------------------------------------------------------------------------
// Level meter
// ---------------------------------------------------------------------------

/** Drive the level meter from the live stream via an AnalyserNode. */
function startMeter(stream) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) return; // Meter is decorative; recording still works without it.

  state.audioCtx = new AudioCtx();
  const source = state.audioCtx.createMediaStreamSource(stream);
  state.analyser = state.audioCtx.createAnalyser();
  state.analyser.fftSize = 1024;
  source.connect(state.analyser);

  const samples = new Uint8Array(state.analyser.fftSize);

  const tick = () => {
    state.analyser.getByteTimeDomainData(samples);
    // RMS around the 128 midpoint, scaled to something that looks alive.
    let sumSquares = 0;
    for (const sample of samples) {
      const centred = (sample - 128) / 128;
      sumSquares += centred * centred;
    }
    const rms = Math.sqrt(sumSquares / samples.length);
    const level = Math.min(100, rms * 280);
    el.meterFill.style.width = `${level}%`;
    el.meterFill.style.background = level > 80 ? "var(--warn)" : "var(--ok)";
    state.meterRaf = requestAnimationFrame(tick);
  };
  tick();
}

function stopMeter() {
  if (state.meterRaf) cancelAnimationFrame(state.meterRaf);
  state.meterRaf = null;
  if (state.audioCtx) state.audioCtx.close().catch(() => {});
  state.audioCtx = null;
  state.analyser = null;
  el.meterFill.style.width = "0%";
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

/** Pick a container the browser will actually give us. */
function pickMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4", // Safari
    "audio/ogg;codecs=opus",
  ];
  return candidates.find((t) => MediaRecorder.isTypeSupported(t)) || "";
}

function startTimer() {
  state.startedAt = Date.now();
  el.timer.classList.add("active");

  state.timerId = setInterval(() => {
    const elapsed = (Date.now() - state.startedAt) / 1000;
    lastDurationSeconds = elapsed;
    el.timer.textContent = formatTime(elapsed);
    el.timer.classList.toggle("limit", elapsed >= WARN_SECONDS);
    if (elapsed >= MAX_SECONDS) {
      showBanner("Reached the 5 minute limit &mdash; stopping the recording.", "warn");
      stopRecording();
    }
  }, 200);
}

function stopTimer() {
  if (state.timerId) clearInterval(state.timerId);
  state.timerId = null;
  el.timer.classList.remove("active", "limit");
}

async function startRecording() {
  clearBanner();

  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    showBanner(
      "This browser cannot record audio. Use Chrome, Edge or Safari &mdash; " +
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
          "(and in System Settings &rsaquo; Privacy &amp; Security &rsaquo; Microphone), then reload."
        : `Could not open the microphone: ${err.name}. Is another app using it?`
    );
    return;
  }

  const mimeType = pickMimeType();
  try {
    state.recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  } catch (err) {
    stream.getTracks().forEach((t) => t.stop());
    showBanner(`Could not start the recorder: ${err.message}`);
    return;
  }

  state.stream = stream;
  state.chunks = [];

  state.recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size > 0) state.chunks.push(event.data);
  });

  state.recorder.addEventListener("error", (event) => {
    showBanner(`Recording error: ${event.error?.message || "unknown"}`);
    cleanupStream();
    resetRecordButton();
  });

  state.recorder.addEventListener("stop", () => {
    const type = state.recorder.mimeType || mimeType || "audio/webm";
    const blob = new Blob(state.chunks, { type });
    cleanupStream();
    resetRecordButton();
    if (blob.size === 0) {
      showBanner("The recording came out empty. Check your microphone and try again.");
      return;
    }
    sendForTranscription(blob, `recording.${extensionFor(type)}`);
  });

  state.recorder.start(250); // Flush chunks regularly so nothing is lost.
  lastDurationSeconds = 0;
  startTimer();
  startMeter(stream);

  el.recordBtn.classList.add("recording");
  el.recordLabel.textContent = "Stop recording";
  el.recorderHint.textContent = "Recording\u2026 click stop when you are done.";
}

function stopRecording() {
  if (state.recorder && state.recorder.state !== "inactive") {
    state.recorder.stop(); // The "stop" handler does the rest.
  }
}

function cleanupStream() {
  stopTimer();
  stopMeter();
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
}

function resetRecordButton() {
  el.recordBtn.classList.remove("recording");
  el.recordLabel.textContent = "Start recording";
  el.recorderHint.textContent =
    "Up to 5 minutes. Just ramble — you can fix the text afterwards.";
}

function extensionFor(mimeType) {
  if (mimeType.includes("mp4")) return "m4a";
  if (mimeType.includes("ogg")) return "ogg";
  return "webm";
}

// ---------------------------------------------------------------------------
// Transcription
// ---------------------------------------------------------------------------

async function sendForTranscription(blob, filename) {
  setBusy(true);
  el.resultCard.hidden = true;
  clearBanner();

  setStatus(
    "Transcribing…",
    `${(blob.size / 1024 / 1024).toFixed(1)} MB. The first run downloads the model, ` +
    `which takes a minute; after that expect roughly a tenth of the audio length.`
  );

  const form = new FormData();
  form.append("audio", blob, filename);
  if (el.modelSelect.value) form.append("model", el.modelSelect.value);
  if (el.vocabulary.value.trim()) form.append("vocabulary", el.vocabulary.value.trim());

  try {
    const response = await fetch("/transcribe", { method: "POST", body: form });
    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      const detail = payload?.detail || `Request failed (${response.status}).`;
      showBanner(detail);
      hideStatus();
      return;
    }
    showResult(payload);
  } catch (err) {
    showBanner(
      `Could not reach the backend: ${err.message}. Is the server still running?`
    );
    hideStatus();
  } finally {
    setBusy(false);
  }
}

function showResult(payload) {
  hideStatus();
  el.transcript.value = payload.text;
  el.resultMeta.textContent =
    `${formatTime(payload.duration)} audio · ${payload.elapsed_s}s ` +
    `· ${payload.model} · ${payload.language}`;
  el.resultCard.hidden = false;
  el.analyzeCard.hidden = false;
  el.analyzeResult.hidden = true; // Stale structure would mislead.
  el.analyzeMeta.textContent = "";
  updateCharCount();
  el.transcript.focus();
}

function updateCharCount() {
  const chars = el.transcript.value.length;
  const words = el.transcript.value.trim().split(/\s+/).filter(Boolean).length;
  el.charCount.textContent = `${words} words · ${chars} characters`;
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

el.recordBtn.addEventListener("click", () => {
  if (state.busy) return;
  const isRecording = state.recorder && state.recorder.state === "recording";
  if (isRecording) stopRecording();
  else startRecording();
});

el.fileInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  el.fileName.textContent = file.name;
  el.timer.textContent = "00:00";
  lastDurationSeconds = file.size / 4000; // Rough: ~32 kbps compressed audio.
  sendForTranscription(file, file.name);
  event.target.value = ""; // Allow re-picking the same file.
});

el.transcript.addEventListener("input", updateCharCount);

el.copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(el.transcript.value);
    el.copyBtn.textContent = "Copied";
    setTimeout(() => (el.copyBtn.textContent = "Copy"), 1400);
  } catch {
    el.transcript.select(); // Clipboard API needs a secure context; fall back.
    showBanner("Could not copy automatically &mdash; the text is selected instead.", "warn");
  }
});

/** Warn about a missing ffmpeg before someone records five minutes for nothing. */
async function checkHealth() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    if (!health.ffmpeg) {
      showBanner(
        "<strong>ffmpeg is not installed.</strong> Transcription will fail until " +
        "you run <code>brew install ffmpeg</code> and restart the server."
      );
    }
    el.healthInfo.textContent = `whisper: ${health.whisper_model} · local only`;
  } catch {
    el.healthInfo.textContent = "backend unreachable";
  }
}


/**
 * Fill the model dropdown.
 *
 * Called once at load with the built-in sizes so the control is never empty,
 * then again from /health to mark which models are already warm. An empty
 * dropdown would leave the user unable to pick a model at all, so this must
 * not depend on the request succeeding.
 */
function populateModels(health = null) {
  const options = health?.available_models?.length
    ? health.available_models
    : Object.keys(MODEL_SPEED);

  // Keep the user's choice across the /health refresh.
  const previous = el.modelSelect.value;

  el.modelSelect.innerHTML = "";
  for (const name of options) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = health?.loaded_models?.includes(name)
      ? `${name} (ready)`
      : name;
    el.modelSelect.append(option);
  }

  // "small" is the best speed/accuracy trade-off on a CPU-only Mac.
  const fallback = options.includes("small") ? "small" : options[0];
  el.modelSelect.value = options.includes(previous) ? previous : fallback;
  updateModelHint();
}

function updateModelHint() {
  const info = MODEL_SPEED[el.modelSelect.value];
  if (!info) return;
  const perFiveMin = Math.round(info.rtf * 300);
  el.modelHint.textContent = `${info.note} \u00b7 ~${perFiveMin}s per 5 min of audio`;
}

el.modelSelect.addEventListener("change", updateModelHint);

/**
 * Warn when the hint stops looking like a word list.
 *
 * The hint shares Whisper's 224-token window with the audio context, and
 * pasting prose in here biases the decoder towards continuing that prose
 * instead of sharpening rare words. The backend truncates regardless; this
 * just makes the limit visible rather than silent.
 */
function updateVocabHint() {
  const value = el.vocabulary.value;
  const tooLong = value.length > MAX_VOCABULARY_CHARS;
  const looksLikeProse = value.split(/\s+/).filter(Boolean).length > 40;

  el.vocabHint.classList.toggle("over-limit", tooLong || looksLikeProse);

  if (tooLong) {
    el.vocabHint.innerHTML =
      `Too long \u2014 only the first ${MAX_VOCABULARY_CHARS} characters are used ` +
      `(${value.length} entered). Keep it to names and jargon.`;
  } else if (looksLikeProse) {
    el.vocabHint.innerHTML =
      "This looks like prose. Use only the <strong>terms</strong> Whisper " +
      "mishears \u2014 sentences here can bias the transcript.";
  } else {
    el.vocabHint.innerHTML =
      "Comma-separated <strong>terms</strong>, not sentences. " +
      "Free \u2014 costs no extra time.";
  }
}

el.vocabulary.addEventListener("input", updateVocabHint);

// Populate from the built-in list first so the control is usable immediately,
// then let /health refine it with which models are already warm.
populateModels();
checkHealth();


// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

/** Render one extracted value, which may be a string, a list, or empty. */
function renderValue(value) {
  const cell = document.createElement("div");
  cell.className = "field-value";

  if (Array.isArray(value) && value.length > 0) {
    const list = document.createElement("ul");
    for (const item of value) {
      const entry = document.createElement("li");
      entry.textContent = item;
      list.append(entry);
    }
    cell.append(list);
  } else {
    cell.textContent = value;
  }
  return cell;
}

/** Show the extraction, marking empty fields with the question to be asked. */
function showAnalysis(payload) {
  const missingByField = new Map(payload.missing.map((m) => [m.field, m.question]));

  el.fieldList.innerHTML = "";
  for (const [name, label] of Object.entries(FIELD_LABELS)) {
    const row = document.createElement("li");

    const nameCell = document.createElement("div");
    nameCell.className = "field-name";
    nameCell.textContent = label;
    row.append(nameCell);

    if (missingByField.has(name)) {
      const cell = document.createElement("div");
      cell.className = "field-value field-missing";
      cell.textContent = "not mentioned";
      const ask = document.createElement("span");
      ask.className = "ask";
      ask.textContent = missingByField.get(name);
      cell.append(ask);
      row.append(cell);
    } else {
      row.append(renderValue(payload.extraction[name]));
    }
    el.fieldList.append(row);
  }

  el.rawJson.textContent = JSON.stringify(payload.extraction, null, 2);
  el.analyzeMeta.textContent =
    `${payload.elapsed_s}s \u00b7 ${payload.model} \u00b7 ` +
    `${payload.missing.length} field(s) missing`;
  el.analyzeResult.hidden = false;
}

el.analyzeBtn.addEventListener("click", async () => {
  const transcript = el.transcript.value.trim();
  if (!transcript) {
    showBanner("There is no transcript to analyse yet.", "warn");
    return;
  }

  clearBanner();
  el.analyzeBtn.disabled = true;
  el.analyzeBtn.textContent = "Analyzing\u2026";
  setStatus(
    "Extracting structure\u2026",
    "The local model reads the whole transcript. Expect one to two minutes " +
    "for a long recording on a CPU."
  );

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript }),
    });
    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      showBanner(payload?.detail || `Analysis failed (${response.status}).`);
      return;
    }
    showAnalysis(payload);
  } catch (err) {
    showBanner(`Could not reach the backend: ${err.message}`);
  } finally {
    hideStatus();
    el.analyzeBtn.disabled = false;
    el.analyzeBtn.textContent = "Analyze transcript";
  }
});

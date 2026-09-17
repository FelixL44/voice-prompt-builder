/**
 * VoxPrompt — agentic console.
 *
 * Pipeline: capture audio -> transcribe -> extract variables -> build prompt.
 * Every step is inspectable and editable before the next one runs, because the
 * models paraphrase and mishear and the user is the authority on what they meant.
 *
 * The two slow steps run as server-side jobs, polled for progress and
 * cancellable, because on a CPU they take tens of seconds to minutes.
 *
 * Session history lives in SQLite on the server, so it survives a cleared
 * browser cache. It is still entirely local: the server talks to nobody.
 * Only the language preference is kept in localStorage.
 */
"use strict";

const MAX_SECONDS = 300;      // 5 minutes for live recording, per the brief.
const MAX_UPLOAD_SECONDS = 1800;   // Mirrors VPB_MAX_AUDIO_SECONDS.
const WARN_SECONDS = 270;
const WAVE_BARS = 56;
const MAX_VOCABULARY_CHARS = 600;   // Mirrors backend/config.py.
const MAX_ANSWER_SECONDS = 60;      // A follow-up answer is a sentence or two.


/**
 * UI copy in both languages.
 *
 * Kept as one flat table per language so a gap is obvious and testable: a
 * missing key renders as a raw identifier, so `tests/test_i18n.py` asserts both
 * tables hold exactly the same keys, and that they match the backend's set of
 * supported languages.
 *
 * Backend errors are translated by their `error` code rather than their text,
 * which is why the server can keep returning English detail strings.
 */
const STRINGS = {
  en: {
    tagline: "Agentic Prompt Engineering",
    new_session: "New session",
    recent_sessions: "Recent sessions",
    no_sessions: "No sessions yet. Record something and it will be saved here.",
    local_processing: "100% Local Processing",
    local_blurb: "Audio is transcribed by a server on this machine and deleted straight after. Nothing leaves your system.",
    stat_engine: "Engine",
    stat_whisper: "Whisper",
    stat_last_run: "Last run",
    private_sandbox: "Private Sandbox:",
    agent_name: "VoxPrompt Agent",
    agent_blurb: "Record or upload a rough spoken idea. I transcribe it locally, sort it into prompt fields you can correct, and assemble a production-ready prompt. Fully offline.",
    accuracy_settings: "Accuracy Settings",
    transcription_model: "Transcription model",
    target_model: "Target model",
    target_model_help: "The local LLM that extracts variables. Set with VPB_OLLAMA_MODEL when starting the server.",
    names_jargon: "Names & jargon",
    vocab_hint: "Custom technical keywords or spelling targets to bias the Whisper engine. Comma-separated terms, not sentences.",
    vocab_prose: "This looks like prose. Use only the terms Whisper mishears \u2014 sentences here can bias the transcript.",
    vocab_long: "Too long \u2014 only the first {max} characters are used ({count} entered). Keep it to names and jargon.",
    session_cache: "Session history",
    clear_cache: "Clear session history",
    upload_audio: "Upload audio file",
    process_audio: "Process audio",
    cancel: "Cancel",
    cancelling: "Cancelling\u2026",
    step_transcript: "Refined Transcript",
    step_variables: "Extracted System Variables",
    step_prompt: "System Prompt Blueprint",
    transcript_hint: "Editable \u2014 fix anything misheard before the next step.",
    variables_hint: "Parsed from the transcript by the local model. Every field is editable \u2014 correct what it got wrong, fill what it missed. Blanks are simply left out.",
    prompt_hint: "Assembled by a fixed template, not by a model \u2014 the same fields always produce the same prompt.",
    raw_mapping: "Raw variable mapping (JSON)",
    copy_transcript: "Copy transcript",
    extract_variables: "Extract variables",
    copy_json: "Copy JSON",
    build_prompt: "Build prompt",
    rebuild_prompt: "Rebuild prompt",
    refine_variables: "Refine variables",
    copy_prompt: "Copy final prompt",
    copied: "Copied",
    recorder_idle: "Local recorder idle",
    recorder_active: "Local recorder active",
    recorder_ready: "Ready to process",
    recorder_hint_idle: "Click record to start local input streaming.",
    recorder_hint_active: "Recording \u2014 click the button again to stop.",
    recorder_hint_ready: "Press Process audio to transcribe locally.",
    recorder_left: "Recording \u2014 {seconds}s left",
    upload_formats: "mp3, wav, m4a, webm \u00b7 up to {minutes} min",
    workflow_running: "RUNNING",
    workflow_done: "COMPLETED",
    workflow_failed: "FAILED",
    workflow_cancelled: "CANCELLED",
    note_transcribing: "Whisper is processing the audio locally\u2026",
    note_extracting: "Extracting structured variables with the local model\u2026",
    note_restored: "Restored from the session history.",
    note_transcribed: "Transcribed {duration}s of audio in {elapsed}s with {model}.",
    note_structured: "Structured into {count} variables.",
    note_cancelled: "Cancelled.",
    status_transcribing: "Transcribing with {model}\u2026",
    status_extracting: "Extracting variables\u2026",
    status_queued: "Waiting for a free slot\u2026",
    hint_first_load: "First use of {model}: it is being loaded, which takes a minute or two, once.",
    hint_estimate: "Roughly {seconds}s.",
    words_chars: "{words} words \u00b7 {chars} characters",
    fields_all: "All {total} fields filled",
    fields_some: "{filled} of {total} filled \u00b7 blanks are fine, they are simply left out",
    prompt_stats: "~{tokens} tokens \u00b7 {chars} characters \u00b7 {sections} sections",
    out_of_date: "out of date \u2014 rebuild",
    cache_summary: "{count} sessions \u00b7 {size}",
    sandbox_secure: "Sandbox secure",
    sandbox_loading: "Loading {model}\u2026 everything runs on this machine.",
    sandbox_ready: "Whisper ({whisper}) and {llm}, running entirely offline.",
    backend_unreachable: "Backend unreachable",
    confirm_reextract: "Re-extracting replaces all variables and discards your edits. Continue?",
    confirm_clear: "Delete every saved session? This cannot be undone.",
    confirm_delete_session: "Delete this session?",
    err_no_transcript: "There is no transcript to analyse yet.",
    err_generic: "Something went wrong ({status}).",
    err_network: "Could not reach the backend: {message}. Is the server still running?",
    err_copy: "Could not copy automatically. Select the text and press Cmd+C.",
    err_too_long: "That file is {minutes} minutes long, over the {limit} minute limit.",
    err_ffmpeg_missing: "ffmpeg is not installed. Transcription will fail until you run: brew install ffmpeg",
    err_ollama_unavailable: "Ollama is not running, so variable extraction will fail. Start it with: ollama serve",
    err_ollama_model_missing: "The model {model} is not pulled. Run: ollama pull {model}",
    err_audio_too_long: "That recording is too long. Record or upload something shorter.",
    err_transcript_too_long: "The transcript is too long for the model to read in one pass. Shorten the recording.",
    err_empty_audio: "No speech was detected. Check your microphone level and try again.",
    err_conversion_failed: "That audio could not be decoded. Try a different file.",
    err_empty_prompt: "There is nothing to build a prompt from yet. Fill in at least one field.",
    err_mic_denied: "Microphone access was denied. Allow it in your browser settings, then reload.",
    err_mic_failed: "Could not open the microphone ({name}). Is another app using it?",
    err_no_recorder: "This browser cannot record audio. Upload an audio file instead.",
    err_empty_recording: "The recording came out empty. Check your microphone and try again.",
    err_too_many_jobs: "Too much is running at once. Wait for the current step to finish, or cancel it.",
    answer_aloud: "Answer this out loud",
    answer_stop: "Stop and transcribe",
    answer_recording: "Listening\u2014 {seconds}s left",
    answer_transcribing: "Transcribing your answer\u2026",
    answer_empty: "Nothing was heard. Try again, closer to the microphone.",
  },
  de: {
    tagline: "Agentisches Prompt-Engineering",
    new_session: "Neue Sitzung",
    recent_sessions: "Letzte Sitzungen",
    no_sessions: "Noch keine Sitzungen. Nimm etwas auf, dann erscheint es hier.",
    local_processing: "100% lokale Verarbeitung",
    local_blurb: "Audio wird von einem Server auf diesem Rechner transkribiert und danach sofort gel\u00f6scht. Nichts verl\u00e4sst dein System.",
    stat_engine: "Engine",
    stat_whisper: "Whisper",
    stat_last_run: "Letzter Lauf",
    private_sandbox: "Private Sandbox:",
    agent_name: "VoxPrompt Agent",
    agent_blurb: "Nimm eine grobe gesprochene Idee auf oder lade sie hoch. Ich transkribiere sie lokal, sortiere sie in Prompt-Felder, die du korrigieren kannst, und baue daraus einen einsatzfertigen Prompt. Vollst\u00e4ndig offline.",
    accuracy_settings: "Genauigkeits-Einstellungen",
    transcription_model: "Transkriptionsmodell",
    target_model: "Zielmodell",
    target_model_help: "Das lokale LLM, das die Variablen extrahiert. Beim Serverstart \u00fcber VPB_OLLAMA_MODEL setzen.",
    names_jargon: "Namen & Fachbegriffe",
    vocab_hint: "Eigene Fachbegriffe oder Schreibweisen, die Whisper beeinflussen sollen. Kommagetrennte Begriffe, keine S\u00e4tze.",
    vocab_prose: "Das sieht nach Flie\u00dftext aus. Nutze nur die Begriffe, die Whisper falsch versteht \u2014 S\u00e4tze verzerren das Transkript.",
    vocab_long: "Zu lang \u2014 nur die ersten {max} Zeichen werden genutzt ({count} eingegeben). Beschr\u00e4nke dich auf Namen und Fachbegriffe.",
    session_cache: "Sitzungsverlauf",
    clear_cache: "Sitzungsverlauf l\u00f6schen",
    upload_audio: "Audiodatei hochladen",
    process_audio: "Audio verarbeiten",
    cancel: "Abbrechen",
    cancelling: "Wird abgebrochen\u2026",
    step_transcript: "Bereinigtes Transkript",
    step_variables: "Extrahierte Systemvariablen",
    step_prompt: "System-Prompt-Blueprint",
    transcript_hint: "Bearbeitbar \u2014 korrigiere Missverstandenes vor dem n\u00e4chsten Schritt.",
    variables_hint: "Vom lokalen Modell aus dem Transkript gelesen. Jedes Feld ist bearbeitbar \u2014 korrigiere Fehler, erg\u00e4nze Fehlendes. Leere Felder werden einfach weggelassen.",
    prompt_hint: "Aus einer festen Vorlage zusammengesetzt, nicht von einem Modell \u2014 gleiche Felder ergeben immer denselben Prompt.",
    raw_mapping: "Rohe Variablenzuordnung (JSON)",
    copy_transcript: "Transkript kopieren",
    extract_variables: "Variablen extrahieren",
    copy_json: "JSON kopieren",
    build_prompt: "Prompt bauen",
    rebuild_prompt: "Prompt neu bauen",
    refine_variables: "Variablen anpassen",
    copy_prompt: "Prompt kopieren",
    copied: "Kopiert",
    recorder_idle: "Lokale Aufnahme bereit",
    recorder_active: "Aufnahme l\u00e4uft",
    recorder_ready: "Bereit zur Verarbeitung",
    recorder_hint_idle: "Klicke auf Aufnahme, um lokal aufzunehmen.",
    recorder_hint_active: "Aufnahme l\u00e4uft \u2014 klicke erneut zum Stoppen.",
    recorder_hint_ready: "Klicke auf Audio verarbeiten, um lokal zu transkribieren.",
    recorder_left: "Aufnahme \u2014 noch {seconds}s",
    upload_formats: "mp3, wav, m4a, webm \u00b7 bis {minutes} Min.",
    workflow_running: "L\u00c4UFT",
    workflow_done: "FERTIG",
    workflow_failed: "FEHLER",
    workflow_cancelled: "ABGEBROCHEN",
    note_transcribing: "Whisper verarbeitet das Audio lokal\u2026",
    note_extracting: "Das lokale Modell extrahiert strukturierte Variablen\u2026",
    note_restored: "Aus dem Sitzungsverlauf geladen.",
    note_transcribed: "{duration}s Audio in {elapsed}s mit {model} transkribiert.",
    note_structured: "In {count} Variablen strukturiert.",
    note_cancelled: "Abgebrochen.",
    status_transcribing: "Transkribiere mit {model}\u2026",
    status_extracting: "Extrahiere Variablen\u2026",
    status_queued: "Warte auf einen freien Platz\u2026",
    hint_first_load: "Erste Nutzung von {model}: es wird geladen, das dauert einmalig ein bis zwei Minuten.",
    hint_estimate: "Etwa {seconds}s.",
    words_chars: "{words} W\u00f6rter \u00b7 {chars} Zeichen",
    fields_all: "Alle {total} Felder ausgef\u00fcllt",
    fields_some: "{filled} von {total} ausgef\u00fcllt \u00b7 leere Felder sind in Ordnung, sie entfallen einfach",
    prompt_stats: "~{tokens} Tokens \u00b7 {chars} Zeichen \u00b7 {sections} Abschnitte",
    out_of_date: "veraltet \u2014 neu bauen",
    cache_summary: "{count} Sitzungen \u00b7 {size}",
    sandbox_secure: "Sandbox sicher",
    sandbox_loading: "Lade {model}\u2026 alles l\u00e4uft auf diesem Rechner.",
    sandbox_ready: "Whisper ({whisper}) und {llm}, vollst\u00e4ndig offline.",
    backend_unreachable: "Backend nicht erreichbar",
    confirm_reextract: "Beim erneuten Extrahieren werden alle Variablen ersetzt und deine \u00c4nderungen verworfen. Fortfahren?",
    confirm_clear: "Alle gespeicherten Sitzungen l\u00f6schen? Das l\u00e4sst sich nicht r\u00fcckg\u00e4ngig machen.",
    confirm_delete_session: "Diese Sitzung l\u00f6schen?",
    err_no_transcript: "Es gibt noch kein Transkript zum Analysieren.",
    err_generic: "Etwas ist schiefgelaufen ({status}).",
    err_network: "Backend nicht erreichbar: {message}. L\u00e4uft der Server noch?",
    err_copy: "Kopieren nicht m\u00f6glich. Markiere den Text und dr\u00fccke Cmd+C.",
    err_too_long: "Diese Datei ist {minutes} Minuten lang, mehr als das Limit von {limit} Minuten.",
    err_ffmpeg_missing: "ffmpeg ist nicht installiert. Die Transkription schl\u00e4gt fehl, bis du ausf\u00fchrst: brew install ffmpeg",
    err_ollama_unavailable: "Ollama l\u00e4uft nicht, die Variablenextraktion schl\u00e4gt fehl. Starte es mit: ollama serve",
    err_ollama_model_missing: "Das Modell {model} ist nicht geladen. F\u00fchre aus: ollama pull {model}",
    err_audio_too_long: "Diese Aufnahme ist zu lang. Nimm etwas K\u00fcrzeres auf oder lade eine k\u00fcrzere Datei hoch.",
    err_transcript_too_long: "Das Transkript ist zu lang, um es in einem Durchgang zu lesen. K\u00fcrze die Aufnahme.",
    err_empty_audio: "Es wurde keine Sprache erkannt. Pr\u00fcfe den Mikrofonpegel und versuche es erneut.",
    err_conversion_failed: "Dieses Audio konnte nicht dekodiert werden. Probiere eine andere Datei.",
    err_empty_prompt: "Es gibt noch nichts, woraus ein Prompt gebaut werden k\u00f6nnte. F\u00fclle mindestens ein Feld aus.",
    err_mic_denied: "Mikrofonzugriff wurde verweigert. Erlaube ihn in den Browser-Einstellungen und lade neu.",
    err_mic_failed: "Mikrofon konnte nicht ge\u00f6ffnet werden ({name}). Nutzt eine andere App es gerade?",
    err_no_recorder: "Dieser Browser kann kein Audio aufnehmen. Lade stattdessen eine Datei hoch.",
    err_empty_recording: "Die Aufnahme war leer. Pr\u00fcfe dein Mikrofon und versuche es erneut.",
    err_too_many_jobs: "Es l\u00e4uft schon zu viel gleichzeitig. Warte, bis der aktuelle Schritt fertig ist, oder brich ihn ab.",
    answer_aloud: "Diese Frage laut beantworten",
    answer_stop: "Stoppen und transkribieren",
    answer_recording: "H\u00f6re zu \u2014 noch {seconds}s",
    answer_transcribing: "Antwort wird transkribiert\u2026",
    answer_empty: "Nichts geh\u00f6rt. Versuch es noch einmal, n\u00e4her am Mikrofon.",
  },
};

const LANG_KEY = "voxprompt.language";
let lang = readLanguage();

function readLanguage() {
  try {
    const stored = localStorage.getItem(LANG_KEY);
    if (stored && STRINGS[stored]) return stored;
  } catch { /* Storage may be unavailable; fall through. */ }
  return navigator.language?.toLowerCase().startsWith("de") ? "de" : "en";
}

/** Look up a string and fill its {placeholders}. */
function t(key, vars = {}) {
  const table = STRINGS[lang] || STRINGS.en;
  const template = table[key] ?? STRINGS.en[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (_m, name) =>
    Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : `{${name}}`);
}

/** Translate a backend error by its code, falling back to the server's text. */
function errorText(payload, status) {
  const code = payload?.error;
  const key = code ? `err_${code}` : null;
  if (key && (STRINGS[lang]?.[key] || STRINGS.en[key])) {
    return t(key, { model: lastHealth?.ollama_model ?? "" });
  }
  return payload?.detail || t("err_generic", { status });
}

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
  langEn: el("langEn"), langDe: el("langDe"),
  cancelBtn: el("cancelBtn"), progressTrack: el("progressTrack"), progressFill: el("progressFill"),
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
let lastHealth = null;
let activeJob = null;            // { id, kind } while work is in flight
let current = newSession();

// ---------------------------------------------------------------------------
// Session cache (localStorage)
// ---------------------------------------------------------------------------

function newSession() {
  return {
    id: null,                    // Assigned by the server on first save.
    title: "", language: lang,
    audio_name: "", audio_seconds: 0,
    transcript: "", extraction: null, prompt: "",
    meta: {},
  };
}

/**
 * Persist the session.
 *
 * History lives in SQLite on the server rather than in the browser, so it
 * survives a cleared cache and is visible from any browser on this machine.
 * It is still entirely local: the server writes one file and talks to nobody.
 */
async function persistCurrent() {
  if (!current.transcript) return;   // Nothing worth remembering yet.

  current.title = deriveTitle(current);
  current.language = lang;

  try {
    const response = await fetch("/sessions", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(current),
    });
    if (!response.ok) return;

    const saved = await response.json();
    current.id = saved.id;          // Subsequent saves update this row.
    await refreshSessions();
  } catch {
    /* History is a convenience; a save failure must not break the workflow. */
  }
}

/** A readable name: the extracted goal if there is one, else the first words. */
function deriveTitle(session) {
  const goal = session.extraction?.goal;
  const source = (goal && String(goal).trim()) || session.transcript;
  if (!source) return "Untitled session";

  const words = source.trim().split(/\s+/).slice(0, 6).join(" ");
  return words.length > 46 ? `${words.slice(0, 46)}\u2026` : words;
}

async function refreshSessions() {
  let sessions = [];
  try {
    const response = await fetch("/sessions");
    if (response.ok) sessions = await response.json();
  } catch {
    /* Listed as empty rather than blocking the page. */
  }

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
    title.textContent = session.title || "Untitled session";
    button.append(title);
    button.addEventListener("click", () => restoreSession(session.id));

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "session-delete";
    remove.textContent = "\u00d7";
    remove.title = t("confirm_delete_session");
    remove.addEventListener("click", async (event) => {
      event.stopPropagation();   // Do not also open the session.
      if (!window.confirm(t("confirm_delete_session"))) return;
      await fetch(`/sessions/${session.id}`, { method: "DELETE" });
      if (session.id === current.id) startNewSession();
      else { await refreshSessions(); await refreshCacheSize(); }
    });

    item.append(button, remove);
    ui.sessionList.append(item);
  }
}

async function refreshCacheSize() {
  try {
    const stats = await (await fetch("/sessions/stats")).json();
    const size = stats.bytes > 1048576
      ? `${(stats.bytes / 1048576).toFixed(1)} MB`
      : `${Math.ceil(stats.bytes / 1024)} KB`;
    ui.cacheSize.textContent = t("cache_summary", { count: stats.sessions, size });
  } catch {
    ui.cacheSize.textContent = "\u2014";
  }
}

/** Reopen a stored session, restoring every stage that had been reached. */
async function restoreSession(id) {
  let session;
  try {
    const response = await fetch(`/sessions/${id}`);
    if (!response.ok) return;
    session = await response.json();
  } catch {
    return;
  }

  current = {
    id: session.id, title: session.title, language: session.language,
    audio_name: session.audio_name, audio_seconds: session.audio_seconds,
    transcript: session.transcript, extraction: session.extraction,
    prompt: session.prompt, meta: session.meta || {},
  };

  clearBanner();
  hideStatus();
  pending = null;
  ui.processBtn.disabled = true;
  ui.audioCard.hidden = true;

  ui.workflow.hidden = false;
  setWorkflow("done", t("note_restored"));

  ui.transcript.value = session.transcript || "";
  ui.resultMeta.textContent = `${(session.transcript || "").length} chars`;
  ui.resultCard.hidden = !session.transcript;
  updateCharCount();

  if (session.extraction) {
    ui.analyzeCard.hidden = false;
    renderFields(session.extraction, []);
    ui.analyzeMeta.textContent = session.meta?.analyze || "";
    ui.analyzeResult.hidden = false;
  } else {
    ui.analyzeCard.hidden = true;
    ui.analyzeResult.hidden = true;
  }

  if (session.prompt) {
    ui.buildCard.hidden = false;
    ui.promptOutput.textContent = session.prompt;
    ui.promptStats.textContent = session.meta?.prompt_stats || "";
    ui.buildMeta.textContent = session.meta?.sections || "";
    ui.buildMeta.classList.remove("stale");
    ui.buildResult.hidden = false;
  } else {
    ui.buildCard.hidden = true;
    ui.buildResult.hidden = true;
  }

  await refreshSessions();
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
  ui.timer.textContent = "00:00";
  ui.recordLabel.textContent = t("recorder_idle");

  refreshSessions();
  refreshCacheSize();
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
  const keys = {
    run: "workflow_running", done: "workflow_done",
    fail: "workflow_failed", cancelled: "workflow_cancelled",
  };
  ui.workflowPill.textContent = t(keys[state] || "workflow_running");
  ui.workflowPill.className =
    `pill ${state === "done" ? "done" : state === "run" ? "" : "fail"}`;
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
      showBanner(t("recorder_left", { seconds: 0 }), "warn");
      stopRecording();
    } else if (elapsed >= WARN_SECONDS) {
      ui.recordLabel.textContent = t("recorder_left", { seconds: Math.ceil(MAX_SECONDS - elapsed) });
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
    showBanner(t("err_no_recorder"));
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    // The distinction matters: one is fixable in settings, one is not.
    const denied = err.name === "NotAllowedError" || err.name === "SecurityError";
    showBanner(denied ? t("err_mic_denied") : t("err_mic_failed", { name: err.name }));
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
      showBanner(t("err_empty_recording"));
      return;
    }
    stageAudio(blob, `recording.${extensionFor(type)}`, seconds, levels);
  });

  rec.recorder.start(250);   // Flush chunks regularly so nothing is lost.
  startTimer();
  startMeter(stream);

  ui.recordBtn.classList.add("recording");
  ui.recordBtn.setAttribute("aria-label", "Stop recording");
  ui.recordLabel.textContent = t("recorder_active");
  ui.recorderHint.textContent = t("recorder_hint_active");
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
  ui.recordLabel.textContent = t("recorder_idle");
  ui.recorderHint.textContent = t("recorder_hint_idle");
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
  ui.recordLabel.textContent = t("recorder_ready");
  ui.recorderHint.textContent = t("recorder_hint_ready");
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

function setProgress(fraction) {
  if (fraction === null || fraction === undefined) {
    // Unknowable rather than zero: an invented bar is worse than none.
    ui.progressTrack.hidden = true;
    return;
  }
  ui.progressTrack.hidden = false;
  ui.progressFill.style.width = `${Math.round(fraction * 100)}%`;
}

/**
 * Poll a job until it settles.
 *
 * Polling rather than a socket: the work takes tens of seconds, so a request
 * per second is negligible, and it survives a dropped connection without any
 * reconnect logic.
 *
 * @returns the job's result, or null if it failed or was cancelled.
 */
async function awaitJob(jobId, onUpdate) {
  activeJob = jobId;
  ui.cancelBtn.hidden = false;
  ui.cancelBtn.disabled = false;
  ui.cancelBtn.textContent = t("cancel");

  try {
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, 700));

      let job;
      try {
        const response = await fetch(`/jobs/${jobId}`);
        if (response.status === 404) {
          showBanner(t("err_generic", { status: 404 }));
          return null;
        }
        job = await response.json();
      } catch (err) {
        showBanner(t("err_network", { message: err.message }));
        return null;
      }

      onUpdate?.(job);
      setProgress(job.progress);

      if (job.state === "done") {
        // The result is in hand, so the server need not keep the transcript
        // in memory for the rest of its retention window.
        releaseJob(jobId);
        return job.result;
      }
      if (job.state === "cancelled") {
        releaseJob(jobId);
        setWorkflow("cancelled", t("note_cancelled"));
        return null;
      }
      if (job.state === "error") {
        const failure = { error: job.error_code, detail: job.error };
        releaseJob(jobId);
        setWorkflow("fail", job.error || "");
        showBanner(errorText(failure, 500));
        return null;
      }
    }
  } finally {
    activeJob = null;
    ui.cancelBtn.hidden = true;
    setProgress(null);
  }
}

/** Tell the server to forget a finished job. Failure here is not important. */
function releaseJob(jobId) {
  fetch(`/jobs/${jobId}`, { method: "DELETE" }).catch(() => {});
}

async function cancelActiveJob() {
  if (!activeJob) return;
  ui.cancelBtn.disabled = true;
  ui.cancelBtn.textContent = t("cancelling");
  try {
    await fetch(`/jobs/${activeJob}/cancel`, { method: "POST" });
  } catch {
    /* The poll loop will report whatever state it settles into. */
  }
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
  setWorkflow("run", t("note_transcribing"));

  const chosen = ui.modelSelect.value || "base";
  const rtf = MODEL_SPEED[chosen]?.rtf ?? 0.1;
  const estimate = Math.max(2, Math.round((pending.seconds || 30) * rtf));
  setStatus(
    t("status_transcribing", { model: chosen }),
    warmModels.has(chosen)
      ? t("hint_estimate", { seconds: estimate })
      : t("hint_first_load", { model: chosen })
  );

  const form = new FormData();
  form.append("audio", pending.blob, pending.filename);
  form.append("model", chosen);
  if (ui.vocabulary.value.trim()) form.append("vocabulary", ui.vocabulary.value.trim());

  try {
    const response = await fetch("/jobs/transcribe", { method: "POST", body: form });
    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      setWorkflow("fail", "");
      showBanner(errorText(payload, response.status));
      return;
    }

    const result = await awaitJob(payload.job_id, (job) => {
      if (job.note) ui.statusHint.textContent = job.note;
    });
    if (result) showTranscript(result);
  } catch (err) {
    setWorkflow("fail", "");
    showBanner(t("err_network", { message: err.message }));
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

  setWorkflow("done", t("note_transcribed", {
    duration: payload.duration.toFixed(1),
    elapsed: payload.elapsed_s,
    model: payload.model,
  }));
  ui.statLatency.textContent = `${payload.elapsed_s}s`;

  current.transcript = payload.text;
  current.audio_name = pending?.filename || "";
  current.audio_seconds = payload.duration;
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
  ui.charCount.textContent = t("words_chars", { words, chars: text.length });
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

const MIC_ICON =
  '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round"><path d="M12 2a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V5a3 3 0 0 1 3-3z"/>' +
  '<path d="M19 10v1a7 7 0 0 1-14 0v-1M12 18v4"/></svg>';

const STOP_ICON =
  '<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor" ' +
  'stroke="none"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';

const SPIN_ICON =
  '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round"><path d="M12 3a9 9 0 1 0 9 9"/></svg>';

/**
 * State of the one field recording that may be in flight.
 *
 * Only one at a time, and never alongside the main recorder: they would
 * compete for the microphone and for the transcription slot.
 */
const answer = { field: null, recorder: null, stream: null, timer: null, startedAt: 0 };

/** The mic button that records a spoken answer for one field. */
function buildFieldMic(name, question) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "field-mic";
  button.dataset.mic = name;
  button.innerHTML = MIC_ICON;
  button.title = t("answer_aloud");
  button.setAttribute("aria-label", t("answer_aloud"));

  button.addEventListener("click", () => {
    if (answer.field === name) stopFieldAnswer();
    else startFieldAnswer(name, question);
  });
  return button;
}

/** Disable every other mic while one is busy, so they cannot overlap. */
function setMicsEnabled(enabled, except = null) {
  for (const button of ui.fieldList.querySelectorAll(".field-mic")) {
    button.disabled = !enabled && button.dataset.mic !== except;
  }
  ui.recordBtn.disabled = !enabled;
}

async function startFieldAnswer(name, question) {
  if (answer.field || (rec.recorder && rec.recorder.state === "recording")) return;
  clearBanner();

  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    showBanner(t("err_no_recorder"));
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    const denied = err.name === "NotAllowedError" || err.name === "SecurityError";
    showBanner(denied ? t("err_mic_denied") : t("err_mic_failed", { name: err.name }));
    return;
  }

  const mimeType = pickMimeType();
  let recorder;
  try {
    recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  } catch (err) {
    stream.getTracks().forEach((track) => track.stop());
    showBanner(`${err.message}`);
    return;
  }

  const chunks = [];
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  });
  recorder.addEventListener("stop", () => {
    const type = recorder.mimeType || mimeType || "audio/webm";
    stream.getTracks().forEach((track) => track.stop());
    finishFieldAnswer(name, question, new Blob(chunks, { type }), type);
  });

  answer.field = name;
  answer.recorder = recorder;
  answer.stream = stream;
  answer.startedAt = Date.now();
  recorder.start(250);

  const button = ui.fieldList.querySelector(`[data-mic="${name}"]`);
  button.classList.add("recording");
  button.innerHTML = STOP_ICON;
  button.title = t("answer_stop");
  setMicsEnabled(false, name);
  showAnswerStatus(name, t("answer_recording", { seconds: MAX_ANSWER_SECONDS }));

  answer.timer = setInterval(() => {
    const left = Math.ceil(MAX_ANSWER_SECONDS - (Date.now() - answer.startedAt) / 1000);
    if (left <= 0) stopFieldAnswer();
    else showAnswerStatus(name, t("answer_recording", { seconds: left }));
  }, 250);
}

function stopFieldAnswer() {
  if (answer.recorder && answer.recorder.state !== "inactive") answer.recorder.stop();
}

/** A small line under the field, so the state is visible where you are looking. */
function showAnswerStatus(name, text) {
  const row = ui.fieldList.querySelector(`[data-row="${name}"]`);
  if (!row) return;

  let status = row.querySelector(".field-answer");
  if (!status) {
    status = document.createElement("div");
    status.className = "field-answer";
    status.style.gridColumn = "2";
    row.append(status);
  }
  status.hidden = false;
  status.textContent = text;
}

function clearAnswerStatus(name) {
  const status = ui.fieldList.querySelector(`[data-row="${name}"] .field-answer`);
  if (status) status.hidden = true;
}

/** Transcribe the spoken answer and put it in the field. */
async function finishFieldAnswer(name, question, blob, type) {
  if (answer.timer) clearInterval(answer.timer);
  Object.assign(answer, { field: null, recorder: null, stream: null, timer: null });

  const button = ui.fieldList.querySelector(`[data-mic="${name}"]`);
  button.classList.remove("recording");
  button.classList.add("busy");
  button.innerHTML = SPIN_ICON;
  button.title = t("answer_transcribing");
  showAnswerStatus(name, t("answer_transcribing"));

  try {
    if (blob.size === 0) {
      showBanner(t("err_empty_recording"), "warn");
      return;
    }

    const form = new FormData();
    form.append("audio", blob, `answer.${extensionFor(type)}`);
    form.append("model", ui.modelSelect.value || "base");
    if (ui.vocabulary.value.trim()) form.append("vocabulary", ui.vocabulary.value.trim());
    // Sent as decoding context. Measured to make little difference by itself;
    // the vocabulary box above is what actually rescues a short answer.
    if (question) form.append("context", question);

    const response = await fetch("/jobs/transcribe", { method: "POST", body: form });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      showBanner(errorText(payload, response.status));
      return;
    }

    const result = await awaitJob(payload.job_id);
    if (!result) return;

    const text = (result.text || "").trim();
    if (!text) {
      showBanner(t("answer_empty"), "warn");
      return;
    }
    applyAnswer(name, text);
  } catch (err) {
    showBanner(t("err_network", { message: err.message }));
  } finally {
    button.classList.remove("busy");
    button.innerHTML = MIC_ICON;
    button.title = t("answer_aloud");
    setMicsEnabled(true);
    clearAnswerStatus(name);
  }
}

/**
 * Split a spoken list answer into items.
 *
 * People say list items as sentences -- "Fast. Cheap. Local." -- rather than
 * as separate lines, so sentence boundaries are the natural split. Trailing
 * sentence punctuation is dropped: "It must be fast!" reads oddly as a
 * constraint, and the list is a set of phrases, not prose.
 */
function splitSpokenList(text) {
  return text
    .split(/(?<=[.!?])\s+|\n+/)
    .map((part) => part.replace(/[.!?\s]+$/, "").trim())
    .filter(Boolean);
}

/**
 * Put a spoken answer into its field.
 *
 * Appended rather than replacing, so a second answer adds to the first and an
 * existing typed value is never silently destroyed. List fields get one item
 * per sentence, which is how people say them out loud.
 */
function applyAnswer(name, text) {
  const input = ui.fieldList.querySelector(`[data-field="${name}"]`);
  if (!input) return;

  const existing = input.value.trim();
  if (FIELDS[name].list) {
    input.value = [existing, ...splitSpokenList(text)].filter(Boolean).join("\n");
  } else {
    input.value = existing ? `${existing} ${text}` : text;
  }

  autoGrow(input);
  refreshAnalysisState();
  input.focus();
  input.setSelectionRange(input.value.length, input.value.length);
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
    filled === total ? t("fields_all", { total }) : t("fields_some", { filled, total });

  ui.rawJson.textContent = JSON.stringify(extraction, null, 2);

  // A prompt built before this edit no longer matches the fields above.
  if (!ui.buildResult.hidden) {
    ui.buildMeta.textContent = t("out_of_date");
    ui.buildMeta.classList.add("stale");
  }

  current.extraction = extraction;
  return extraction;
}

/**
 * Questions seen so far, by field.
 *
 * The server only sends a question for a field it found empty, but a filled
 * field can still be answered aloud, and the question is what primes Whisper.
 */
const FIELD_QUESTIONS = {};

function renderFields(extraction, missing) {
  const questionFor = new Map((missing || []).map((m) => [m.field, m.question]));
  for (const [field, question] of questionFor) FIELD_QUESTIONS[field] = question;

  ui.fieldList.innerHTML = "";
  for (const [name, spec] of Object.entries(FIELDS)) {
    const row = document.createElement("li");
    row.dataset.row = name;

    const label = document.createElement("label");
    label.className = "field-name";
    label.textContent = spec.label;
    label.htmlFor = `field-${name}`;

    const question = questionFor.get(name) || FIELD_QUESTIONS[name] || "";
    row.append(
      label,
      buildFieldInput(name, extraction[name], questionFor.get(name)),
      buildFieldMic(name, question),
    );
    ui.fieldList.append(row);
  }

  refreshAnalysisState();
  ui.fieldList.querySelectorAll(".field-input").forEach(autoGrow);
}

async function analyzeTranscript() {
  const transcript = ui.transcript.value.trim();
  if (!transcript) {
    showBanner(t("err_no_transcript"), "warn");
    return;
  }

  // Re-analysing replaces every field, so do not silently bin typed answers.
  if (!ui.analyzeResult.hidden && !window.confirm(t("confirm_reextract"))) return;

  clearBanner();
  ui.analyzeBtn.disabled = true;
  setWorkflow("run", t("note_extracting"));
  setStatus(t("status_extracting"), "");

  try {
    const response = await fetch("/jobs/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript, language: lang }),
    });
    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      setWorkflow("fail", "");
      showBanner(errorText(payload, response.status));
      return;
    }

    const result = await awaitJob(payload.job_id, (job) => {
      if (job.note) ui.statusHint.textContent = job.note;
    });
    if (!result) return;

    ui.analyzeCard.hidden = false;
    renderFields(result.extraction, result.missing);
    ui.analyzeMeta.textContent = `${result.elapsed_s}s \u00b7 ${result.model}`;
    ui.analyzeResult.hidden = false;
    setWorkflow("done", t("note_structured", { count: Object.keys(FIELDS).length }));

    current.transcript = transcript;
    current.meta = { ...current.meta, analyze: ui.analyzeMeta.textContent };
    persistCurrent();
  } catch (err) {
    setWorkflow("fail", "");
    showBanner(t("err_network", { message: err.message }));
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
      showBanner(errorText(payload, response.status));
      return;
    }

    ui.buildCard.hidden = false;
    ui.promptOutput.textContent = payload.prompt;
    ui.promptStats.textContent = t("prompt_stats", {
      tokens: payload.estimated_tokens,
      chars: payload.characters,
      sections: payload.sections.length,
    });
    ui.buildMeta.textContent = payload.sections.join(", ");
    ui.buildMeta.classList.remove("stale");
    ui.buildResult.hidden = false;
    ui.buildBtn.textContent = t("rebuild_prompt");

    current.prompt = payload.prompt;
    current.meta = {
      ...current.meta,
      prompt_stats: ui.promptStats.textContent,
      sections: ui.buildMeta.textContent,
    };
    persistCurrent();

    ui.buildCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    showBanner(t("err_network", { message: err.message }));
  } finally {
    ui.buildBtn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Clipboard
// ---------------------------------------------------------------------------

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    const original = button.textContent;
    button.textContent = t("copied");
    setTimeout(() => (button.textContent = original), 1400);
  } catch {
    // The clipboard API needs a secure context; say so rather than failing mutely.
    showBanner(t("err_copy"), "warn");
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
    showBanner(t("err_ffmpeg_missing"));
    return;
  }
  if (!health.ollama) {
    showBanner(t("err_ollama_unavailable"), "warn");
  } else if (!health.ollama_model_available) {
    showBanner(t("err_ollama_model_missing", { model: health.ollama_model }), "warn");
  }
}

function setSandboxStatus(health) {
  const problems = [];
  if (!health.ffmpeg) problems.push("ffmpeg missing");
  if (!health.ollama) problems.push("Ollama offline");
  else if (!health.ollama_model_available) problems.push("model not pulled");

  const ok = problems.length === 0;
  ui.statusDot.className = `dot-status ${ok ? "ok" : health.ffmpeg ? "warn" : "bad"}`;
  ui.statusLabel.textContent = ok ? t("sandbox_secure") : problems.join(", ");
  ui.sandboxLine.textContent = health.model_warming
    ? t("sandbox_loading", { model: health.whisper_model })
    : t("sandbox_ready", { whisper: health.whisper_model, llm: health.ollama_model });
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    lastHealth = health;

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
    ui.statusLabel.textContent = t("backend_unreachable");
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
    ui.vocabHint.textContent = t("vocab_long", { max: MAX_VOCABULARY_CHARS, count: value.length });
  } else if (looksLikeProse) {
    ui.vocabHint.textContent = t("vocab_prose");
  } else {
    ui.vocabHint.textContent = t("vocab_hint");
  }
}

// ---------------------------------------------------------------------------
// Language
// ---------------------------------------------------------------------------

/** Apply the current language to everything marked with data-i18n. */
function applyLanguage() {
  document.documentElement.lang = lang;

  for (const node of document.querySelectorAll("[data-i18n]")) {
    node.textContent = t(node.dataset.i18n);
  }

  ui.langEn.classList.toggle("active", lang === "en");
  ui.langDe.classList.toggle("active", lang === "de");

  // Text set from JavaScript is not covered by data-i18n, so refresh it here.
  ui.fileName.textContent = t("upload_formats", { minutes: MAX_UPLOAD_SECONDS / 60 });
  if (!rec.recorder || rec.recorder.state === "inactive") {
    ui.recordLabel.textContent = pending ? t("recorder_ready") : t("recorder_idle");
    ui.recorderHint.textContent = pending ? t("recorder_hint_ready") : t("recorder_hint_idle");
  }
  if (!ui.buildResult.hidden) ui.buildBtn.textContent = t("rebuild_prompt");
  if (!ui.analyzeResult.hidden) refreshAnalysisState();
  updateCharCount();
  updateModelHint();
  updateVocabHint();
  refreshCacheSize();
  if (lastHealth) setSandboxStatus(lastHealth);
}

function setLanguage(next) {
  if (!STRINGS[next] || next === lang) return;
  lang = next;
  try {
    localStorage.setItem(LANG_KEY, next);   // A UI preference, not history.
  } catch { /* Fine if storage is unavailable. */ }
  applyLanguage();
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

/**
 * Read an uploaded file's duration without decoding it.
 *
 * Lets an over-long file be rejected here rather than after an upload and a
 * wait. The server enforces the same limit regardless; this is only courtesy.
 */
function probeDuration(file) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const audio = new Audio();
    const done = (value) => { URL.revokeObjectURL(url); resolve(value); };

    audio.addEventListener("loadedmetadata", () =>
      done(Number.isFinite(audio.duration) ? audio.duration : null));
    audio.addEventListener("error", () => done(null));
    setTimeout(() => done(null), 5000);   // Some containers never report.
    audio.src = url;
  });
}

ui.fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  event.target.value = "";   // Allow re-picking the same file.
  if (!file) return;

  clearBanner();
  const seconds = await probeDuration(file);

  if (seconds !== null && seconds > MAX_UPLOAD_SECONDS) {
    showBanner(t("err_too_long", {
      minutes: Math.round(seconds / 60), limit: MAX_UPLOAD_SECONDS / 60,
    }), "warn");
    return;
  }

  ui.fileName.textContent = file.name;
  // Uploads carry no level data, so the waveform stays empty until transcribed.
  stageAudio(file, file.name, seconds ?? 0, []);
  ui.audioTime.textContent = seconds
    ? formatTime(seconds)
    : `${(file.size / 1_048_576).toFixed(1)} MB`;
});

ui.transcript.addEventListener("input", () => {
  updateCharCount();
  current.transcript = ui.transcript.value;
});

ui.refineBtn.addEventListener("click", () => {
  ui.analyzeCard.scrollIntoView({ behavior: "smooth", block: "start" });
  ui.fieldList.querySelector(".field-input")?.focus();
});

ui.copyBtn.addEventListener("click", () => copyText(ui.transcript.value, ui.copyBtn));
ui.copyJsonBtn.addEventListener("click", () =>
  copyText(JSON.stringify(readExtraction(), null, 2), ui.copyJsonBtn));
ui.copyPromptBtn.addEventListener("click", () =>
  copyText(ui.promptOutput.textContent, ui.copyPromptBtn));

ui.modelSelect.addEventListener("change", updateModelHint);
ui.vocabulary.addEventListener("input", updateVocabHint);
ui.newSessionBtn.addEventListener("click", startNewSession);

ui.clearCacheBtn.addEventListener("click", async () => {
  if (!window.confirm(t("confirm_clear"))) return;
  try {
    await fetch("/sessions", { method: "DELETE" });
  } catch {
    /* Reported by the refresh below if it did not take. */
  }
  startNewSession();
});

// Populate from the built-in list first so the controls work immediately,
// then let /health refine them.
ui.cancelBtn.addEventListener("click", cancelActiveJob);
ui.langEn.addEventListener("click", () => setLanguage("en"));
ui.langDe.addEventListener("click", () => setLanguage("de"));

applyLanguage();
populateModels();
refreshSessions();
refreshCacheSize();
checkHealth();

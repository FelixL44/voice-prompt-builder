# CLAUDE.md

Context for future sessions. Read this before changing anything.

## What this is

A local web app that turns a spoken brain-dump (up to ~5 minutes) into a
well-structured, prompt-engineered prompt for large, expensive LLMs.
**Everything runs on this machine. No audio or text leaves it.** The only
network call in the whole app is the one-time Whisper model download.

This is a portfolio project: code quality, a clear README, and a clean commit
history matter as much as the feature set.

## Hard constraints

These are not preferences. Breaking them breaks the build on this machine.

- **Intel Mac (x86_64), no GPU.** Every dependency must have an x86_64 macOS
  wheel.
- **No PyTorch, TensorFlow, or other heavy DL frameworks.** Many no longer ship
  Intel-Mac builds. `faster-whisper` is used precisely because it runs on
  CTranslate2 instead of torch.
- **`uv` resolves universally by default**, which picks packages that have no
  x86_64 macOS wheel (this already bit us: `onnxruntime` 1.30 is arm64-only on
  macOS). `pyproject.toml` therefore sets
  `tool.uv.required-environments = ["sys_platform == 'darwin' and platform_machine == 'x86_64'"]`.
  **Do not remove that line**, or `uv sync` will start failing.
- Local LLM via **Ollama, CPU only**. Keep default models small (3B-8B) and
  configurable.

## Stack

| Layer | Choice |
|---|---|
| Backend | Python 3.11 + FastAPI, deps via `uv` |
| Transcription | `faster-whisper` (CTranslate2), `int8` on CPU |
| Audio | `ffmpeg` converts WebM/Opus to 16 kHz mono WAV |
| Frontend | Plain HTML/CSS/JS, no build step |
| LLM | Ollama (from v0.2) |

Measured here (CPU-only Intel Mac), on jargon-heavy speech:

| Model | RTF | 5 min of audio |
|---|---|---|
| `base` | 0.10 | ~30s |
| `small` | 0.40 | ~2 min |

Models are cached in a dict keyed by `(size, device, compute_type)`, so the UI
can switch sizes without paying the load cost twice. First load of a size
downloads it (~40s for `base`, ~90s for `small`).

## Layout

```
backend/
  main.py        FastAPI app, routes, error->HTTP mapping
  transcribe.py  ffmpeg conversion + whisper
  schemas.py     Pydantic models (API contract)
  config.py      env-var settings, all VPB_* prefixed
  prompts/       system prompts as .md files (v0.2+)
  analyze.py     Ollama extraction + missing-field detection
  builder.py     deterministic prompt template
frontend/        index.html, app.js, style.css
tests/
```

## Commands

```bash
uv sync                              # install
uv run uvicorn backend.main:app --reload --port 8000
uv run pytest -q                     # full suite, ~10s
VPB_WHISPER_MODEL=tiny uv run pytest # faster iteration
```

Then open http://127.0.0.1:8000.

## Configuration

All settings live in `backend/config.py` and come from `VPB_*` env vars:
`VPB_WHISPER_MODEL` (default `base`), `VPB_COMPUTE_TYPE` (`int8`),
`VPB_LANGUAGE` (auto-detect if unset), `VPB_VAD_FILTER`, `VPB_BEAM_SIZE`,
`VPB_MAX_UPLOAD_BYTES`, `VPB_FFMPEG_PATH`, `VPB_TMP_DIR`, `VPB_OLLAMA_MODEL`
(default `llama3.2:3b`), `VPB_OLLAMA_URL`, `VPB_OLLAMA_TIMEOUT_S`,
`VPB_OLLAMA_NUM_CTX`.

`get_settings()` is `lru_cache`d, so tests that change the env must call
`get_settings.cache_clear()`.

## Conventions

- Functions stay small and fully type-annotated; `from __future__ import annotations`
  at the top of every module.
- Errors are typed exceptions (`TranscriptionError` and subclasses) carrying a
  `code`, mapped to HTTP status codes in one table in `main.py`. The UI shows
  `detail` verbatim, so those messages must be human-readable and actionable
  (e.g. "Install it with: brew install ffmpeg").
- **Temp audio is always deleted**, including on failure. `_scratch_dir()` is a
  context manager with `shutil.rmtree` in a `finally`. There is a test for this;
  keep it.
- Tests generate sample audio with macOS `say` instead of committing binaries,
  and skip cleanly when `say`/`ffmpeg` are absent.

## Design decisions worth keeping

- **The transcript is editable.** Whisper reliably mishears domain words
  (observed: "ramble" -> "rumble", "goal" -> "girl"), so the user fixes text
  before it reaches the LLM.
- **`initial_prompt` is the cheapest accuracy win available** and was the fix
  for v0.1's weak transcripts. Conditioning the decoder on likely vocabulary
  costs no extra time and repaired "pedantic" -> "Pydantic",
  "olama" -> "Ollama", "Cubernates" -> "Kubernetes". It also **stopped the
  decoder truncating**: on an uncertain tail `small` silently dropped the last
  clause, and grounding it in real terms brought the text back. If transcripts
  regress, check this path first. The user's terms are prepended (Whisper
  weights the start of the hint most) and truncated to `MAX_VOCABULARY_CHARS`,
  because the hint shares Whisper's 224-token window with the audio context.
- **`vad_filter` was *not* the cause of that truncation** -- it was tested both
  ways and made no difference. Do not go chasing VAD parameters for it.
- **Static assets are served with `Cache-Control: no-cache`**
  (`NoCacheStaticFiles` in `main.py`). Without it, browsers apply heuristic
  caching and keep running a stale `app.js` against freshly served HTML, which
  presents as a broken UI -- it cost a debugging round already. The ETag keeps
  revalidation cheap (304). There is a test asserting the header.
- **The model dropdown populates from a built-in list before `/health`
  answers**, so a failed or slow health check never leaves it empty and
  unusable. A test asserts the frontend list matches `ALLOWED_MODELS`.
- **The vocabulary field is for terms, not prose.** Sentences there bias the
  decoder towards continuing them; the UI warns past ~40 words and the backend
  truncates at `MAX_VOCABULARY_CHARS`.
- **Small models lie about absence, so absence is computed in code.**
  `find_missing()` decides what is empty; the model is never asked what it
  failed to find, because one that invented an answer will not report it.
- **`llama3.2:3b` emits the *string* `"null"`, not JSON `null`**, even under a
  schema that permits null. `NULL_STRINGS` + `normalize()` in `analyze.py`
  convert those to real absence before anything downstream sees them. Without
  it the question loop goes quiet and v0.4 would print "null" into the prompt.
  If extraction regresses, check this first.
- **The 3B model still over-fills `success_criteria` and `examples`** with
  restatements of the goal, despite the prompt saying not to. `qwen2.5:7b`
  follows the instruction better at roughly double the runtime. The prompt's
  anti-invention rules are load-bearing -- weakening them makes this worse.
- **The transcript is sent to `/analyze` as JSON, not as audio**, so the user's
  corrections in the transcript box are what gets analysed.
- **The transcript is wrapped in `<transcript>` tags** in the prompt, so a
  brain-dump containing instructions reads as data rather than as commands.
- **`is_empty` exists twice**: `analyze.py` judges the extraction on arrival,
  and `isEmptyValue` in `app.js` re-judges it live as the user types. That
  duplication is deliberate (no round-trip per keystroke) and guarded by
  `tests/test_frontend_parity.py`, which runs both implementations over the
  same cases via node and compares. Change one, change the other.
- **`tests/test_frontend_parity.py` also checks every `getElementById` in
  app.js resolves to an id in index.html.** A typo there yields a control that
  silently does nothing -- which is how the model dropdown once shipped empty.
- **The edited fields in the DOM are the source of truth for v0.4**, via
  `readExtraction()`. The server holds no session state.
- **`getUserMedia({ audio: true })` is intentional -- do not "fix" it.** That
  bare constraint lets the browser apply its voice-call DSP (confirmed live:
  `noiseSuppression`, `autoGainControl` and `echoCancellation` all true), which
  is tuned for phone calls rather than for ASR. Measured on this project:
  Opus bitrate makes no difference to accuracy at all (identical error from
  64k down to 12k), while background noise roughly doubles it. The laptop mic
  is far-field, so uploads from a phone transcribe noticeably better.
  The user considered disabling the DSP and **chose to leave it**, because in a
  genuinely noisy room the suppression may be helping, and nothing here proves
  otherwise. Revisit only with an A/B recording of the same words.
- **`builder.py` must never call a model.** The extraction is already
  structured; assembling it is string work. Determinism is the feature -- the
  same fields give byte-identical output, and there is a test asserting it.
- **Empty sections are omitted, not emitted blank.** An empty `<examples>` tag
  announces a section and then says nothing, which invites the model to invent
  content for it.
- **Section order is deliberate**: context first (long reference material),
  then task, then the qualifiers, with `output_format` and `success_criteria`
  last because formatting instructions hold best nearest generation.
- **`/transcribe` must keep its blocking work off the event loop.** It is
  `async def` (it needs `await audio.read()`), so ffmpeg and Whisper run via
  `run_in_threadpool`. Calling them directly froze every other request,
  including `/health`, for the whole transcription -- measured at 8.9s for a
  22s clip, so ~2 minutes for a 5-minute recording. `tests/test_concurrency.py`
  guards this with a stubbed 3s transcription; it stubs deliberately, because
  a real short clip finishes before the probe lands and an earlier version of
  that test passed with the bug present.
- **`get_model` uses double-checked locking.** Requests share a threadpool, so
  two could otherwise load the same model twice.
- **Concurrent transcriptions are capped at one** by default
  (`VPB_MAX_CONCURRENT_TRANSCRIPTIONS`). Measured: two at once finish only
  1.29x faster than two in sequence while making each ~50% slower, so queueing
  is the better trade on a CPU with no headroom. A single `WhisperModel` is
  safe across threads -- verified, output was byte-identical -- so this is a
  resource decision, not a safety one.
- **Uploads are size-checked before being buffered**: the declared
  Content-Length is refused outright, then the body is read in 1 MB chunks so a
  missing or dishonest header still cannot fill memory.
- **Warmup runs in the lifespan handler and is never awaited.** The server
  must accept requests immediately; `/health` reports `model_warming` so the UI
  can explain the wait. A failing warmup is logged and swallowed -- no network
  at startup must not stop the server, and the real error surfaces on the first
  request anyway. `tests/conftest.py` sets `VPB_WARMUP=false` before anything
  imports `backend.config`, or every TestClient would start a real download.
- **`/health` checks that the Ollama model is *pulled*, not just that Ollama
  answers.** `ollama_models()` returns `None` for unreachable and `[]` for
  running-but-empty; those are different states and the UI reports them
  differently. An untagged configured name matches `name:latest`, since that is
  how Ollama stores an untagged pull.
- **The UI is a three-pane console** (sessions / stream / settings) built to a
  Figma mockup. The product name in the interface is **VoxPrompt**; the repo and
  Python package stay `voice-prompt-builder`.
- **Session history is SQLite on the server** (`backend/store.py`, `data/`),
  not `localStorage`. Browser-only history is lost with the cache and invisible
  from any other browser. It stays local: stdlib `sqlite3`, one file, no
  network. A connection is opened per call, never shared, because requests run
  on a threadpool and SQLite connections are not thread-safe. Only the language
  preference still lives in `localStorage`.
- **Working files are swept at startup** (`sweep_scratch`), not only after each
  run. A `finally` cannot run if the process is killed, which would leave the
  user's audio on disk and break a promise the README makes. The sweep only
  touches `job-` and `upload-` prefixed entries, since `VPB_TMP_DIR` may point
  somewhere shared.
- **Uploads are staged to disk before the job starts**, not carried in the
  closure. A queued job can wait minutes for the transcription slot, and
  several waiting uploads held in memory would add up to however much audio
  was sent. `transcribe_file()` deletes the staged file in a `finally`.
- **Finished jobs are released by the client** once it has the result, because
  that result *is* the transcript. `VPB_JOB_RETENTION_S` (120s) is only the
  backstop for a client that never comes back -- do not raise it casually.
- **Active jobs are capped** (`VPB_MAX_ACTIVE_JOBS`, 4) and the cap is a 429.
  Without it every POST spawns a thread and stages an upload, unbounded.
- **SQLite runs in WAL mode.** Requests share a threadpool, so a reader and a
  writer overlap; the default rollback journal makes them collide and wait out
  the busy timeout.
- **Each operation exists twice at the HTTP layer and once underneath.**
  `/transcribe` and `/jobs/transcribe` (likewise `/analyze`) both call
  `_transcription_response` / `_analysis_response`; only the waiting differs.
  Keeping two copies had already let them drift: the synchronous upload path
  missed the chunked size-check that the job path received, so it buffered
  whole uploads before rejecting them. If you add behaviour to one route,
  it belongs in the shared function.
- **The two slow steps run as jobs** (`backend/jobs.py`), polled rather than
  awaited. Job state is in memory on purpose: a job whose thread died cannot be
  resumed, and persisting it would let the UI show something untrue after a
  restart. Cancellation is cooperative -- checked between Whisper segments and
  between streamed Ollama tokens.
- **A retry must change something.** At temperature 0 an identical request
  reproduces an identical failure, so `_request_body()` takes the previous
  failure kind. `done_reason == "length"` means truncation: give it the room
  actually left in the context window, since doubling is a guess that can still
  be too small (observed: 24 -> 48 failed again). Anything else means it had
  room and still produced rubbish: raise the temperature and tell it what went
  wrong. Transport errors are never retried.
- **`num_predict` is set to the response reserve**, so the reserved tokens are a
  real budget rather than an assumption. If a legitimate extraction outgrows it
  the truncation retry recovers, which is why the two work as a pair.
- **Analysis streams from Ollama** (`"stream": True`). That is what makes it
  interruptible and observable; the chunks are reassembled so callers still get
  one response. Progress there is reported as tokens generated with a `None`
  fraction, because the total is unknowable mid-generation and an invented
  progress bar is worse than none. Tests must patch `httpx.stream`, not
  `httpx.post`.
- **`[hidden] { display: none !important; }` is load-bearing -- do not remove
  it.** The `hidden` attribute is applied by the browser's default stylesheet,
  so any author rule setting `display` silently beats it. `.status-row` and
  `.workflow` both use `display: flex`, which left the progress row and
  workflow block on screen from page load showing their placeholder text
  ("Working...", "RUNNING") as though a job were running. A component rule can
  add `display` at any time, so the guard is global rather than per-selector.
  `tests/test_frontend_parity.py` asserts the rule exists with `!important`,
  and that every element the JS reveals starts out `hidden` in the markup.
- **The UI is bilingual (en/de)**; `STRINGS` in `app.js` holds one flat table
  per language and `tests/test_i18n.py` asserts they have identical keys and
  match the backend's languages. Backend errors are translated **by their
  `error` code**, not their text, which is why the server can keep returning
  English detail strings. Follow-up questions are localised server-side in
  `QUESTIONS_BY_LANGUAGE`; field *values* follow the spoken language.
- **No decorative telemetry.** The mockup showed invented figures (WebGPU ON,
  12ms latency, 98% clarity, 99.8% precision); those slots are filled with real
  values instead -- the actual engine, the actual elapsed time, the actual cache
  size. Do not add a metric the system cannot measure.
- **`tests/test_frontend_parity.py` checks element lookups resolve**, and has a
  second test asserting that check is not vacuous. Changing how app.js looks up
  elements once made it match nothing and pass regardless; if the lookup pattern
  changes again, update `_referenced_ids()`.
- **Ollama truncates an oversized prompt silently, from the front.** Measured:
  ~6,600 tokens sent with `num_ctx` 2048 returned HTTP 200 having evaluated
  1,026, and the extraction confidently invented a value for the part it never
  saw. Since a brain-dump states its goal first, that is the half which gets
  lost. `check_transcript_fits()` therefore refuses before calling the model,
  and `_warn_if_truncated()` compares `prompt_eval_count` afterwards as a net.
  **Never relax this into a silent truncate-and-continue.**
- **Audio length is capped separately from upload size** (`VPB_MAX_AUDIO_SECONDS`,
  30 min). 100 MB of 24 kbps Opus is ~10 hours; the byte cap alone allows a job
  that would run for hours and then produce an unanalysable transcript. The
  duration is probed with `ffprobe` *before* the concurrency slot is taken, so a
  bad upload is refused in ~0.3s instead of queueing behind real work.
- **The two limits are deliberately consistent**: 30 minutes of speech is about
  4,500 words (~6,000 tokens), which fits the ~7,200-token transcript budget at
  the default `num_ctx`. Raising one without the other reopens the hole.
- **The model name is allow-listed** (`ALLOWED_MODELS`) because the per-request
  override reaches the Hugging Face hub; it must never be free-form.
- **Code, not the model, decides what is missing** in v0.3 (null/empty fields).
- **A deterministic template builds the final prompt** in v0.4, not the LLM.
- `/health` reports `ffmpeg` presence so the UI can warn *before* someone
  records five minutes for nothing.

## Roadmap

- **v0.1 - done.** Recording page, `/transcribe`, editable transcript, plus a
  model selector and vocabulary hint for accuracy. Verified by the user:
  recording, upload, editing, live word count, copy and mic-denial all work.
- **v0.2 - done.** `/analyze` with Ollama structured output (`format` + JSON
  schema from `Extraction`), field view plus a raw-JSON panel in the UI.
- **v0.3 - done.** Rather than a separate question loop, every field is
  editable inline and a missing field shows its question as the placeholder.
  Same intent, fewer steps, and it also lets the user fix what the model
  paraphrased badly.
- v0.4 - deterministic template builder, copy button, token estimate
  (chars / 4). Completes v0.
- v1 - voice answers, URL detection with fetched summaries, target-model
  profiles, prompt history.

**Workflow: finish a milestone, then stop and confirm before starting the next.**

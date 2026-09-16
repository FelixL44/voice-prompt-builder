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
  builder.py     deterministic prompt template (v0.4, not yet written)
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
- v0.3 - question loop, at most 1-2 rounds, every question skippable.
- v0.4 - deterministic template builder, copy button, token estimate
  (chars / 4). Completes v0.
- v1 - voice answers, URL detection with fetched summaries, target-model
  profiles, prompt history.

**Workflow: finish a milestone, then stop and confirm before starting the next.**

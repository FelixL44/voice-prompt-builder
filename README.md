# VoxPrompt

**Fully local: your voice never leaves your machine.**

Ramble into your microphone for a few minutes, and get back a clean,
well-structured prompt for a large, expensive LLM. Transcription and analysis
both run on your own CPU &mdash; the only network request the app ever makes is
downloading the Whisper model once.

Thinking out loud is easy; writing a good prompt is not. This bridges the two.

---

## Status

**v0 complete** &mdash; record or upload audio, transcribe it locally, edit the
transcript, extract it into structured fields with a local LLM, edit those
fields, and assemble the final prompt. All of it on your own machine.

## Requirements

- macOS (built and tested on an **Intel / x86_64** Mac, CPU only)
- Python 3.11+ &mdash; [`uv`](https://docs.astral.sh/uv/) manages this for you
- `ffmpeg`
- [Ollama](https://ollama.com), running, with a small model pulled

Deliberately **no PyTorch or TensorFlow**. Transcription runs on
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) via CTranslate2,
which keeps the install small and works on Intel Macs, where recent torch
builds are no longer published.

## Setup

```bash
brew install ffmpeg
git clone <your-repo-url> && cd voice-prompt-builder
uv sync

# For the analysis step
ollama pull llama3.2:3b
ollama serve          # or just launch the Ollama app
```

## Run

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

Open <http://127.0.0.1:8000>, click **Start recording**, talk, then click stop.

The Whisper model loads in the background as the server starts, so the first
recording does not pay for it. The very first run also downloads it (~150 MB
for `base`); `/health` reports `model_warming` while that happens, and the
footer says "loading" rather than leaving you guessing.

After that, expect roughly **a tenth of the audio length** with `base` &mdash;
about 30 seconds for a 5-minute brain-dump on a CPU-only Intel Mac.

`GET /health` also reports whether the configured Ollama model has actually
been pulled, not just whether Ollama is running, so a missing model is a
warning on page load instead of a failure after you have already recorded.

Microphone access needs a secure context, which `127.0.0.1` counts as. If the
browser never prompts, check
*System Settings &rsaquo; Privacy &amp; Security &rsaquo; Microphone*.

## Configuration

Every setting is an environment variable:

| Variable | Default | Notes |
|---|---|---|
| `VPB_WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, `medium` |
| `VPB_COMPUTE_TYPE` | `int8` | CTranslate2 quantisation |
| `VPB_LANGUAGE` | auto-detect | e.g. `en`, `de` |
| `VPB_INITIAL_PROMPT` | generic tech hint | Decoding hint; see Accuracy below |
| `VPB_VAD_FILTER` | `true` | Skip silence before transcribing |
| `VPB_BEAM_SIZE` | `5` | Lower is faster, slightly less accurate |
| `VPB_MAX_UPLOAD_BYTES` | `104857600` | 100 MB |
| `VPB_MAX_AUDIO_SECONDS` | `1800` | 30 minutes; a size cap is not a length cap |
| `VPB_OLLAMA_RESPONSE_RESERVE` | `512` | Context tokens held back for the reply |
| `VPB_DB_PATH` | `./data/sessions.db` | Session history |
| `VPB_MAX_ACTIVE_JOBS` | `4` | Unfinished jobs allowed at once |
| `VPB_JOB_RETENTION_S` | `120` | How long a finished job stays readable |
| `VPB_FFMPEG_PATH` | `ffmpeg` | |
| `VPB_MAX_CONCURRENT_TRANSCRIPTIONS` | `1` | Queue rather than thrash a CPU |
| `VPB_WARMUP` | `true` | Load the model at startup, not on first use |
| `VPB_OLLAMA_MODEL` | `llama3.2:3b` | Any pulled model; 3B&ndash;8B is the usable range |
| `VPB_OLLAMA_URL` | `http://localhost:11434` | |
| `VPB_OLLAMA_TIMEOUT_S` | `600` | |
| `VPB_OLLAMA_NUM_CTX` | `8192` | Must fit transcript + system prompt |
| `VPB_TMP_DIR` | `./tmp` | Scratch space, wiped after each request |

```bash
VPB_WHISPER_MODEL=small VPB_LANGUAGE=en uv run uvicorn backend.main:app --port 8000
```

## Accuracy

Speech-to-text on a CPU-only Intel Mac is a real trade-off, so the UI exposes
both levers under **Accuracy settings**.

**Model size.** Measured here on jargon-heavy speech:

| Model | 5 min of audio | Quality |
|---|---|---|
| `tiny` | ~18s | Draft only |
| `base` | ~30s | Fine for plain prose, mangles technical terms |
| `small` | ~2 min | **Recommended.** Clearly better on names and acronyms |
| `medium` | ~6 min | Rarely worth the wait on CPU |

**Vocabulary.** The bigger win, and it is free. Whisper conditions decoding on
an initial prompt, so listing the terms it is likely to mishear &mdash; product
names, acronyms, people &mdash; measurably improves them at **no cost in time**.
The same 19-second clip, with `small`:

> *without:* "a fast API backend with **pedantic** schemas and an
> **olama-structured** output call &hellip; the deliverable is a markdown"
>
> *with:* "a **FastAPI** backend with **Pydantic** schemas and an **Ollama**
> structured output call &hellip; the deliverable is a markdown **RFC with
> latency benchmarks**"

Note that the hint also recovered the truncated ending: when Whisper is
uncertain it can stop early, and grounding it in real vocabulary prevents that.

The transcript box stays editable regardless, because no model gets everything
right.

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | The recording page |
| `GET /health` | Whether `ffmpeg` is present and which model is configured |
| `POST /transcribe` | Multipart `audio` file &rarr; transcript JSON. Optional `model` and `vocabulary` fields |
| `POST /analyze` | `{"transcript": "..."}` &rarr; structured fields plus what is missing |
| `POST /build` | `{"extraction": {...}}` &rarr; the assembled prompt |
| `POST /jobs/transcribe` | Same as `/transcribe`, but returns a job id |
| `POST /jobs/analyze` | Same as `/analyze`, but returns a job id |
| `GET /jobs/{id}` | Poll state, progress and result |
| `POST /jobs/{id}/cancel` | Ask a running job to stop |
| `GET /sessions` | Session history, newest first |
| `PUT /sessions` | Create or update a session |
| `DELETE /sessions/{id}` | Delete one, or all with no id |
| `GET /docs` | Interactive OpenAPI docs |

```bash
curl -s -X POST \
  -F "audio=@recording.webm" \
  -F "model=small" \
  -F "vocabulary=Kubernetes, Pydantic, Ollama" \
  http://127.0.0.1:8000/transcribe
```

```json
{
  "text": "I want to build a small web application that ...",
  "language": "en",
  "duration": 22.64,
  "model": "base",
  "elapsed_s": 2.3,
  "segments": [{ "start": 0.0, "end": 4.8, "text": "I want to build ..." }]
}
```

Errors come back with the same shape throughout &mdash; a machine-readable
`error` code and a `detail` string meant to be shown to a human:

```json
{ "error": "ffmpeg_missing", "detail": "ffmpeg was not found on PATH. Install it with: brew install ffmpeg" }
```

## Structuring (v0.2)

`POST /analyze` sends the transcript &mdash; **as edited**, not the audio &mdash; to a
local model via Ollama, constrained to the JSON schema of the `Extraction`
Pydantic model using Ollama's `format` parameter. So the response is always
parseable JSON with the right shape.

Parseable is not the same as truthful, and a 3B model will fill a blank with a
plausible guess if allowed to. Two things guard against that:

- The system prompt ([`backend/prompts/extract.md`](backend/prompts/extract.md))
  states that empty fields are the expected answer, not a failure.
- **Code decides what is missing**, never the model. `find_missing()` walks the
  extraction and reports every null, blank or empty list. A model that invented
  an answer would not report it as absent, so it is never asked to.

Model stand-ins for absence are normalised first: `llama3.2:3b` writes the
*string* `"null"` rather than emitting JSON `null`, which would otherwise look
like a real answer and leave the question loop with nothing to ask.

```json
{
  "extraction": {
    "goal": "Draft a first reply for incoming tickets, for an agent to edit",
    "audience": "Support agents",
    "constraints": ["Cannot send automatically", "Must handle German and English"],
    "examples": [],
    "output_format": "Reply with confidence score"
  },
  "missing": [
    { "field": "examples", "question": "Do you have an example of what good looks like?" }
  ],
  "model": "llama3.2:3b",
  "elapsed_s": 21.8
}
```

Expect **20&ndash;90 seconds** depending on transcript length. A larger model such
as `qwen2.5:7b` follows the "do not invent" instruction more reliably, at
roughly double the time.

## The console

The UI is a three-pane agentic console: sessions on the left, the workflow
stream in the middle, accuracy settings on the right.

**Sessions are stored on the server**, in a SQLite file under `data/`. Each run
is saved with its transcript, variables and final prompt; reopening one restores
every stage, and it survives a cleared browser cache or a restart. Still
entirely local &mdash; the server writes one file and talks to nobody.

**The two slow steps run as jobs.** Transcription and extraction return a job id
immediately, then report real progress and can be cancelled. Whisper progress
comes from segment timestamps, so it is measured rather than guessed; extraction
streams from Ollama and reports tokens generated, because the total is genuinely
unknowable mid-generation. Cancellation is cooperative &mdash; checked between
segments and between streamed tokens &mdash; so it lands within a second or two.

**The interface switches between English and German** with the toggle in the
header. That covers the follow-up questions too, which come from the server.
Extracted *values* follow whatever language you actually spoke.

Every number in the interface is a real measurement &mdash; actual engine,
actual elapsed time, actual cache size. There is no decorative telemetry.

## Filling the gaps (v0.3)

Every extracted field is editable in the UI, not only the empty ones: the model
paraphrases, and you are the authority on what you meant.

Fields the backend reported missing show **the follow-up question as the
placeholder**, so answering is just typing and skipping is just leaving it
blank. List fields (constraints, examples, success criteria) take one item per
line, and pasted bullets (`-`, `*`, `•`) are stripped automatically.

The status line and the raw-JSON panel update as you type, so what you see is
exactly what the prompt builder will receive. Re-running the analysis asks
first, since it replaces every field.

## Building the prompt (v0.4)

`POST /build` assembles the final prompt from the edited fields. **No model is
involved.** The extraction is already structured, so turning it into a prompt is
string assembly: instant, reproducible, and diffable. The same fields always
produce byte-identical output, which a model could never promise.

Sections are XML-style, and empty ones are omitted rather than left blank &mdash;
an empty `<examples>` tag announces a section and then says nothing, which
invites the model to fill the gap itself.

```
<context>          reference material first: longest, and read as background
<task>             the instruction, after the background
<audience>
<constraints>      one dash-prefixed item per line
<examples>
<output_format>    formatting lands near generation, where it still holds
<success_criteria>
```

The UI shows the prompt with a copy button and a rough token estimate
(characters / 4 &mdash; enough to warn you that a prompt is large, which is all
that number is for). Editing a field marks a built prompt **out of date**, so
you never copy something that no longer matches the fields above.

## Limits, and why they exist

Two limits are enforced by the server, and they are the same problem seen twice.

**Audio length: 30 minutes.** A size cap is not a length cap &mdash; 100 MB of
24 kbps Opus is nearly ten hours, which would occupy the machine for hours.
`ffprobe` reads the duration from container metadata before anything is
decoded, so an over-long file is refused in **under a second** rather than
after a long transcription.

**Transcript length: whatever fits the context window.** This one matters more,
because the failure is silent. Measured against `llama3.2:3b`: a prompt of
~6,600 tokens sent with `num_ctx` 2048 returned **HTTP 200** having evaluated
only 1,026 tokens. Ollama discards the overflow without complaint, and it
truncates from the *front* &mdash; which is where a spoken brain-dump states its
goal. In that test a codename given in the first sentence came back fabricated,
while a deadline from the last sentence came back correct.

So the transcript is measured against the window before the model is called,
and an oversized one is refused with the value that would fit:

```json
{
  "error": "transcript_too_long",
  "detail": "The transcript is about 25,000 tokens but only 7,187 fit the context window. Ollama would silently drop the beginning of it, which is usually where the goal is. Shorten the recording, or restart the server with VPB_OLLAMA_NUM_CTX=32768."
}
```

A rough estimate can be wrong, so `prompt_eval_count` is compared against the
tokens sent after every call and a large shortfall is logged as a warning.

## Privacy

- Audio is converted and transcribed in a per-request temp directory that is
  deleted afterwards, **including when transcription fails**. There is a test
  asserting this.
- Nothing is logged except durations and sizes &mdash; never transcript text.
- Session history is written to `data/sessions.db` on this machine. It is the
  only thing the app keeps, it never leaves, and the settings pane clears it.
- Working files are swept at startup as well as deleted after each run. A
  `kill -9` or a power cut cannot run a cleanup handler, so anything found in
  the working directory on the next start is an orphan and is removed.
- A finished job holds its transcript in memory only until the browser collects
  it; the client then releases it explicitly, and a short retention window is
  the backstop for a client that never returns.
- No telemetry, no analytics, no outbound requests beyond the model download.

## Tests

```bash
uv run pytest -q
```

Sample audio is generated at test time with macOS `say` rather than committed
to the repo, so there are no binaries in version control. Tests skip cleanly on
machines without `say` or `ffmpeg`.

## Roadmap

- [x] **v0.1** &mdash; recording page, `/transcribe`, editable transcript
- [x] **v0.2** &mdash; `/analyze`: Ollama extracts the transcript into a fixed JSON
      schema using structured output
- [x] **v0.3** &mdash; every extracted field is editable, with the follow-up
      question shown in place for anything missing, and skipping is just
      leaving it blank
- [x] **v0.4** &mdash; deterministic template assembles the final prompt with
      XML-style sections, copy button, token estimate
- [ ] **v1** &mdash; voice answers to follow-ups, URL detection with fetched
      summaries, target-model profiles, prompt history

## How it works

```
browser (MediaRecorder, WebM/Opus)
        |  multipart POST /transcribe
        v
ffmpeg  ->  16 kHz mono PCM WAV
        v
faster-whisper (CTranslate2, int8, CPU)
        v
editable transcript  ->  [v0.2] Ollama extraction  ->  [v0.4] prompt template
```

The design keeps the LLM on a short leash: it extracts structure, while
ordinary code decides what is missing and a fixed template assembles the final
prompt. That makes the output reproducible and debuggable instead of subject to
a model's mood.

## License

MIT

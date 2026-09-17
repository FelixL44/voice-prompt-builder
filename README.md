# Voice Prompt Builder

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

## Privacy

- Audio is converted and transcribed in a per-request temp directory that is
  deleted afterwards, **including when transcription fails**. There is a test
  asserting this.
- Nothing is logged except durations and sizes &mdash; never transcript text.
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

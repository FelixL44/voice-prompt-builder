# Voice Prompt Builder

**Fully local: your voice never leaves your machine.**

Ramble into your microphone for a few minutes, and get back a clean,
well-structured prompt for a large, expensive LLM. Transcription and analysis
both run on your own CPU &mdash; the only network request the app ever makes is
downloading the Whisper model once.

Thinking out loud is easy; writing a good prompt is not. This bridges the two.

---

## Status

**v0.1** &mdash; record or upload audio, transcribe it locally, edit the transcript.
The analysis and prompt-building steps are on the roadmap below.

## Requirements

- macOS (built and tested on an **Intel / x86_64** Mac, CPU only)
- Python 3.11+ &mdash; [`uv`](https://docs.astral.sh/uv/) manages this for you
- `ffmpeg`
- [Ollama](https://ollama.com) &mdash; not needed until v0.2

Deliberately **no PyTorch or TensorFlow**. Transcription runs on
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) via CTranslate2,
which keeps the install small and works on Intel Macs, where recent torch
builds are no longer published.

## Setup

```bash
brew install ffmpeg
git clone <your-repo-url> && cd voice-prompt-builder
uv sync
```

## Run

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

Open <http://127.0.0.1:8000>, click **Start recording**, talk, then click stop.

The first transcription downloads the Whisper model (~150 MB for `base`) and
takes a minute. After that, expect roughly **a tenth of the audio length** &mdash;
about 30 seconds for a 5-minute brain-dump on a CPU-only Intel Mac.

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
- [ ] **v0.2** &mdash; `/analyze`: Ollama extracts the transcript into a fixed JSON
      schema using structured output
- [ ] **v0.3** &mdash; follow-up questions for whatever is missing, at most two
      rounds, every one skippable
- [ ] **v0.4** &mdash; deterministic template assembles the final prompt with
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

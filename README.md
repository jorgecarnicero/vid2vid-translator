# 🎬 AI Video Dubbing Pipeline (Spanish → English)

Automatic video dubbing pipeline: give it a video with Spanish audio and it will **transcribe, translate, and re-voice it in English using your own cloned voice**, syncing the new audio back onto the original video.

Currently supports **Spanish → English** only. More language pairs are planned.

## How it works

The pipeline runs in 7 stages (see `main.py`):

1. **Audio extraction** — `ffmpeg` pulls the audio track out of the input video into an intermediate `.ogg` file.
2. **Transcription** — [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) transcribes the Spanish audio with **word-level timestamps**, which are later grouped into sentence-like chunks (a gap between words below `UMBRAL_PAUSA_AGRUPACION` keeps them in the same chunk).
3. **Global summary** — before translating chunk by chunk, Ollama is asked to summarize the whole video and list recurring technical terms / UI labels / proper nouns. This summary is injected as context into every later translation call so terminology stays consistent across the whole video (instead of drifting sentence by sentence).
4. **Sequential translation + anchor detection** — each chunk is translated ES→EN with Ollama, keeping the previous sentence as context so translations flow naturally across chunk boundaries. If a chunk contains a quoted string, a filename, or a capitalized identifier (an "anchor" — e.g. a button label or a file the user clicks on screen), the pipeline detects it, locates its exact timestamp in the original Spanish audio, and asks the model to mark its English equivalent so it can be re-synced to the same moment on screen.
5. **Parallel audio generation** — each translated chunk is sent to **VoiceBox** (local TTS/voice-cloning) to generate the English audio in your cloned voice, using a thread pool (`MAX_HILOS` workers). Segments are stretched/compressed and padded to match the original timing (with anchor-aware splitting for the segments detected in step 4), and results are cached to disk so re-runs don't regenerate unchanged segments.
6. **Assembly** — all generated segments are stitched together chronologically. Real pauses in the original video are preserved as silence; small gaps are smoothed with a crossfade so consecutive sentences don't sound choppy.
7. **Final render** — `ffmpeg` muxes the new English audio track back onto the original video (video stream copied, audio re-encoded to AAC).

## Requirements

- Python 3.10+
- `ffmpeg` installed and available on your `PATH`
- A GPU with CUDA is recommended (Whisper runs on `cuda:0` automatically if available, otherwise falls back to CPU)
- [Ollama](https://ollama.com) running locally, with a translation model pulled
- [VoiceBox](https://github.com) (or your local instance) running locally, with a voice profile created

Python packages:
```bash
pip install torch transformers pydub requests
```

> `transformers` will download `openai/whisper-large-v3-turbo` automatically the first time you run the script — no manual download needed.

## Setup

### 1. Install and run Ollama

Ollama is what translates and polishes the transcript.

```bash
# Install Ollama: https://ollama.com/download
ollama pull qwen2.5:7b   # or whichever model you configure in MODELO_OLLAMA
ollama serve             # usually starts automatically after install
```

By default the pipeline expects Ollama's chat API at `http://localhost:11434/api/chat` (see `URL_OLLAMA` in `constantes.py`).

### 2. Install and run VoiceBox

VoiceBox is the free/open-source voice cloning engine used for text-to-speech.
https://voicebox.sh/
1. Install and run VoiceBox locally (it should expose a `/generate` endpoint, by default on `http://127.0.0.1:17493`).
2. Create a **voice profile** from a sample of your own voice inside VoiceBox.
3. Copy the resulting **profile ID** — you'll need it for `VOICE_PROFILE` in `constantes.py`.

### 3. Whisper

Nothing to install manually — the first run of `main.py` downloads the Whisper Turbo model through `transformers` automatically.

## Configuration (`constantes.py`)

| Variable | What it does |
|---|---|
| `VIDEO_ENTRADA` | Path to your **input video**. Put the video you want to dub here. |
| `AUDIO_TEMPORAL` | Intermediate extracted audio file (auto-deleted after the run). |
| `AUDIO_SALIDA` | Final synced English audio track (`.wav`), before muxing into the video. |
| `VIDEO_FINAL` | Output path for the finished dubbed video. |
| `CACHE_DIR` | Where translated text + generated TTS segments are cached, keyed by content hash, so re-running the script skips work that hasn't changed. |
| `MAX_HILOS` | Number of parallel worker threads used to generate TTS audio (Stage 5). |
| `MIN_INICIO_MS` | The dubbed voice never starts before this many milliseconds into the video. |
| `UMBRAL_CONTINUIDAD_MS` | Gaps between segments smaller than this are treated as continuous speech (crossfaded) rather than a real pause. |
| `CROSSFADE_MS` | Crossfade duration used when stitching two segments that are meant to sound continuous. |
| `FADE_EDGES_MS` | Fade in/out applied to every TTS segment to avoid audio clicks. |
| `UMBRAL_PAUSA_AGRUPACION` | Pause (in seconds) between words below which Whisper's word-level output gets merged into the same sentence chunk. |
| `TIMEOUT_OLLAMA` / `TIMEOUT_VOICEBOX` | Request timeouts (seconds) so a stuck local server doesn't hang the pipeline silently. |
| `MAX_INTENTOS_STATUS` | Max polling attempts (≈1/sec) while waiting for a VoiceBox job to finish. |
| `URL_OLLAMA` / `URL_VOICEBOX` | Local API endpoints for Ollama and VoiceBox. |
| `MODELO_OLLAMA` | Which Ollama model to use for the global summary and translation (e.g. `qwen2.5:7b`). |
| `VOICE_PROFILE` | The VoiceBox profile ID for your cloned voice. |
| `TARGET_LANGUAGE` | Output language code passed to VoiceBox (currently `"en"`). |
| `CACHE_VERSION` | Bump this string whenever you change the translation prompt or anchor logic, to invalidate old cache entries without deleting the cache folder by hand. |
| `MAX_PAUSA_INTERNA` | Cap (ms) on how much silence can be inserted *inside* a stretched TTS segment when padding it to match the original timing. |
| `ANCHOR_REGEX` | Regex used to detect "anchors" in the Spanish text: quoted strings, or capitalized words/identifiers (UI labels, filenames, proper nouns) that must be translated consistently and time-aligned. |

## Usage

1. Drop your source video into the project folder and point `VIDEO_ENTRADA` at it in `constantes.py`.
2. Make sure Ollama and VoiceBox are both running locally.
3. Run the pipeline:

```bash
python main.py
```

4. The finished dubbed video will be written to whatever path `VIDEO_FINAL` points to.

The console prints progress through all 7 stages, including each Spanish/English chunk pair as it's translated, and a summary of any segments that failed to generate audio (these are left silent in the final video rather than failing the whole run).

## Caching

Every translated + generated segment is cached under `CACHE_DIR`, keyed by a hash of the Spanish text, voice profile, target language, and `CACHE_VERSION`. Re-running the script on the same video only regenerates segments that changed — useful when you're iterating on the translation prompt or fixing a single bad segment. Bump `CACHE_VERSION` to force a full re-generation.

## Roadmap

- [ ] Support additional source/target language pairs
- [ ] Streamlit UI for uploading videos and running the pipeline interactively (no need to edit `constantes.py` by hand)
- [ ] Voice profile management from the UI

## Notes

- Code and console output are currently in Spanish (variable/function names, log messages); this README documents the project in English for wider reach.
- Everything runs locally — no cloud APIs are called; only your local Ollama and VoiceBox instances.
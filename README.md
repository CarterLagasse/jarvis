# Jarvis — Push-to-Talk Voice Assistant

Hold a key, talk, release. Transcription runs locally, answers come from
Gemini's free tier, and speech comes from a local TTS engine. No billing
anywhere, no cloud TTS.

## Quickstart

```
pip install faster-whisper sounddevice pynput numpy openai
pip install edge-tts            # default voice (free, needs net)

copy .env.example .env          # then fill in GEMINI_API_KEY
```

Get the key at `https://aistudio.google.com/apikey` (free tier, no billing).
Put it in `.env` next to `push_to_talk.py`. The startup
banner prints `key=set/MISSING` so you can verify before talking.

```
python push_to_talk.py                       # hold PrintScreen to talk
python push_to_talk.py --list-mics           # check audio devices
python push_to_talk.py --no-speech           # text replies only
```

Press the hotkey again while it speaks to interrupt. Esc quits.

## How it works

```
[hotkey hold] → mic (16 kHz) → faster-whisper (local, GPU) → transcript
      → router: local tool? weather/sports? web search? → else Gemini
      → reply sink (internal queue + reply.txt) → local TTS → speaker
```

- **STT** is fully local (`base.en` Whisper, CUDA if available).
- **LLM** is `gemini-3.5-flash-lite` by default, stepping up to
  `gemini-3.5-flash` (code/reasoning) or `gemini-2.5-pro` ("think harder").
- **Fresh facts** (weather, scores, news) come from keyless free APIs
  (Open-Meteo, ESPN, DuckDuckGo) — no quotas, no keys. Gemini search
  grounding is a last resort only.
- **The LLM never touches audio.** It emits text into an internal queue
  (`speech_q`) plus a `reply.txt` tee; a separate worker speaks it. Swap
  voices without touching model code.

## Voice (TTS)

All speech is local — Gemini TTS was removed. Pick an engine
(`--tts-engine` or `JARVIS_TTS_ENGINE`, default `auto` → first installed):

| Engine | Quality | Offline | Install |
|--------|---------|---------|---------|
| `edge` | Best free | No | `pip install edge-tts` |
| `kokoro` | Very good | Yes | `pip install kokoro-onnx soundfile` + model files (see TTS.md) |
| `sapi` | Robotic, instant | Yes | `pip install pyttsx3` |
| `none` | — | — | text-only |

```
python tts_local.py --list-engines
python tts_local.py --text "Hello, I am Jarvis" --engine edge
echo Hello | python tts_local.py --stdin
```

Voices: `JARVIS_TTS_VOICE_EDGE=en-US-AriaNeural` (list with
`edge-tts --list-voices`), `JARVIS_TTS_VOICE_KOKORO=af_heart`,
`JARVIS_TTS_WPM=175` for SAPI rate. Replies stream to TTS as they generate:
first chunk at ~60 chars for fast audio start, then 140–400 on clause
boundaries with markdown/links stripped, so speech starts early and prosody
stays natural.

## Hotkey

PrintScreen is swallowed system-wide on Windows (no screenshots fire).
Trigger key is `.env`-only (`JARVIS_HOTKEY=printscreen` or `f9`) — there is
no CLI flag for it. If PrintScreen misbehaves, set `JARVIS_HOTKEY=f9`.

## Configuration

| Var | Required | Purpose |
|-----|----------|---------|
| `GEMINI_API_KEY` | yes | base + step-ups + search fallback (free tier) |
| `OPENROUTER_API_KEY` | no | free Laguna coding model ("ask laguna", 429 shed) |
| `JARVIS_LOCATION` | no | default city for "what's the weather" (e.g. `Chicago`) |
| `JARVIS_HOTKEY` | no | `printscreen` (default) / `f9` — `.env` only, no flag |
| `JARVIS_TTS_ENGINE` | no | `auto` / `edge` / `kokoro` / `sapi` / `none` |
| `JARVIS_MIC` | no | mic index (`1`) or full device name (see `--list-mics`) |
| `JARVIS_USE_GPU` | no | `1` (default) / `0` for Whisper |
| `JARVIS_NOTES` | no | notes file (default `notes.txt`) |
| `JARVIS_TOOLS` / `JARVIS_SEARCH` / `JARVIS_WEB` | no | `1`/`0` toggles for local tools, Gemini grounding, keyless web |

## Files

| File | Owns |
|------|------|
| `push_to_talk.py` | hotkey, mic, STT, LLM loop, sink, playback |
| `config.py` | settings + CLI flags + `.env` loader |
| `router.py` | routing heuristics (no network, fail-closed) |
| `tools.py` | local tools, grounding fallback, voice summaries |
| `webtools.py` | keyless providers (search, weather, scores) |
| `tts_local.py` | local TTS engines, pipe consumer |

## Further reading

- `ROUTING.md` — model decision table, limits, failure policy
- `TOOLS.md` — ability triggers, quotas, add-a-tool recipe
- `CAPABILITIES.md` — what works now + recommended next abilities

"""Jarvis voice assistant: hold hotkey to talk, release to send.

Speech -> text (local faster-whisper) -> router (router.py: local tool,
keyless web, or step-up) -> reply sink (internal queue + reply.txt tee) ->
local TTS (tts_local.py: edge/kokoro/sapi, no Gemini TTS).

Press the hotkey again while it's speaking to interrupt. Esc quits.

Install:
  pip install -r requirements-voice.txt   # or the manual set below
  pip install faster-whisper sounddevice pynput numpy openai
  pip install edge-tts            (default voice, free, needs net)
  # offline voice alternative:
  pip install kokoro-onnx soundfile   (+ model files, downloaded separately)
  # zero-download fallback on Windows:
  pip install pyttsx3

Keys (copy `.env.example` to `.env` next to this file):
  GEMINI_API_KEY       required (free tier) — base + search grounding
  OPENROUTER_API_KEY   step-ups, "think harder"/"use ultra" (or OPEN_ROUTER_API_KEY)

Abilities (tools.py + webtools.py): local tools (clock, notes, open URL) run
with no API call; fresh-facts questions go to keyless web providers, with
Google Search grounding as last resort; everything else goes through the
model router (router.py).

Usage:
  python push_to_talk.py [--no-speech] [--print-reply]
                         [--tts-engine auto|edge|kokoro|sapi|none]
  python push_to_talk.py --tts-stdin   # pipe consumer: echo hi | python tts_local.py --stdin
  python push_to_talk.py --list-mics   # find JARVIS_MIC for .env
"""
import os
import queue
import re
import site
import sys
import threading
import time
from pathlib import Path

import faulthandler
faulthandler.enable()  # native tracebacks for any future segfaults

if sys.platform == "win32":
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    for sp in roots:
        for sub in ("cublas", "cudnn"):
            d = Path(sp) / "nvidia" / sub / "bin"
            if d.exists():
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]

import numpy as np

import config
import router
import tools
import tts_local
import webtools

ARGS = config.build_arg_parser().parse_args()

HOTKEY_NAME = config.HOTKEY_NAME  # .env only (JARVIS_HOTKEY)
HOTKEY_VK = config.HOTKEY_VK_MAP[HOTKEY_NAME]
QUIT_KEY = None  # set after pynput import (Esc)
SPEAK = not ARGS.no_speech and ARGS.tts_engine != "none"
TTS_ENGINE = None if ARGS.tts_engine == "none" else ARGS.tts_engine

# ---------------------------------------------------------------------------
# Pipe-consumer / mic-list modes: no mic, no LLM key needed.
# ---------------------------------------------------------------------------
if ARGS.list_mics:
    import sounddevice as sd

    print(sd.query_devices())
    sys.exit(0)

if ARGS.tts_stdin:
    sys.exit(tts_local.main(["--stdin", "--engine", TTS_ENGINE or "auto"]))

# --- LLM clients (lazy per provider, cached) --------------------------------
_clients: dict = {}


def _client_for(provider: str):
    if provider not in _clients:
        _clients[provider] = router.make_client(provider)
    return _clients[provider]


def _require_gemini_key():
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not found. Put it in a `.env` file next to "
                 "push_to_talk.py (see .env.example) or export it first.")


_require_gemini_key()

import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard

QUIT_KEY = keyboard.Key.esc
HOTKEY = {
    "printscreen": keyboard.Key.print_screen,
    "f9": keyboard.Key.f9,
}[HOTKEY_NAME]

if config.USE_GPU:
    model = WhisperModel(config.WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
else:
    model = WhisperModel(config.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

print("Warming up STT model...")
list(model.transcribe(np.zeros(config.SAMPLE_RATE, dtype=np.float32), beam_size=1)[0])

chunks = []
recording = False

# ---------------------------------------------------------------------------
# Reply sink: LLM text -> speech_q (internal queue, primary) + reply.txt tee
# + optional stdout (for `| python tts_local.py --stdin` piping).
# TTS never sees the LLM; it only consumes this sink.
# ---------------------------------------------------------------------------
speech_q: queue.Queue = queue.Queue()  # (epoch, text-chunk)
audio_q: queue.Queue = queue.Queue()  # (epoch, int16 audio, sample rate)
speech_epoch = 0
consecutive_failures = 0
gemini_rate_limited = False


def tee_reply(text: str):
    """Append final assistant reply to reply log + stdout if requested."""
    text = text.strip()
    if not text:
        return
    try:
        with open(config.REPLY_LOG, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except OSError as e:
        print(f"[warn] could not write {config.REPLY_LOG}: {e}")
    if ARGS.print_reply:
        # stdout is the external pipe: `... --print-reply | python tts_local.py --stdin`
        print(text, flush=True)


def speak(text: str, epoch: int):
    if SPEAK and text.strip():
        speech_q.put((epoch, text))


def tts_worker():
    while True:
        epoch, text = speech_q.get()
        if epoch != speech_epoch:
            continue
        try:
            result = tts_local.synthesize(text, engine=TTS_ENGINE)
        except Exception as e:
            print(f"[error] TTS failed ({TTS_ENGINE}): {e}")
            continue
        if result is not None and epoch == speech_epoch:
            audio_q.put((epoch, result[0], result[1]))
        # result None => sapi spoke directly (blocking); nothing to queue.


def playback_worker():
    while True:
        epoch, audio, rate = audio_q.get()
        if epoch != speech_epoch:
            continue
        sd.play(audio, rate)
        if epoch != speech_epoch:
            sd.stop()
            continue
        sd.wait()


def stop_speaking():
    global speech_epoch
    speech_epoch += 1

    def _cut():
        for q in (speech_q, audio_q):
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
        try:
            sd.stop()
        except Exception:
            pass

    threading.Thread(target=_cut, daemon=True).start()


if SPEAK:
    eng = tts_local.resolve_engine(TTS_ENGINE)
    if eng is None:
        print("[warn] no local TTS engine installed; speech muted. "
              "pip install edge-tts (or kokoro-onnx + model files, or pyttsx3).")
        SPEAK = False
    else:
        print(f"[tts] engine={eng}")
        threading.Thread(target=tts_worker, daemon=True).start()
        threading.Thread(target=playback_worker, daemon=True).start()

# ---------------------------------------------------------------------------
# LLM with free-tier routing
# ---------------------------------------------------------------------------
history = [{"role": "system", "content": config.SYSTEM_PROMPT}]
llm_lock = threading.RLock()  # reentrant: fallbacks re-enter one turn deep


def _startup_context() -> str:
    """Date + location appended to the system prompt (no network calls)."""
    import datetime
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    where = (config.DEFAULT_LOCATION
             or "unknown (user hasn't set JARVIS_LOCATION)")
    return (f" Today is {today}. The user is in {where} — treat that as "
            f"'here'/'my location' and never ask where they are for weather "
            f"or local questions.")


history[0]["content"] += _startup_context()


def ask_llm(text: str, note: str = "") -> str:
    global consecutive_failures, gemini_rate_limited
    with llm_lock:
        my_epoch = speech_epoch
        decision = router.classify(text, history_len=len(history),
                                   consecutive_failures=consecutive_failures,
                                   gemini_rate_limited=gemini_rate_limited)
        return ask_llm_with_decision(text, decision, my_epoch, note)


def ask_llm_with_decision(text: str, decision, my_epoch: int, note: str = "") -> str:
    """Run one LLM turn for an already-made decision. llm_lock must be held.

    `note` is appended to the user turn (e.g. search-failure context) so the
    model answers honestly instead of pretending to check live sources.
    Tries router.fallbacks_for() in order until one returns a reply.
    """
    global consecutive_failures, gemini_rate_limited
    with llm_lock:
        if decision.route != "direct":
            print(f"[route] -> {decision.provider}:{decision.model} ({decision.reason})")
        user_turn = text + (f"\n[{note}]" if note else "")
        history.append({"role": "user", "content": user_turn})
        reply, buf, pending, spoken_any = "", "", "", False
        chain = router.fallbacks_for(decision)

        for attempt_n, (prov, mod) in enumerate(chain):
            if attempt_n:
                print(f"[route] backup {attempt_n} -> {prov}:{mod}")
            # Model/temperature: Flash-Lite ignores custom temp; keep default.
            # Escalated models use low temperature for voice-safe determinism.
            kwargs = {"stream": True}
            if prov != "gemini" or decision.route != "direct":
                kwargs["temperature"] = 0.3

            stream_resp = None
            try:
                client = _client_for(prov)
                stream_resp = client.chat.completions.create(
                    model=mod, messages=history, **kwargs
                )
                for stream_chunk in stream_resp:
                    if speech_epoch != my_epoch:
                        print("\n[interrupted]")
                        break
                    if not stream_chunk.choices:
                        continue
                    piece = stream_chunk.choices[0].delta.content or ""
                    if not piece:
                        continue
                    reply += piece
                    print(piece, end="", flush=True)
                    buf += piece
                    # Incremental flush on sentence ends. First chunk uses the
                    # smaller fast-start size (audio begins sooner); the rest
                    # merge to MIN_CHARS_PER_CHUNK for natural prosody.
                    *sentences, buf = tts_local.SENT_SPLIT.split(buf)
                    for s in sentences:
                        pending = f"{pending} {s}".strip()
                        limit = (config.FIRST_CHUNK_CHARS if not spoken_any
                                 else config.MIN_CHARS_PER_CHUNK)
                        if len(pending) >= limit:
                            speak(pending, my_epoch)
                            pending = ""
                            spoken_any = True
                else:
                    print()
                    tail = f"{pending} {buf}".strip()
                    if tail:
                        # Final flush may be long; split properly for prosody.
                        for ch in tts_local.chunk_text(tail):
                            speak(ch, my_epoch)
            except Exception as e:
                msg = str(e)
                print(f"\n[error] LLM request failed ({mod}): {msg[:200]}")
                if "401" in msg and prov == "gemini":
                    print("[hint] Gemini rejected the key (401). Check GEMINI_API_KEY "
                          "in .env (https://aistudio.google.com/apikey).")
                if "401" in msg and prov == "openrouter":
                    print("[hint] OpenRouter rejected the key (401). Check "
                          "OPENROUTER_API_KEY in .env (https://openrouter.ai/settings/keys).")
                if prov == "gemini" and ("429" in msg or "rate" in msg.lower()
                                         or "quota" in msg.lower()):
                    gemini_rate_limited = True
            finally:
                try:
                    if stream_resp is not None:
                        stream_resp.close()
                except Exception:
                    pass
            if reply.strip():
                break  # backup chain stops at first real reply
            # Empty reply: reset speech buffers, try the next backup.
            buf, pending, spoken_any = "", "", False
        if reply.strip():
            consecutive_failures = 0
            if "429" not in reply:
                gemini_rate_limited = False
        else:
            history.pop()
            consecutive_failures += 1
            return ""
        history.append({"role": "assistant", "content": reply})
        if len(history) > config.MAX_HISTORY_MESSAGES + 1:
            del history[1:len(history) - config.MAX_HISTORY_MESSAGES]
        tee_reply(reply)
        return reply


def answer_direct(spoken: str, detail: str = ""):
    """Speak + log a tool/grounded answer without spending an LLM turn."""
    spoken = spoken.strip()
    if not spoken:
        return
    print(spoken)
    tee_reply(detail.strip() or spoken)  # sources stay in console/log only
    my_epoch = speech_epoch
    for ch in tts_local.chunk_text(spoken):
        speak(ch, my_epoch)


def handle_text(text: str):
    global consecutive_failures
    print(f"> {text}")
    # Order: local (instant) -> keyless APIs (weather/sports) -> web search
    # (DDG+Lite) -> Gemini grounding (last resort, quota-unreliable) -> LLM.
    kind, tool_detail = router.route_tools(text)
    if kind == "local":
        answer_direct(tools.run_local_tool(tool_detail, text))
        return
    if kind == "play":
        print("[tools] resolving song…")
        spoken, console = webtools.play_song_answer(text)
        if spoken:
            print(f"[tools] play: {console}")
            answer_direct(spoken, console)
            consecutive_failures = 0
            return
        print(f"[tools] play skipped: {console}")
        kind = "web"  # fall through to general search below
    if kind in ("weather", "sports", "news"):
        if kind == "weather":
            spoken, console = webtools.weather_answer(text)
        elif kind == "sports":
            spoken, console = webtools.sports_answer(text)
        else:
            spoken, console = webtools.news_answer()
        if spoken:
            print(f"[tools] {kind}: {console}")
            answer_direct(spoken, console)
            consecutive_failures = 0
            return
        print(f"[tools] {kind} skipped: {console} — trying web search")
        kind = "web"  # fall through to general search below
    if kind == "web":
        print(f"[tools] web search (trigger: {tool_detail})")
        spoken, console = webtools.web_search_answer(text)
        if spoken:
            print(f"[tools] web: {console}")
            answer_direct(spoken, console)
            consecutive_failures = 0
            return
        print(f"[tools] web search empty: {console} — trying grounding")
        g_spoken, g_console = tools.grounded_answer(text)
        if g_spoken:
            print(g_console)
            answer_direct(g_spoken, g_console)
            consecutive_failures = 0
            return
        print(f"[tools] grounded path skipped: {g_console}")
        # Last resort: plain LLM, told that live lookup failed so it answers
        # honestly instead of pretending to check sources.
        ask_llm(text, note=f"Live lookup failed ({g_console}). "
                           "Answer from training knowledge in one short spoken "
                           "reply, saying upfront you can't see live results.")
        return
    ask_llm(text)


# ---------------------------------------------------------------------------
# Recording + transcription
# ---------------------------------------------------------------------------
def audio_callback(indata, frames, time_info, status):
    if recording:
        chunks.append(indata.copy())


def transcribe(audio):
    t0 = time.time()
    try:
        segments, _ = model.transcribe(
            audio,
            language="en",
            beam_size=config.WHISPER_BEAM,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
    except Exception as e:
        print(f"[error] transcription failed: {e}")
        return
    if text:
        print(f"  (transcribed in {time.time() - t0:.2f}s)")
        handle_text(text)


def _blip(freq: float = 880.0, ms: int = 90, when: str = ""):
    """Short synthesized beep (no audio files). Threaded, never blocks hook."""
    def _play():
        try:
            n = max(1, int(config.SAMPLE_RATE * ms / 1000))
            t = np.arange(n, dtype=np.float32) / config.SAMPLE_RATE
            tone = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            sd.play(tone, config.SAMPLE_RATE)
            sd.wait()
        except Exception as e:
            print(f"[warn] blip failed ({when}): {e}")

    threading.Thread(target=_play, daemon=True).start()


def start_recording():
    global recording
    if not recording:
        stop_speaking()
        chunks.clear()
        recording = True
        _blip(880.0, ms=90, when="start")


def _resample_mono(audio, src_rate: int) -> "np.ndarray":
    """Downmix + resample mic audio to mono 16k for STT."""
    dst_rate = config.SAMPLE_RATE
    audio = np.asarray(audio, dtype=np.float32)
    if INPUT_CHANNELS > 1 and audio.size % INPUT_CHANNELS == 0:
        audio = audio.reshape(-1, INPUT_CHANNELS).mean(axis=1)
    audio = audio.flatten()
    if src_rate == dst_rate or len(audio) < 2:
        return audio
    n = max(1, int(len(audio) * dst_rate / src_rate))
    return np.interp(
        np.linspace(0, len(audio) - 1, n),
        np.linspace(0, len(audio) - 1, len(audio)),
        audio,
    ).astype(np.float32)


def stop_recording():
    global recording
    if recording:
        recording = False
        _blip(440.0, ms=90, when="stop")
        if chunks:
            audio = np.concatenate(chunks).flatten()
            audio = _resample_mono(audio, INPUT_RATE)
            # Run in a thread so the keyboard hook is never blocked.
            threading.Thread(target=transcribe, args=(audio,), daemon=True).start()


# --- Keyboard handling -------------------------------------------------------
# On Windows the win32 filter swallows the hotkey system-wide (suppress_event
# called LAST, after handling). The pynput on_press/on_release path is the
# non-Windows fallback. Trigger key comes from .env (JARVIS_HOTKEY): if
# PrintScreen still triggers screenshots, set JARVIS_HOTKEY=f9.
def on_press(key):
    if key == HOTKEY:
        start_recording()


def on_release(key):
    if key == QUIT_KEY:
        return False
    if key == HOTKEY:
        stop_recording()


def win32_filter(msg, data):
    if data.vkCode == HOTKEY_VK:
        if msg in (0x100, 0x104):
            start_recording()
        elif msg in (0x101, 0x105):
            stop_recording()
        listener.suppress_event()
    return True


listener_kwargs = {}
if sys.platform == "win32":
    listener_kwargs["win32_event_filter"] = win32_filter

listener = keyboard.Listener(on_press=on_press, on_release=on_release, **listener_kwargs)

stream = None
INPUT_RATE = config.SAMPLE_RATE
INPUT_CHANNELS = 1
MIC_IN_USE = config.MIC_DEVICE
try:
    _dev_info = sd.query_devices(config.MIC_DEVICE)
except Exception:
    print(f"[warn] configured mic {config.MIC_DEVICE!r} not found; "
          f"falling back to the Windows default input. "
          f"Run --list-mics to re-pin JARVIS_MIC.")
    MIC_IN_USE = None
    try:
        _dev_info = sd.query_devices(kind="input")  # OS default input
    except Exception as e2:
        sys.exit(f"No usable input device at all ({e2}). "
                 f"Run: python push_to_talk.py --list-mics")
try:
    _native = int(_dev_info.get("default_samplerate") or config.SAMPLE_RATE)
    if _native != config.SAMPLE_RATE:
        INPUT_RATE = _native
    print(f"[mic] {_dev_info['name']} @ {INPUT_RATE}Hz"
          + (f"; resampling to {config.SAMPLE_RATE}Hz for STT."
             if INPUT_RATE != config.SAMPLE_RATE else "."))
except Exception as e:
    print(f"[warn] could not probe mic rate ({e}); trying "
          f"{config.SAMPLE_RATE}Hz.")

for _ch in (1, int(_dev_info.get("max_input_channels") or 1)):
    try:
        stream = sd.InputStream(
            samplerate=INPUT_RATE,
            device=MIC_IN_USE,
            channels=_ch,
            dtype="float32",
            callback=audio_callback,
        )
        INPUT_CHANNELS = _ch
        if _ch > 1:
            print(f"[mic] mono rejected, opened {_ch}ch; downmixing.")
        break
    except Exception as e:
        print(f"[warn] {_ch}ch open failed ({str(e)[:100]}); retrying…")
if stream is None:
    sys.exit("Could not open the mic in any channel mode. "
             "Run: python mic_test.py --all")

listener.start()
print(f"Ready. Hold {HOTKEY_NAME} to talk{' (speech on)' if SPEAK else ' (speech off)'}. "
      "Esc or Ctrl+C to quit.")
print(f"[keys] gemini={'set' if os.environ.get('GEMINI_API_KEY') else 'MISSING'} "
      f"openrouter={'set' if os.environ.get('OPENROUTER_API_KEY') else 'MISSING'} "
      f"(.env={'found' if config.DOTENV_FOUND else 'missing'})")
print(f"[models] base={config.LLM_MODEL} stepup=laguna "
      f"fast={config.OPENROUTER_FAST} big={config.OPENROUTER_BIG} "
      f"tts={TTS_ENGINE} "
      f"tools=local:{config.ENABLE_LOCAL_TOOLS} "
      f"search:{config.ENABLE_SEARCH_GROUNDING} "
      f"(quota {tools.search_quota_used()}/{config.SEARCH_DAILY_LIMIT})")

try:
    with stream:
        while listener.is_alive():
            time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    listener.stop()
    stop_speaking()
    print("Bye.")
    # Native teardown (CUDA/cuBLAS DLL unload vs PortAudio/pynput threads)
    # segfaults on Windows *after* Bye prints. All state is append-only
    # files flushed per write, so a hard exit here is safe and silent.
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(0)

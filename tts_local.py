"""Local TTS sink for Jarvis. LLM produces text; this module speaks it.

No Gemini/cloud TTS here by design. Backends (first available wins
in "auto" mode):

  edge   — `pip install edge-tts` (free, best quality, needs net).
           Voice e.g. en-US-AriaNeural. MP3 decoded via `av` (already
           installed with faster-whisper).
  kokoro — `pip install kokoro-onnx soundfile` + model files
           (offline, Apache-2.0). See TTS.md for download paths.
  sapi   — `pip install pyttsx3` (Windows built-in voices, zero DL).

Interface:
  synthesize(text, engine="auto", voice=None) -> (int16 mono, rate) | None
    None means the engine spoke directly (sapi) — nothing to queue.
  speak_text(text, ...) — blocking playback via sounddevice (or sapi).

Also a pipe consumer: `python tts_local.py --stdin` reads text lines
from stdin and speaks them, so any producer can pipe in:
  python push_to_talk.py --print-reply | python tts_local.py --stdin
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import tempfile
from pathlib import Path

import config

_ABBR = {
    "dr.": "doctor", "mr.": "mister", "mrs.": "missus", "ms.": "miss",
    "st.": "saint", "vs.": "versus", "etc.": "etcetera",
    "e.g.": "for example", "i.e.": "that is",
}


def clean_for_speech(text: str) -> str:
    """Normalize LLM output for spoken delivery."""
    text = re.sub(r"```.*?```", " code omitted. ", text, flags=re.S)
    text = re.sub(r"[*_`#>~|]", "", text)
    text = re.sub(r"https?://\S+", " link. ", text)
    for k, v in _ABBR.items():
        text = re.sub(re.escape(k), v, text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str,
               min_chars: int | None = None,
               max_chars: int | None = None) -> list[str]:
    """Split into TTS-friendly chunks (default 140–400 chars).

    Merges short sentences so prosody stays natural (the old 40-char
    chunks were the main 'bad TTS' cause besides the engine itself).
    """
    min_chars = min_chars or config.MIN_CHARS_PER_CHUNK
    max_chars = max_chars or config.MAX_CHARS_PER_CHUNK
    text = clean_for_speech(text)
    if not text:
        return []
    sentences = [s for s in SENT_SPLIT.split(text) if s]
    chunks, pending = [], ""
    for s in sentences:
        if len(s) > max_chars:  # hard split very long sentences on commas
            if pending:
                chunks.append(pending)
                pending = ""
            parts = re.split(r"(?<=,)\s+", s)
            buf = ""
            for p in parts:
                buf = f"{buf} {p}".strip()
                if len(buf) >= min_chars:
                    chunks.append(buf)
                    buf = ""
            if buf:
                pending = buf
            continue
        candidate = f"{pending} {s}".strip()
        if len(candidate) >= max_chars and pending:
            chunks.append(pending)
            pending = s
        elif len(candidate) >= min_chars:
            chunks.append(candidate)
            pending = ""
        else:
            pending = candidate
    if pending:
        chunks.append(pending)
    return chunks


def available_engines() -> dict[str, bool]:
    out = {}
    try:
        import edge_tts  # noqa: F401
        out["edge"] = True
    except ImportError:
        out["edge"] = False
    try:
        import kokoro_onnx  # noqa: F401
        out["kokoro"] = True
    except ImportError:
        out["kokoro"] = False
    try:
        import pyttsx3  # noqa: F401
        out["sapi"] = True
    except ImportError:
        # winsound-free fallback: SAPI via PowerShell still possible,
        # but report False so auto skips unless forced.
        out["sapi"] = False
    return out


def resolve_engine(requested: str | None) -> str | None:
    req = (requested or config.TTS_ENGINE).lower()
    avail = available_engines()
    if req != "auto":
        return req if req != "none" else None
    for eng in ("edge", "kokoro", "sapi"):
        if avail.get(eng):
            return eng
    return None


# --- backends ---------------------------------------------------------------
def _synth_edge(text: str, voice: str | None):
    import asyncio
    import edge_tts

    voice = voice or config.TTS_VOICE_EDGE

    async def _fetch() -> bytes:
        comm = edge_tts.Communicate(text, voice)
        buf = io.BytesIO()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    mp3 = asyncio.run(_fetch())
    if not mp3:
        raise RuntimeError("edge-tts returned no audio")
    return _mp3_to_pcm(mp3)


def _mp3_to_pcm(mp3: bytes):
    """Decode MP3 bytes to (int16 mono, rate) using `av` (installed)."""
    import av
    import numpy as np

    container = av.open(io.BytesIO(mp3))
    stream = next(s for s in container.streams if s.type == "audio")
    samples, rate = [], stream.rate or 24000
    for frame in container.decode(stream):
        arr = frame.to_ndarray()  # (channels, samples) float32 planar
        if arr.ndim == 2 and arr.shape[0] > 1:
            arr = arr.mean(axis=0, keepdims=True)
        samples.append(arr.reshape(-1))
    container.close()
    if not samples:
        raise RuntimeError("could not decode edge-tts MP3")
    import numpy as _np

    pcm = _np.concatenate(samples)
    if pcm.dtype != _np.float32:
        pcm = pcm.astype(_np.float32)
    pcm = _np.clip(pcm, -1.0, 1.0)
    return (pcm * 32767).astype(_np.int16), int(rate)


_KOKORO = None


def _get_kokoro():
    global _KOKORO
    if _KOKORO is not None:
        return _KOKORO
    from kokoro_onnx import Kokoro

    model = Path(os.environ.get("KOKORO_MODEL_PATH", "kokoro-v1.0.onnx"))
    voices = Path(os.environ.get("KOKORO_VOICES_PATH", "voices-v1.0.bin"))
    search = [model, voices, Path.home() / "kokoro" / model.name,
              Path.home() / "kokoro" / voices.name, Path("kokoro") / model.name]
    m = next((p for p in search if p.exists()), None)
    v = next((p for p in search if p.exists() and p.suffix == ".bin"), None)
    if m is None or v is None:
        raise RuntimeError(
            "Kokoro model files not found. See TTS.md: download "
            "kokoro-v1.0.onnx + voices-v1.0.bin and set KOKORO_MODEL_PATH / "
            "KOKORO_VOICES_PATH."
        )
    _KOKORO = Kokoro(str(m), str(v))
    return _KOKORO


def _synth_kokoro(text: str, voice: str | None):
    import numpy as np

    kokoro = _get_kokoro()
    voice = voice or config.TTS_VOICE_KOKORO
    audio, rate = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16), int(rate)


def _speak_sapi(text: str, voice: str | None = None):
    try:
        import pyttsx3
    except ImportError:
        raise RuntimeError("pip install pyttsx3 for the sapi engine")
    eng = pyttsx3.init()
    if config.TTS_RATE_WPM:
        eng.setProperty("rate", config.TTS_RATE_WPM)
    eng.say(text)
    eng.runAndWait()


# --- public API ---------------------------------------------------------------
def synthesize(text: str, engine: str | None = None,
               voice: str | None = None):
    """Return (int16 mono PCM, sample_rate), or None if engine speaks
    directly (sapi). Raises RuntimeError with install hint on failure."""
    import os  # local to keep module import light

    eng = resolve_engine(engine)
    if eng is None:
        return None
    text = clean_for_speech(text)
    if not text:
        return None
    if eng == "edge":
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            raise RuntimeError("pip install edge-tts for the edge engine")
        return _synth_edge(text, voice)
    if eng == "kokoro":
        try:
            import kokoro_onnx  # noqa: F401
        except ImportError:
            raise RuntimeError("pip install kokoro-onnx soundfile (see TTS.md)")
        return _synth_kokoro(text, voice)
    if eng == "sapi":
        _speak_sapi(text, voice)
        return None
    raise ValueError(f"unknown TTS engine: {eng}")


def speak_text(text: str, engine: str | None = None,
               voice: str | None = None) -> bool:
    """Blocking speak of one chunk. True if audio was played."""
    eng = resolve_engine(engine)
    if eng is None:
        print(f"[tts muted] {text}")
        return False
    if eng == "sapi":
        _speak_sapi(clean_for_speech(text), voice)
        return True
    result = synthesize(text, engine=eng, voice=voice)
    if result is None:
        return False
    audio, rate = result
    import sounddevice as sd

    sd.play(audio, rate)
    sd.wait()
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Local TTS sink (pipe consumer)")
    ap.add_argument("--stdin", action="store_true",
                    help="Read text lines from stdin and speak each")
    ap.add_argument("--text", default=None, help="Speak this text once")
    ap.add_argument("--save", default=None,
                    help="Save stdin/text audio to WAV file instead of playing")
    ap.add_argument("--engine", default=None,
                    help="edge|kokoro|sapi|auto (default from config)")
    ap.add_argument("--voice", default=None, help="Voice id for engine")
    ap.add_argument("--list-engines", action="store_true")
    args = ap.parse_args(argv)

    if args.list_engines:
        avail = available_engines()
        for k, v in avail.items():
            print(f"{k}: {'installed' if v else 'missing'}")
        print(f"resolved(auto): {resolve_engine('auto')}")
        return 0

    lines: list[str] = []
    if args.text:
        lines = [args.text]
    elif args.stdin or (not sys.stdin.isatty()):
        lines = [ln.strip() for ln in sys.stdin if ln.strip()]
    else:
        ap.print_help()
        return 2

    if args.save:
        import wave
        import numpy as np

        pcm_parts, rate = [], None
        for ln in lines:
            for ch in chunk_text(ln):
                res = synthesize(ch, engine=args.engine, voice=args.voice)
                if res is None:
                    print("[warn] sapi engine can't --save; skipping")
                    continue
                pcm, rate = res
                pcm_parts.append(pcm)
        if not pcm_parts:
            print("nothing to save")
            return 1
        audio = np.concatenate(pcm_parts)
        with wave.open(args.save, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate or 24000)
            w.writeframes(audio.tobytes())
        print(f"saved {args.save}")
        return 0

    for ln in lines:
        for ch in chunk_text(ln):
            try:
                speak_text(ch, engine=args.engine, voice=args.voice)
            except Exception as e:
                print(f"[error] TTS failed: {e}")
    return 0


if __name__ == "__main__":
    import os

    raise SystemExit(main())

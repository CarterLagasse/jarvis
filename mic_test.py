"""Mic diagnostics: record a few seconds per candidate and report levels.

Usage:
  python mic_test.py                  # test the configured JARVIS_MIC
  python mic_test.py --all            # test every input device (short)
  python mic_test.py --device NAME    # test one device by index or name
  python mic_test.py --seconds 5       # longer sample

Speak normally while it records. RMS < ~0.005 => silence/dead device.
Saves mic_test.wav (last device tested) for listening back.
"""
from __future__ import annotations

import argparse
import os
import site
import sys
import time
import wave

# Same CUDA DLL preload as push_to_talk.py (must run before faster_whisper).
if sys.platform == "win32":
    from pathlib import Path
    _roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    for _sp in _roots:
        for _sub in ("cublas", "cudnn"):
            _d = Path(_sp) / "nvidia" / _sub / "bin"
            if _d.exists():
                os.add_dll_directory(str(_d))
                os.environ["PATH"] = str(_d) + os.pathsep + os.environ["PATH"]

import numpy as np

import config


def record(device, seconds: int, rate: int):
    import sounddevice as sd
    frames: list = []

    def cb(indata, nframes, time_info, status):
        if status:
            print(f"  [stream status] {status}")
        frames.append(indata.copy())

    print(f"Recording {seconds}s from {device!r} @ {rate}Hz — speak now…")
    with sd.InputStream(samplerate=rate, channels=1, dtype="float32",
                        device=device, callback=cb):
        time.sleep(seconds)
    audio = np.concatenate(frames).flatten().astype(np.float32)
    return audio


def report(name: str, audio: np.ndarray, rate: int):
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    verdict = "OK — signal present" if rms >= 0.005 else "SILENT — no usable signal"
    print(f"[{name}] rate={rate} samples={len(audio)} "
          f"rms={rms:.5f} peak={peak:.4f} -> {verdict}")
    return rms >= 0.005


def save_wav(path: str, audio: np.ndarray, rate: int):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    print(f"saved {path} (listen back to confirm)")


def candidates():
    import sounddevice as sd
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append((i, d["name"], int(d.get("default_samplerate") or 48000)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mic signal diagnostics")
    ap.add_argument("--all", action="store_true", help="test every input")
    ap.add_argument("--device", default=None, help="index or name fragment")
    ap.add_argument("--seconds", type=int, default=3)
    ap.add_argument("--transcribe", action="store_true",
                    help="resample to 16k (app path) and run Whisper on it")
    args = ap.parse_args(argv)

    import sounddevice as sd
    if args.device is not None:
        dev = int(args.device) if str(args.device).isdigit() else args.device
        info = sd.query_devices(dev)
        rate = int(info.get("default_samplerate") or 48000)
        audio = record(dev, args.seconds, rate)
        ok = report(str(dev), audio, rate)
        if args.transcribe:
            # Same math as push_to_talk._resample_mono (kept inline: importing
            # push_to_talk would launch the whole assistant).
            import numpy as _np
            audio16 = _np.interp(
                _np.linspace(0, len(audio) - 1, max(1, int(len(audio) * 16000 / rate))),
                _np.linspace(0, len(audio) - 1, len(audio)),
                audio).astype(_np.float32)
            print(f"resampled {len(audio)} -> {len(audio16)} samples")
            from faster_whisper import WhisperModel
            m = WhisperModel("base.en", device="cuda", compute_type="float16")
            segs, _ = m.transcribe(audio16, language="en", beam_size=1,
                                   vad_filter=False,
                                   condition_on_previous_text=False,
                                   without_timestamps=True)
            print("TRANSCRIPT:", " ".join(s.text for s in segs) or "<empty>")
        else:
            save_wav("mic_test.wav", audio, rate)
        return 0 if ok else 1

    if args.all:
        results = []
        for i, name, rate in candidates():
            if "Mapper" in name or "Primary Sound Capture" in name:
                continue
            try:
                audio = record(i, 2, rate)
                ok = report(f"{i} {name}", audio, rate)
                results.append((ok, i, name))
            except Exception as e:
                print(f"[{i} {name}] FAILED: {str(e)[:150]}")
                results.append((False, i, name))
        print("\n--- summary (speak during EACH sample; quiet room => retest) ---")
        for ok, i, name in results:
            print(("PASS " if ok else "fail "), i, name)
        return 0

    dev = config.MIC_DEVICE
    info = sd.query_devices(dev)
    rate = int(info.get("default_samplerate") or 48000)
    print(f"Configured mic: {config.MIC_DEVICE!r} -> [{info['index']}] {info['name']}")
    audio = record(dev, args.seconds, rate)
    ok = report("configured", audio, rate)
    save_wav("mic_test.wav", audio, rate)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

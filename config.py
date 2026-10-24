"""Central config for Jarvis push-to-talk assistant.

One place for model IDs, free-tier endpoints, TTS engine choice,
and hotkey behaviour. Env vars override defaults; CLI flags override env.
"""
from __future__ import annotations

import argparse
import os


_HERE = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(_HERE, ".env")
DOTENV_FOUND = False


def _load_dotenv(path: str | None = None):
    """Load KEY=value lines into os.environ (stdlib only, no dependency).

    Looks next to config.py (NOT the current working directory), so the app
    finds it no matter which folder you launch from. Existing process env
    wins — a real exported var always overrides .env. Lines starting with
    # and blank lines are ignored. Values may be single/double quoted.
    """
    global DOTENV_FOUND
    path = path or DOTENV_PATH
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                    v = v[1:-1]
                if k and k not in os.environ:
                    os.environ[k] = v
        DOTENV_FOUND = True
    except OSError:
        pass  # no .env yet — see .env.example


_load_dotenv()

# Common alias names some tools use — accept them as fallback.
if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

# ---------------------------------------------------------------------------
# LLM: base model + free-only escalation pool
# ---------------------------------------------------------------------------
# Base model: fast, free-tier, handles all default traffic.
LLM_MODEL = os.environ.get("JARVIS_LLM_MODEL", "gemini-3.5-flash-lite")

# Step-up inside Gemini free tier (same API key, same base URL).
ESCALATION_MODEL_FLASH = os.environ.get(
    "JARVIS_FLASH_MODEL", "gemini-3.5-flash"
)
# Retired: gemini-2.5-pro returns 404 for new keys (use paid 3.1-pro-preview).
# Kept as a name only so old .env files don't break; nothing routes to it.
ESCALATION_MODEL_PRO = os.environ.get(
    "JARVIS_PRO_MODEL", "gemini-2.5-pro"
)

GEMINI_BASE_URL = os.environ.get(
    "JARVIS_GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)

# OpenRouter (OpenAI-compatible): free Laguna coding model. Key accepts
# OPENROUTER_API_KEY or OPEN_ROUTER_API_KEY. Free variant id ends in :free.
OPENROUTER_BASE_URL = os.environ.get(
    "JARVIS_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL = os.environ.get(
    "JARVIS_OPENROUTER_MODEL", "poolside/laguna-s-2.1:free"
)
# Backup chain models (free :free variants, same key).
OPENROUTER_FAST = os.environ.get(
    "JARVIS_OPENROUTER_FAST", "nvidia/nemotron-3.5-lightning:free"
)
OPENROUTER_BIG = os.environ.get(
    "JARVIS_OPENROUTER_BIG", "nvidia/nemotron-3-ultra-550b-a55b:free"
)
# Optional but recommended by OpenRouter (ranking/attribution header).
OPENROUTER_SITE = os.environ.get("JARVIS_OPENROUTER_SITE", "")
if "OPENROUTER_API_KEY" not in os.environ and "OPEN_ROUTER_API_KEY" in os.environ:
    os.environ["OPENROUTER_API_KEY"] = os.environ["OPEN_ROUTER_API_KEY"]

SYSTEM_PROMPT = (
    "You are Jarvis, a push-to-talk voice assistant. "
    "Replies are read aloud by a separate local TTS engine, so: "
    "answer in short, natural spoken sentences. "
    "No markdown, bullet points, code blocks, or emojis unless the user "
    "explicitly asks for code. "
    "Reply with ONLY the final answer — never show thinking, reasoning, "
    "or process. "
    "Keep routine answers under 60 words. "
    "If the request needs code, deep reasoning, or current facts, say so "
    "briefly at the start (e.g. 'Let me work through this...') — "
    "the router may escalate you to a stronger model."
)

MAX_HISTORY_MESSAGES = int(os.environ.get("JARVIS_MAX_HISTORY", "20"))

# Heuristic escalation thresholds (see router.py / ROUTING.md).
ESCALATE_KEYWORDS = tuple(
    os.environ.get(
        "JARVIS_ESCALATE_KEYWORDS",
        "debug,refactor,prove,proof,derive,step by step,think harder,"
        "write a function,write code,algorithm,regex,sql,stack trace,"
        "research,compare,analyze,essay,plan my,legal,medical,tax",
    ).split(",")
)
ESCALATE_MIN_CHARS = int(os.environ.get("JARVIS_ESCALATE_MIN_CHARS", "220"))
ESCALATE_HISTORY_DEPTH = int(os.environ.get("JARVIS_ESCALATE_DEPTH", "6"))

# ---------------------------------------------------------------------------
# Tools: local commands + Google Search grounding (see TOOLS.md)
# ---------------------------------------------------------------------------
ENABLE_LOCAL_TOOLS = os.environ.get("JARVIS_TOOLS", "1") == "1"
ENABLE_SEARCH_GROUNDING = os.environ.get("JARVIS_SEARCH", "1") == "1"
# Keyless direct APIs (DDG/Wikipedia, Open-Meteo, ESPN). No keys, no quotas.
ENABLE_WEBTOOLS = os.environ.get("JARVIS_WEB", "1") == "1"

# Recency/factual triggers -> grounded Gemini path (google_search tool).
SEARCH_TRIGGERS = tuple(
    os.environ.get(
        "JARVIS_SEARCH_TRIGGERS",
        "today,latest,latest news,news,breaking,who won,score,price of,"
        "weather,forecast,release date,announced,this week,currently,"
        "right now,what happened,is it true,stock,exchange rate,election,"
        "search for,search up,look up,google",
    ).split(",")
)
# Free search quota guard (shared Flash/Flash-Lite free allowance).
SEARCH_DAILY_LIMIT = int(os.environ.get("JARVIS_SEARCH_LIMIT", "400"))
SEARCH_QUOTA_FILE = os.environ.get(
    "JARVIS_SEARCH_QUOTA", ".search_quota.json"
)
NOTES_FILE = os.environ.get("JARVIS_NOTES", "notes.txt")
# Default location for weather (and later: local scores/news). Used when the
# transcript names no city, before falling back to IP geolocation.
# Examples: "Chicago", "New Haven, CT". Set in .env as JARVIS_LOCATION=...
DEFAULT_LOCATION = os.environ.get("JARVIS_LOCATION", "").strip()

# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
# Default mic — set in .env (JARVIS_MIC). Accepts a device index ("1") or a
# name fragment ("Yeti"). Unset/blank = Windows default input.
# Find yours with: python push_to_talk.py --list-mics
_MIC_RAW = (os.environ.get("JARVIS_MIC", "") or "").strip()
MIC_DEVICE = int(_MIC_RAW) if _MIC_RAW.isdigit() else (_MIC_RAW or None)
WHISPER_MODEL_SIZE = os.environ.get("JARVIS_WHISPER_MODEL", "base.en")
WHISPER_BEAM = int(os.environ.get("JARVIS_WHISPER_BEAM", "3"))  # 1=fastest, 3-5=accurate
USE_GPU = os.environ.get("JARVIS_USE_GPU", "1") == "1"

# ---------------------------------------------------------------------------
# TTS sink — LLM only produces text; tts_local.py speaks it.
# ---------------------------------------------------------------------------
# Engine order of preference. First available installed engine wins unless
# overridden by --tts-engine. "auto" tries edge -> kokoro -> sapi.
TTS_ENGINE = os.environ.get("JARVIS_TTS_ENGINE", "auto")
TTS_VOICE_EDGE = os.environ.get("JARVIS_TTS_VOICE_EDGE", "en-US-AriaNeural")
TTS_VOICE_KOKORO = os.environ.get("JARVIS_TTS_VOICE_KOKORO", "af_heart")
TTS_RATE_WPM = int(os.environ.get("JARVIS_TTS_WPM", "0"))  # 0 = engine default
# Speech chunking: merge short sentences so TTS keeps natural prosody.
MIN_CHARS_PER_CHUNK = int(os.environ.get("JARVIS_TTS_CHUNK", "140"))
MAX_CHARS_PER_CHUNK = int(os.environ.get("JARVIS_TTS_MAX_CHUNK", "400"))
# Fast-start: the FIRST chunk of a turn speaks at this smaller size so audio
# starts ~1-2 LLM-seconds sooner; later chunks use the full size above.
FIRST_CHUNK_CHARS = int(os.environ.get("JARVIS_TTS_FIRST", "60"))

# Reply sink tee: always appends assistant replies to this file so an
# external TTS (or anything else) can `tail -f` it. Also printed to stdout
# with --print-reply for `| python tts_local.py --stdin` piping.
REPLY_LOG = os.environ.get("JARVIS_REPLY_LOG", "reply.txt")

# ---------------------------------------------------------------------------
# Hotkey (trigger key) — .env only, no CLI flag.
# ---------------------------------------------------------------------------
# printscreen (VK 0x2C) is swallowed system-wide on Windows so no screenshot
# app fires. f9 is the alternative if PrintScreen misbehaves.
# Set in .env: JARVIS_HOTKEY=printscreen  (or f9)
HOTKEY_NAME = os.environ.get("JARVIS_HOTKEY", "printscreen").lower()
if HOTKEY_NAME not in ("printscreen", "f9"):
    raise ValueError(
        f"JARVIS_HOTKEY={HOTKEY_NAME!r} invalid: use printscreen or f9")
HOTKEY_VK_MAP = {"printscreen": 0x2C, "f9": 0x78}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Jarvis push-to-talk assistant")
    p.add_argument("--no-speech", action="store_true",
                   help="Text replies only, no TTS playback")
    p.add_argument("--print-reply", action="store_true",
                   help="Also print final reply to stdout for piping")
    p.add_argument("--tts-engine", default=TTS_ENGINE,
                   choices=["auto", "edge", "kokoro", "sapi", "none"],
                   help="TTS backend (default from JARVIS_TTS_ENGINE)")
    p.add_argument("--tts-stdin", action="store_true",
                   help="Instead of mic loop, read text lines from stdin "
                        "and speak them (pipe consumer mode)")
    p.add_argument("--list-mics", action="store_true",
                   help="List audio devices and exit")
    return p

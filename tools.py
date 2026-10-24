"""Jarvis abilities: local tools + Google Search grounding.

Two tiers:

1. Local tools — no API, instant: clock/date, note-to-file, open URL.
   run_local_tool() executes them directly; the router never calls a model.
2. Grounded answers — Gemini `google_search` tool via the google-genai SDK
   (NOT the OpenAI-compatible endpoint, which can't pass Google-native
   tools). Quota-guarded, fail-closed to ungrounded base knowledge.

Every function returns SHORT spoken text. Detail/sources go to the
console and reply.txt only — never to TTS.
"""
from __future__ import annotations

import datetime
import json
import re
import webbrowser
from pathlib import Path

import config

_URL_RE = re.compile(r"(https?://[^\s]+|(?:[\w-]+\.)+(?:com|org|net|io|dev|ai|edu|gov)[^\s]*)", re.I)


# ---------------------------------------------------------------------------
# Spoken summaries
# ---------------------------------------------------------------------------
def summarize_for_voice(text: str, max_words: int = 60) -> str:
    """Truncate to whole sentences within max_words for TTS."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    cut = " ".join(words[:max_words])
    # Prefer ending at the last sentence boundary inside the cut.
    m = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if m > len(cut) * 0.4:
        return cut[:m + 1].strip()
    return cut.rstrip(",;:") + "…"


def split_spoken_sources(reply: str) -> tuple[str, str]:
    """Split a grounded reply into (spoken part, sources part).

    The grounding prompt asks the model to put `SOURCES:` last; anything
    after that marker (or bare URLs) is console-only.
    """
    m = re.search(r"\bSOURCES\s*:(.*)$", reply, re.S | re.I)
    if m:
        return reply[:m.start()].strip(), m.group(1).strip()
    urls = _URL_RE.findall(reply)
    spoken = _URL_RE.sub(" ", reply)
    spoken = re.sub(r"\s+", " ", spoken).strip()
    return spoken, " ".join(urls)


# ---------------------------------------------------------------------------
# Local tools
# ---------------------------------------------------------------------------
def tool_clock() -> str:
    now = datetime.datetime.now()
    hour = now.hour % 12 or 12
    day = now.strftime("%A, %B %d")
    return f"It's {hour}:{now.minute:02d} {now.strftime('%p')}, {day}."


def tool_note(text: str) -> str:
    """Save a dictated note. `text` is the full transcript."""
    m = re.match(r"^(note that|remember that|take a note:?)\s*(.*)$",
                 text.strip(), re.I)
    body = (m.group(2) if m else text).strip()
    if not body:
        return "What should I note down?"
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        with open(config.NOTES_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {body}\n")
    except OSError as e:
        return f"I couldn't save that note: {e}"
    return "Noted."


# Bare site names ("open youtube") -> full URLs. Extend freely.
KNOWN_SITES = {
    "youtube": "https://www.youtube.com",
    "youtube music": "https://music.youtube.com",
    "yt music": "https://music.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "maps": "https://maps.google.com",
    "drive": "https://drive.google.com",
    "calendar": "https://calendar.google.com",
    "docs": "https://docs.google.com",
    "github": "https://github.com",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
    "reddit": "https://www.reddit.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "tiktok": "https://www.tiktok.com",
    "netflix": "https://www.netflix.com",
    "twitch": "https://www.twitch.com",
    "spotify": "https://open.spotify.com",
    "amazon": "https://www.amazon.com",
    "wikipedia": "https://www.wikipedia.org",
    "yahoo": "https://www.yahoo.com",
    "bing": "https://www.bing.com",
    "outlook": "https://outlook.live.com",
    "chatgpt": "https://chat.openai.com",
    "chat gpt": "https://chat.openai.com",
    "discord": "https://discord.com",
}


def resolve_open_target(text: str) -> str | None:
    """Turn an "open ..." transcript into a URL. Pure function (no browser).

    Order: explicit URL in transcript -> known site name -> bare word gets
    .com -> anything else becomes a Google search. None only when there is
    nothing after "open".
    """
    remainder = re.sub(r"^open\b\s*", "", text.strip(), flags=re.I)
    remainder = remainder.strip().rstrip(".!?,").strip()
    if not remainder:
        return None
    m = _URL_RE.search(remainder)
    if m:
        url = m.group(1).rstrip(".!?,")
        return url if url.lower().startswith("http") else "https://" + url
    low = remainder.lower()
    if low in KNOWN_SITES:
        return KNOWN_SITES[low]
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", low):
        return f"https://{low}.com"
    from urllib.parse import quote_plus
    return f"https://www.google.com/search?q={quote_plus(remainder)}"


def tool_open_url(text: str) -> str:
    url = resolve_open_target(text)
    if not url:
        return "Which site should I open?"
    try:
        webbrowser.open(url)
    except Exception as e:
        return f"I couldn't open that: {e}"
    host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0].lower()
    if host == "google.com" and "/search" in url:
        return "I searched Google for that."
    return f"Opening {host}."


def run_local_tool(name: str, text: str) -> str:
    """Execute a local tool; never raises (returns spoken error instead)."""
    try:
        if name == "clock":
            return tool_clock()
        if name == "note":
            return tool_note(text)
        if name == "open_url":
            return tool_open_url(text)
        return "I don't know that tool yet."
    except Exception as e:
        return f"That tool failed: {e}"


# ---------------------------------------------------------------------------
# Search quota guard (persisted per-day counter)
# ---------------------------------------------------------------------------
def _quota_path() -> Path:
    return Path(config.SEARCH_QUOTA_FILE)


def search_quota_used() -> int:
    try:
        data = json.loads(_quota_path().read_text("utf-8"))
        if data.get("date") == datetime.date.today().isoformat():
            return int(data.get("count", 0))
    except Exception:
        pass
    return 0


def search_allowed() -> bool:
    return search_quota_used() < config.SEARCH_DAILY_LIMIT


def _quota_bump():
    today = datetime.date.today().isoformat()
    count = search_quota_used() + 1
    # If the file held yesterday, search_quota_used() returned 0 -> reset.
    try:
        _quota_path().write_text(
            json.dumps({"date": today, "count": count}), "utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Grounded answers via google-genai SDK
# ---------------------------------------------------------------------------
_GROUND_SYSTEM = (
    "You are Jarvis, a voice assistant. Answer the user's question using "
    "the search results. First write at most TWO short spoken sentences "
    "suitable for text-to-speech (no markdown, no URLs, under 60 words). "
    "Then on a new line write SOURCES: followed by the source URLs, "
    "one per line."
)


def grounded_answer(prompt: str) -> tuple[str, str]:
    """Ask Gemini with the google_search tool.

    Returns (spoken_text, console_detail). Never raises: on any failure
    (no key, quota out, API error) returns ("", reason) so the caller can
    fail closed to the normal ungrounded path.
    """
    import os

    if not config.ENABLE_SEARCH_GROUNDING:
        return "", "search grounding disabled (JARVIS_SEARCH=0)"
    if not os.environ.get("GEMINI_API_KEY"):
        return "", "no GEMINI_API_KEY"
    if not search_allowed():
        return "", (f"daily search quota reached "
                    f"({config.SEARCH_DAILY_LIMIT}); answering from base knowledge")
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "", "google-genai not installed (pip install google-genai)"

    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        # NOTE: google_search runs through automatic function calling, which
        # the SDK supports on the Chat path (generate_content AFC is
        # deprecated upstream).
        chat = client.chats.create(
            model=config.LLM_MODEL,  # flash-lite supports google_search
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                system_instruction=_GROUND_SYSTEM,
            ),
        )
        resp = chat.send_message(prompt)
        _quota_bump()
        raw = (resp.text or "").strip()
        if not raw:
            return "", "grounded call returned empty text"
        spoken, sources = split_spoken_sources(raw)
        spoken = summarize_for_voice(spoken)
        detail = raw  # full text + sources go to console/reply.txt
        if sources:
            detail = f"{spoken}\nSOURCES: {sources}"
        return spoken, detail
    except Exception as e:
        msg = str(e)
        if "429" in msg:
            return "", "search quota exhausted (429) — resets daily"
        return "", f"grounded call failed: {msg[:150]}"

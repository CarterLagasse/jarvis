"""Gemini-first model router for Jarvis.

  gemini     — GEMINI_API_KEY: base gemini-3.5-flash-lite (+ flash/pro)
  openrouter — OPENROUTER_API_KEY (or OPEN_ROUTER_API_KEY): free Laguna
               coding model, explicit trigger + 429 shed

This module decides — WITHOUT an extra LLM call (saves latency + free
quota) — when a transcript should step up, use a local tool, or take the
Google Search grounded path. Full policy: classify() and fallbacks_for()
below; routing overview in the README.
Pure heuristics + client factory, fail-closed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import config


@dataclass
class RouteDecision:
    route: str  # "direct" | "stepup" | "think" | "ultra"
    model: str  # concrete model id to call
    provider: str  # "gemini" | "openrouter"
    reason: str


DIRECT = RouteDecision(
    route="direct",
    model=config.LLM_MODEL,
    provider="gemini",
    reason="default: short conversational turn",
)


def _norm(text: str) -> str:
    return text.lower().strip()


# ---------------------------------------------------------------------------
# Tool routing. Checked BEFORE model routing in push_to_talk.
# Order: local (instant) -> weather/sports (keyless APIs) -> web (DDG+Lite)
# with Gemini grounding as last resort inside handle_text.
# Returns (kind, detail) or (None, None).
# ---------------------------------------------------------------------------
SPORTS_WORDS = ("score", "scores", "scored", "scoring", "win", "wins",
                "won", "beat", "beaten", "game", "match", "final",
                "inning", "touchdown", "halftime", "playoff", "world series",
                "super bowl", "stanley cup", " finals", "championship")

WEATHER_WORDS = ("weather", "temperature outside", "forecast", "raining",
                 "rain", "snow", "snowing", "snowy", "storm", "storms",
                 "humidity", "how hot", "how cold", "degrees outside",
                 "umbrella", "jacket")

NEWS_HEADLINES = ("headlines", "top news", "breaking news", "breaking",
                  "today's news", "this morning", "top stories")


def route_tools(text: str) -> tuple[str | None, str | None]:
    """Decide whether `text` is a local-tool command or needs grounding."""
    if (not config.ENABLE_LOCAL_TOOLS and not config.ENABLE_SEARCH_GROUNDING
            and not config.ENABLE_WEBTOOLS):
        return None, None
    try:
        t = _norm(text)
        if not t:
            return None, None

        if config.ENABLE_LOCAL_TOOLS:
            if any(k in t for k in (
                "what time", "what's the time", "current time",
                "what day", "what's today", "today's date", "current date",
            )):
                return "local", "clock"
            if t.startswith("note that ") or t.startswith("remember that ") \
                    or t.startswith("take a note"):
                return "local", "note"
            if t.startswith("open ") and t[5:].strip():
                return "local", "open_url"

        # Dedicated keyless paths beat the generic web bucket: weather words
        # and "score" live in both lists, and Open-Meteo/ESPN answer better.
        if config.ENABLE_WEBTOOLS:
            if t.startswith("play ") and t[5:].strip():
                return "play", ""
            if any(k in t for k in WEATHER_WORDS):
                return "weather", ""
            if any(k in t for k in SPORTS_WORDS):
                import webtools
                if webtools.detect_teams(t):
                    return "sports", ""
                # sportsy but no team ("who won the superbowl") -> web search
                return "web", "sports"
            if any(k in t for k in NEWS_HEADLINES):
                return "news", ""
        if config.ENABLE_SEARCH_GROUNDING or config.ENABLE_WEBTOOLS:
            for kw in config.SEARCH_TRIGGERS:
                kw = kw.strip()
                if kw and kw in t:
                    return "web", kw
        return None, None
    except Exception:
        return None, None


def _has_or() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def _step_up(reason: str) -> RouteDecision:
    """Complexity step-up: Laguna first (free, reasoning-capable)."""
    if _has_or():
        return RouteDecision("stepup", config.OPENROUTER_MODEL,
                             "openrouter", reason)
    return RouteDecision("stepup", config.ESCALATION_MODEL_FLASH,
                         "gemini", reason + " (no openrouter key; flash)")


def classify(
    text: str,
    history_len: int = 0,
    consecutive_failures: int = 0,
    gemini_rate_limited: bool = False,
) -> RouteDecision:
    """Pick a route for `text`. Never raises; fails closed to direct."""
    try:
        t = _norm(text)
        if not t:
            return DIRECT

        # 0. Explicit user override always wins.
        if "think harder" in t or "use the smart model" in t:
            # gemini-2.5-pro is retired for new keys: Laguna (free,
            # reasoning-capable) is the deep-thinking route instead.
            if _has_or():
                return RouteDecision("think", config.OPENROUTER_MODEL,
                                     "openrouter", "explicit deep-think request")
            return RouteDecision("think", config.ESCALATION_MODEL_FLASH,
                                 "gemini", "deep-think asked, no laguna key; flash")
        for trigger in ("use laguna", "ask laguna", "laguna"):
            if trigger in t:
                if _has_or():
                    return RouteDecision("think", config.OPENROUTER_MODEL,
                                         "openrouter", "explicit laguna request")
                return RouteDecision("think", config.ESCALATION_MODEL_FLASH,
                                     "gemini", "laguna asked but no key; flash instead")
        for trigger in ("use ultra", "ask ultra", "ultra",
                        "nemotron"):
            if trigger in t:
                if _has_or():
                    return RouteDecision("ultra", config.OPENROUTER_BIG,
                                         "openrouter", "explicit ultra request")
                return RouteDecision("think", config.ESCALATION_MODEL_FLASH,
                                     "gemini", "ultra asked but no key; flash instead")

        # 1. Gemini 429 recently -> shed load to free Laguna (same account).
        if gemini_rate_limited:
            if _has_or():
                return RouteDecision("stepup", config.OPENROUTER_MODEL,
                                     "openrouter", "gemini rate-limited fallback")
            return DIRECT  # nowhere to shed to; stay on base

        # 2. Prior turns failed (empty reply / exception) twice -> step up.
        if consecutive_failures >= 2:
            return _step_up("retry after failures")

        # 3. Keyword triggers: code, deep reasoning, long-form, advice.
        for kw in config.ESCALATE_KEYWORDS:
            kw = kw.strip()
            if kw and kw in t:
                # Long-form/research keywords step up; code-ish too.
                return _step_up(f"keyword: {kw}")

        # 4. Length / multi-part signals complexity.
        if len(t) >= config.ESCALATE_MIN_CHARS or t.count("?") >= 2:
            return _step_up("long/multi-part prompt")

        # 5. Deep conversation (many turns) -> stronger context handling.
        if history_len >= config.ESCALATE_HISTORY_DEPTH * 2:
            # history includes system + user/assistant pairs
            return _step_up("deep conversation")

        return DIRECT
    except Exception:
        return DIRECT


def fallbacks_for(decision: RouteDecision) -> list[tuple[str, str]]:
    """Backup chain for a decision: (provider, model) in attempt order.

    The caller tries each in turn until one returns a non-empty reply.
    Everything here is free-tier; paid models never appear.
    """
    chain = [(decision.provider, decision.model)]
    has_or = _has_or()

    def _add(provider: str, model: str):
        if (provider, model) not in chain:
            chain.append((provider, model))

    if decision.route == "direct":
        # Base failed (usually Gemini 429): fast free backup, then step-up.
        if has_or:
            _add("openrouter", config.OPENROUTER_FAST)
            _add("openrouter", config.OPENROUTER_MODEL)
    elif decision.route == "stepup":
        if decision.provider == "openrouter":
            _add("openrouter", config.OPENROUTER_BIG)  # laguna -> ultra
            _add("gemini", config.ESCALATION_MODEL_FLASH)
        else:  # no openrouter key: flash -> lite retry
            _add("gemini", config.LLM_MODEL)
    elif decision.route in ("think", "ultra"):
        if decision.provider == "openrouter":
            if decision.model == config.OPENROUTER_MODEL:
                _add("openrouter", config.OPENROUTER_BIG)
            else:
                _add("openrouter", config.OPENROUTER_MODEL)
            _add("gemini", config.ESCALATION_MODEL_FLASH)
        else:  # flash-only box: one lite retry, then give up gracefully
            _add("gemini", config.LLM_MODEL)
    return chain


def make_client(provider: str = "gemini"):
    """Build the Gemini OpenAI-compatible client.

    Import is lazy so `router.py` stays importable without keys.
    Raises RuntimeError with a clear message if the key is missing.
    """
    from openai import OpenAI

    if provider != "gemini" and provider != "openrouter":
        raise ValueError(f"unknown provider: {provider}")
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY (openrouter.ai/keys).")
        kwargs: dict = {}
        if config.OPENROUTER_SITE:
            kwargs["default_headers"] = {"HTTP-Referer": config.OPENROUTER_SITE}
        return OpenAI(api_key=api_key, base_url=config.OPENROUTER_BASE_URL,
                      **kwargs)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY first.")
    return OpenAI(api_key=api_key, base_url=config.GEMINI_BASE_URL)


# Model -> notes. Kept here so routing docs and code can't drift apart.
FREE_POOL = {
    "gemini direct": (config.LLM_MODEL, "base, free tier, ~15 RPM"),
    "gemini flash": (config.ESCALATION_MODEL_FLASH, "last-resort backup, 20/day cap"),
    "openrouter laguna": (config.OPENROUTER_MODEL, "step-up + think, free"),
    "openrouter fast": (config.OPENROUTER_FAST, "direct backup, TTFT ~0.2s, steered"),
    "openrouter big": (config.OPENROUTER_BIG, "explicit ultra + backup, best quality"),
}

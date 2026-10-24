"""Keyless web providers for Jarvis — no API keys, no billing, no quotas.

Bypasses Gemini grounding entirely (its free allowance is unreliable):

  web     — DuckDuckGo HTML results (+ Wikipedia fallback) -> Lite summary
  weather — Open-Meteo geocoding + forecast (free, no key)
  sports  — ESPN scoreboard API (free, no key): MLB/NFL/NBA/NHL

All fetchers return "" on any failure so callers fall through to the next
path. Spoken strings are short; detail goes to console/log only.
"""
from __future__ import annotations

import datetime
import html as _html
import re
import time
import urllib.parse

import config

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Jarvis/1.0"}

# url -> (timestamp, payload). Scores 60s, everything else 10 min.
_CACHE: dict = {}
_CACHE_TTL = {"scoreboard": 60, "default": 600}


def _cache_get(key: str, kind: str = "default"):
    entry = _CACHE.get(key)
    if entry and time.time() - entry[0] < _CACHE_TTL.get(kind, 600):
        return entry[1]
    return None


def _cache_put(key: str, value, kind: str = "default"):
    _CACHE[key] = (time.time(), value)
    if len(_CACHE) > 200:  # tiny LRU-ish bound
        _CACHE.pop(next(iter(_CACHE)))


def _http():
    import httpx
    return httpx.Client(headers=_UA, timeout=8.0, follow_redirects=True)


# ---------------------------------------------------------------------------
# General web search: DuckDuckGo HTML, Wikipedia fallback
# ---------------------------------------------------------------------------
def ddg_search(query: str, n: int = 5) -> list[dict]:
    """Return [{title, snippet, url}]. Empty list on any failure."""
    key = f"ddg:{query}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        client = _http()
        r = client.post("https://html.duckduckgo.com/html/",
                        data={"q": query})
        r.raise_for_status()
        page = r.text
        out = []
        # Result anchors carry the target in uddg=..., snippets nearby.
        for m in re.finditer(
                r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.S):
            href, title = m.group(1), re.sub(r"<.*?>", "", m.group(2))
            title = _html.unescape(title).strip()
            um = re.search(r"uddg=([^&]+)", href)
            url = urllib.parse.unquote(um.group(1)) if um else href
            if url.startswith("//"):
                url = "https:" + url
            # Snippet lives in the same result block after the anchor.
            block = page[m.start():m.start() + 2000]
            sm = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.S)
            snippet = _html.unescape(
                re.sub(r"<.*?>", "", sm.group(1)).strip()) if sm else ""
            if title and url.startswith("http"):
                out.append({"title": title, "snippet": snippet, "url": url})
            if len(out) >= n:
                break
        _cache_put(key, out)
        return out
    except Exception:
        return []


def wiki_summary(query: str) -> str:
    """One-paragraph Wikipedia intro for `query`, or ""."""
    key = f"wiki:{query}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        client = _http()
        s = client.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": 1, "format": "json"}).json()
        results = s.get("query", {}).get("search", [])
        if not results:
            return ""
        title = results[0]["title"]
        e = client.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "prop": "extracts", "exintro": 1,
            "explaintext": 1, "titles": title, "format": "json"}).json()
        pages = e.get("query", {}).get("pages", {})
        text = next(iter(pages.values()), {}).get("extract", "").strip()
        text = re.sub(r"\s+", " ", text)
        _cache_put(key, text)
        return text
    except Exception:
        return ""


def web_search_answer(question: str) -> tuple[str, str]:
    """Answer a general question via DDG (+Lite summary) or Wikipedia.

    Returns (spoken, console_detail); ("", reason) when nothing found.
    Refusals and homepage-only results never pass as answers.
    """
    from tools import summarize_for_voice  # local import: tools imports us
    results = [r for r in ddg_search(question) if _usable_result(r)]
    if results:
        chunks = [f"{r['title']}: {r['snippet']}".strip(": ")
                  for r in results if r["snippet"]]
        if chunks:
            spoken = summarize_with_lite(question, chunks)
            if spoken and not _looks_like_refusal(spoken):
                detail = spoken + "\nSOURCES: " + " ".join(
                    r["url"] for r in results)
                return spoken, detail
    wiki = wiki_summary(question)
    if wiki:
        spoken = summarize_for_voice(wiki)
        if not _looks_like_refusal(spoken):
            return spoken, wiki
    return "", "no web results with usable content"


def _usable_result(r: dict) -> bool:
    """Drop bare homepage hits (portal roots with no article path)."""
    try:
        path = urllib.parse.urlparse(r.get("url", "")).path.strip("/")
        if not path and not r.get("snippet"):
            return False
        return True
    except Exception:
        return False


_REFUSAL_MARKERS = ("cannot provide", "can't provide", "unable to provide",
                    "i don't have access", "i do not have access",
                    "no real-time access", "check a major", "check cnn",
                    "as an ai", "as a language model")


def _looks_like_refusal(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _REFUSAL_MARKERS)


_NUMBER_WORDS = ("one", "two", "three", "four", "five")


def news_answer(count: int = 5) -> tuple[str, str]:
    """Top headlines via BBC/NPR RSS (free, keyless, stable).

    Returns (spoken, console_detail); ("", reason) on failure.
    """
    feeds = ["http://feeds.bbci.co.uk/news/rss.xml",
             "https://feeds.npr.org/1001/rss.xml"]
    titles: list[str] = []
    try:
        import xml.etree.ElementTree as ET
        client = _http()
        for url in feeds:
            try:
                raw = client.get(url).content  # bytes: ET honors the XML encoding
                root = ET.fromstring(raw)
                for item in root.iter("item"):
                    title = (item.findtext("title") or "").strip()
                    title = re.sub(r"\s+", " ", title)
                    if title and title not in titles:
                        titles.append(title)
                    if len(titles) >= count:
                        break
            except Exception:
                continue
            if len(titles) >= count:
                break
    except Exception as e:
        return "", f"RSS fetch failed: {str(e)[:120]}"
    if not titles:
        return "", "RSS feeds returned no headlines"
    titles = titles[:count]
    from tools import summarize_for_voice
    parts = [f"{_NUMBER_WORDS[i]}: {t}" for i, t in enumerate(titles)]
    spoken = summarize_for_voice("Headlines: " + ". ".join(parts), max_words=90)
    detail = "HEADLINES:\n" + "\n".join(f"- {t}" for t in titles)
    return spoken, detail


def summarize_with_lite(question: str, snippets: list[str]) -> str:
    """Compress search snippets to <=60 spoken words via plain chat (works)."""
    from tools import summarize_for_voice
    import router
    try:
        client = router.make_client("gemini")
        prompt = ("Answer in at most two short spoken sentences (no markdown, "
                  "no URLs, under 60 words). Question: " + question +
                  "\nSearch results:\n" + "\n".join(f"- {s}" for s in snippets[:6]))
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return summarize_for_voice(resp.choices[0].message.content or "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Weather: Open-Meteo (free, no key) + IP geo fallback
# ---------------------------------------------------------------------------
_WMO = {0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
        45: "foggy", 48: "icy fog", 51: "light drizzle", 53: "drizzle",
        55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
        71: "light snow", 73: "snow", 75: "heavy snow", 80: "light showers",
        81: "showers", 82: "heavy showers", 95: "thunderstorms"}


def geocode(city: str) -> tuple[float, float, str] | None:
    key = f"geo:{city.lower()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        client = _http()
        r = client.get("https://geocoding-api.open-meteo.com/v1/search",
                       params={"name": city, "count": 1}).json()
        hits = r.get("results") or []
        if not hits:
            return None
        h = hits[0]
        name = h.get("name", city)
        admin = h.get("admin1") or h.get("country") or ""
        label = f"{name}, {admin}".rstrip(", ")
        loc = (float(h["latitude"]), float(h["longitude"]), label)
        _cache_put(key, loc)
        return loc
    except Exception:
        return None


def my_location() -> tuple[float, float, str] | None:
    """City-level geo from IP (free, no key). None on failure."""
    try:
        client = _http()
        r = client.get("http://ip-api.com/json/",
                       params={"fields": "status,city,lat,lon"}).json()
        if r.get("status") == "success":
            return float(r["lat"]), float(r["lon"]), r.get("city", "here")
        return None
    except Exception:
        return None


def _forecast_offsets(text: str) -> list[int]:
    """Which days (0=today) the user asked about. Max 10, always >= [0]."""
    t = text.lower()
    m = re.search(r"(\d{1,2})\s*-?\s*day", t)
    if m:
        return list(range(min(max(int(m.group(1)), 1), 10)))
    if "weekend" in t or "saturday" in t or "sunday" in t:
        today = datetime.date.today().weekday()  # Mon=0
        offs = sorted({(5 - today) % 7, (6 - today) % 7} - {0} or {0})
        return offs or [0]
    if "week" in t or "7-day" in t or "7 day" in t:
        return list(range(7))
    if "tomorrow" in t:
        return [1]
    return [0]


def weather_answer(text: str) -> tuple[str, str]:
    """Spoken weather: current conditions + requested horizon.

    City priority: named in transcript -> JARVIS_LOCATION default ->
    IP geo. Explicit city always wins for "weather in X" questions.
    Horizons: "10 day" (max 10), "week", "weekend", "tomorrow", else today.
    """
    m = re.search(r"\bin\s+([A-Za-z][A-Za-z .'-]{1,40})$", text.strip(), re.I)
    city = m.group(1).strip().rstrip(".!?,") if m else ""
    loc = None
    if city:
        loc = geocode(city)
        if not loc:
            return "", f"couldn't geocode {city!r}"
    elif config.DEFAULT_LOCATION:
        loc = geocode(config.DEFAULT_LOCATION)
    if loc is None:
        loc = my_location() or None
    if not loc:
        return "", "no city named and IP geo failed — ask 'weather in <city>'"
    lat, lon, label = loc
    offs = _forecast_offsets(text)
    try:
        client = _http()
        f = client.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "temperature_unit": "fahrenheit",
            "current": "temperature_2m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max,weathercode",
            "timezone": "auto",
            "forecast_days": max(offs) + 1}).json()
        cur, daily = f["current"], f["daily"]
        today = datetime.date.today()
        bits = []
        for n, i in enumerate(offs):
            hi = round(daily["temperature_2m_max"][i])
            lo = round(daily["temperature_2m_min"][i])
            pop = daily["precipitation_probability_max"][i] or 0
            cond = _WMO.get((daily.get("weathercode") or [None])[i], "")
            day = "Today" if i == 0 else (today + datetime.timedelta(days=i)
                                          ).strftime("%A")
            bit = f"{day}: high {hi}, low {lo}"
            if cond and (n > 0 or pop >= 20):
                bit += f", {cond}"
            if pop >= 20:
                bit += f", {pop}% chance of precipitation"
            bits.append(bit)
        if offs == [0]:
            temp = round(cur["temperature_2m"])
            cond = _WMO.get(cur.get("weather_code"), "unsettled")
            spoken = (f"It's {temp} degrees and {cond} in {label}, "
                      + bits[0].split(": ", 1)[1] + ".")
        else:
            spoken = f"{len(offs)}-day forecast for {label}. " + ". ".join(bits) + "."
        return spoken, spoken
    except Exception as e:
        return "", f"forecast fetch failed: {str(e)[:120]}"


# ---------------------------------------------------------------------------
# Sports: ESPN scoreboard API (free, no key). MLB/NFL/NBA/NHL.
# (sport, league, abbr, location, nickname, extra aliases)
# ---------------------------------------------------------------------------
_TEAMS = [
    # MLB
    ("baseball", "mlb", "ARI", "Arizona", "Diamondbacks", ["dbacks", "d-backs"]),
    ("baseball", "mlb", "ATL", "Atlanta", "Braves", []),
    ("baseball", "mlb", "BAL", "Baltimore", "Orioles", ["o's"]),
    ("baseball", "mlb", "BOS", "Boston", "Red Sox", ["red sox", "bosox"]),
    ("baseball", "mlb", "CHC", "Chicago", "Cubs", []),
    ("baseball", "mlb", "CHW", "Chicago", "White Sox", ["white sox", "chisox"]),
    ("baseball", "mlb", "CIN", "Cincinnati", "Reds", []),
    ("baseball", "mlb", "CLE", "Cleveland", "Guardians", []),
    ("baseball", "mlb", "COL", "Colorado", "Rockies", []),
    ("baseball", "mlb", "DET", "Detroit", "Tigers", []),
    ("baseball", "mlb", "HOU", "Houston", "Astros", ["'stros"]),
    ("baseball", "mlb", "KC", "Kansas City", "Royals", []),
    ("baseball", "mlb", "LAA", "Los Angeles", "Angels", []),
    ("baseball", "mlb", "LAD", "Los Angeles", "Dodgers", []),
    ("baseball", "mlb", "MIA", "Miami", "Marlins", []),
    ("baseball", "mlb", "MIL", "Milwaukee", "Brewers", ["brew crew"]),
    ("baseball", "mlb", "MIN", "Minnesota", "Twins", ["twinkies"]),
    ("baseball", "mlb", "NYY", "New York", "Yankees", ["yanks", "bronx bombers"]),
    ("baseball", "mlb", "NYM", "New York", "Mets", ["mets", "amazins"]),
    ("baseball", "mlb", "ATH", "Sacramento", "Athletics", ["athletics", "a's"]),
    ("baseball", "mlb", "PHI", "Philadelphia", "Phillies", ["phils", "fightins"]),
    ("baseball", "mlb", "PIT", "Pittsburgh", "Pirates", ["bucs", "buccos"]),
    ("baseball", "mlb", "SD", "San Diego", "Padres", ["friars"]),
    ("baseball", "mlb", "SF", "San Francisco", "Giants", []),
    ("baseball", "mlb", "SEA", "Seattle", "Mariners", ["m's"]),
    ("baseball", "mlb", "STL", "St. Louis", "Cardinals", ["cards", "redbirds"]),
    ("baseball", "mlb", "TB", "Tampa Bay", "Rays", []),
    ("baseball", "mlb", "TEX", "Texas", "Rangers", []),
    ("baseball", "mlb", "TOR", "Toronto", "Blue Jays", ["jays"]),
    ("baseball", "mlb", "WSH", "Washington", "Nationals", ["nats"]),
    # NFL
    ("football", "nfl", "ARI", "Arizona", "Cardinals", ["cards"]),
    ("football", "nfl", "ATL", "Atlanta", "Falcons", ["dirty birds"]),
    ("football", "nfl", "BAL", "Baltimore", "Ravens", []),
    ("football", "nfl", "BUF", "Buffalo", "Bills", ["bills mafia"]),
    ("football", "nfl", "CAR", "Carolina", "Panthers", []),
    ("football", "nfl", "CHI", "Chicago", "Bears", ["da bears"]),
    ("football", "nfl", "CIN", "Cincinnati", "Bengals", ["who dey"]),
    ("football", "nfl", "CLE", "Cleveland", "Browns", ["dawg pound"]),
    ("football", "nfl", "DAL", "Dallas", "Cowboys", ["america's team", "boys"]),
    ("football", "nfl", "DEN", "Denver", "Broncos", []),
    ("football", "nfl", "DET", "Detroit", "Lions", []),
    ("football", "nfl", "GB", "Green Bay", "Packers", ["pack", "cheeseheads"]),
    ("football", "nfl", "HOU", "Houston", "Texans", []),
    ("football", "nfl", "IND", "Indianapolis", "Colts", []),
    ("football", "nfl", "JAX", "Jacksonville", "Jaguars", ["jags", "duval"]),
    ("football", "nfl", "KC", "Kansas City", "Chiefs", ["chiefs kingdom"]),
    ("football", "nfl", "LAC", "Los Angeles", "Chargers", ["bolt up"]),
    ("football", "nfl", "LAR", "Los Angeles", "Rams", []),
    ("football", "nfl", "LV", "Las Vegas", "Raiders", ["raider nation"]),
    ("football", "nfl", "MIA", "Miami", "Dolphins", ["fins", "phins"]),
    ("football", "nfl", "MIN", "Minnesota", "Vikings", ["skol"]),
    ("football", "nfl", "NE", "New England", "Patriots", ["pats"]),
    ("football", "nfl", "NO", "New Orleans", "Saints", ["who dat"]),
    ("football", "nfl", "NYG", "New York", "Giants", ["g-men", "big blue"]),
    ("football", "nfl", "NYJ", "New York", "Jets", ["gang green"]),
    ("football", "nfl", "PHI", "Philadelphia", "Eagles", ["birds", "fly eagles fly"]),
    ("football", "nfl", "PIT", "Pittsburgh", "Steelers", ["steel curtain"]),
    ("football", "nfl", "SF", "San Francisco", "49ers", ["niners", "faithful"]),
    ("football", "nfl", "SEA", "Seattle", "Seahawks", ["hawks", "12s"]),
    ("football", "nfl", "TB", "Tampa Bay", "Buccaneers", ["bucs"]),
    ("football", "nfl", "TEN", "Tennessee", "Titans", ["titan up"]),
    ("football", "nfl", "WSH", "Washington", "Commanders", []),
    # NBA
    ("basketball", "nba", "ATL", "Atlanta", "Hawks", []),
    ("basketball", "nba", "BOS", "Boston", "Celtics", ["celts", "green team"]),
    ("basketball", "nba", "BKN", "Brooklyn", "Nets", []),
    ("basketball", "nba", "CHA", "Charlotte", "Hornets", ["buzz city"]),
    ("basketball", "nba", "CHI", "Chicago", "Bulls", []),
    ("basketball", "nba", "CLE", "Cleveland", "Cavaliers", ["cavs", "land"]),
    ("basketball", "nba", "DAL", "Dallas", "Mavericks", ["mavs"]),
    ("basketball", "nba", "DEN", "Denver", "Nuggets", []),
    ("basketball", "nba", "DET", "Detroit", "Pistons", ["motorcade", "bad boys"]),
    ("basketball", "nba", "GS", "Golden State", "Warriors", ["dubs"]),
    ("basketball", "nba", "HOU", "Houston", "Rockets", ["clutch city"]),
    ("basketball", "nba", "IND", "Indiana", "Pacers", []),
    ("basketball", "nba", "LAC", "Los Angeles", "Clippers", ["clipper nation"]),
    ("basketball", "nba", "LAL", "Los Angeles", "Lakers", ["lake show"]),
    ("basketball", "nba", "MEM", "Memphis", "Grizzlies", ["grit n grind"]),
    ("basketball", "nba", "MIA", "Miami", "Heat", ["heat culture"]),
    ("basketball", "nba", "MIL", "Milwaukee", "Bucks", ["fear the deer"]),
    ("basketball", "nba", "MIN", "Minnesota", "Timberwolves", ["wolves"]),
    ("basketball", "nba", "NO", "New Orleans", "Pelicans", ["pels"]),
    ("basketball", "nba", "NY", "New York", "Knicks", ["knickerbockers"]),
    ("basketball", "nba", "OKC", "Oklahoma City", "Thunder", ["thunder up"]),
    ("basketball", "nba", "ORL", "Orlando", "Magic", []),
    ("basketball", "nba", "PHI", "Philadelphia", "76ers", ["sixers"]),
    ("basketball", "nba", "PHX", "Phoenix", "Suns", ["valley", "suns up"]),
    ("basketball", "nba", "POR", "Portland", "Trail Blazers", ["blazers", "ripcity"]),
    ("basketball", "nba", "SAC", "Sacramento", "Kings", ["light the beam"]),
    ("basketball", "nba", "SA", "San Antonio", "Spurs", ["go spurs go"]),
    ("basketball", "nba", "TOR", "Toronto", "Raptors", ["we the north"]),
    ("basketball", "nba", "UTAH", "Utah", "Jazz", []),
    ("basketball", "nba", "WSH", "Washington", "Wizards", ["dc family"]),
    # NHL
    ("hockey", "nhl", "ANA", "Anaheim", "Ducks", []),
    ("hockey", "nhl", "BOS", "Boston", "Bruins", ["spoked b"]),
    ("hockey", "nhl", "BUF", "Buffalo", "Sabres", []),
    ("hockey", "nhl", "CAR", "Carolina", "Hurricanes", ["canes", "storm surge"]),
    ("hockey", "nhl", "CBJ", "Columbus", "Blue Jackets", ["jackets"]),
    ("hockey", "nhl", "CGY", "Calgary", "Flames", ["c of red"]),
    ("hockey", "nhl", "CHI", "Chicago", "Blackhawks", []),
    ("hockey", "nhl", "COL", "Colorado", "Avalanche", ["avs", "go avs go"]),
    ("hockey", "nhl", "DAL", "Dallas", "Stars", []),
    ("hockey", "nhl", "DET", "Detroit", "Red Wings", ["wings", "hockeytown"]),
    ("hockey", "nhl", "EDM", "Edmonton", "Oilers", ["let's go oilers"]),
    ("hockey", "nhl", "FLA", "Florida", "Panthers", ["time to hunt", "cats"]),
    ("hockey", "nhl", "LA", "Los Angeles", "Kings", ["go kings go"]),
    ("hockey", "nhl", "MIN", "Minnesota", "Wild", []),
    ("hockey", "nhl", "MTL", "Montreal", "Canadiens", ["habs"]),
    ("hockey", "nhl", "NJ", "New Jersey", "Devils", []),
    ("hockey", "nhl", "NSH", "Nashville", "Predators", ["preds", "smashville"]),
    ("hockey", "nhl", "NYI", "New York", "Islanders", ["isles"]),
    ("hockey", "nhl", "NYR", "New York", "Rangers", ["blueshirts", "no quit"]),
    ("hockey", "nhl", "OTT", "Ottawa", "Senators", ["sens", "go sens go"]),
    ("hockey", "nhl", "PHI", "Philadelphia", "Flyers", ["orange and black"]),
    ("hockey", "nhl", "PIT", "Pittsburgh", "Penguins", ["pens", "lets go pens"]),
    ("hockey", "nhl", "SEA", "Seattle", "Kraken", ["release the kraken"]),
    ("hockey", "nhl", "SJ", "San Jose", "Sharks", ["teal town"]),
    ("hockey", "nhl", "STL", "St. Louis", "Blues", ["lets go blues"]),
    ("hockey", "nhl", "TB", "Tampa Bay", "Lightning", ["bolts", "go bolts"]),
    ("hockey", "nhl", "TOR", "Toronto", "Maple Leafs", ["leafs", "go leafs go"]),
    ("hockey", "nhl", "VAN", "Vancouver", "Canucks", ["go canucks go"]),
    ("hockey", "nhl", "VGS", "Vegas", "Golden Knights", ["knights", "vegas born"]),
    ("hockey", "nhl", "WPG", "Winnipeg", "Jets", ["go jets go"]),
    ("hockey", "nhl", "WSH", "Washington", "Capitals", ["caps", "rock the red"]),
    ("hockey", "nhl", "UTAH", "Utah", "Mammoth", ["utah hockey club"]),
]

_ALIAS_TO_TEAM: dict[str, tuple] = {}


def _build_alias_map():
    for entry in _TEAMS:
        sport, league, abbr, loc, nick, extras = entry
        keys = {loc.lower(), nick.lower(), f"{loc} {nick}".lower(),
                abbr.lower()} | {e.lower() for e in extras}
        for k in keys:
            _ALIAS_TO_TEAM.setdefault(k, entry)


_build_alias_map()


def detect_teams(text: str) -> list[tuple]:
    """Find mentioned teams, longest aliases first. Dedupe by entry."""
    t = text.lower()
    found, used_spans = [], []
    for alias in sorted(_ALIAS_TO_TEAM, key=len, reverse=True):
        for m in re.finditer(r"\b" + re.escape(alias) + r"\b", t):
            span = m.span()
            if any(s <= span[0] < e or s < span[1] <= e for s, e in used_spans):
                continue
            used_spans.append(span)
            entry = _ALIAS_TO_TEAM[alias]
            if entry not in found:
                found.append(entry)
    return found


def _scoreboard(sport: str, league: str) -> dict | None:
    key = f"sb:{league}"
    cached = _cache_get(key, "scoreboard")
    if cached is not None:
        return cached
    try:
        client = _http()
        r = client.get(
            f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard")
        r.raise_for_status()
        data = r.json()
        _cache_put(key, data, "scoreboard")
        return data
    except Exception:
        return None


def _team_in_event(entry: tuple, event: dict) -> dict | None:
    """Return the competitor dict for `entry` in an ESPN event, else None."""
    _, _, abbr, loc, nick, extras = entry
    names = {abbr.lower(), loc.lower(), nick.lower(),
             f"{loc} {nick}".lower()} | {e.lower() for e in extras}
    for comp in event.get("competitions", [{}])[0].get("competitors", []):
        team = comp.get("team", {})
        hay = " ".join(str(team.get(k, "")) for k in
                       ("displayName", "shortDisplayName", "nickname",
                        "location", "abbreviation")).lower()
        if any(n and n in hay for n in names):
            return comp
    return None


def _describe_event(event: dict, focus_abbr: str) -> str:
    comp = event.get("competitions", [{}])[0]
    comps = comp.get("competitors", [])
    state = event.get("status", {}).get("type", {}).get("state", "")
    detail = event.get("status", {}).get("type", {}).get("shortDetail", "")
    names = {}
    for c in comps:
        team = c.get("team", {})
        short = team.get("shortDisplayName") or team.get("nickname") or "?"
        names[c.get("homeAway", "?")] = (short, c.get("score", "?"))
    home, away = names.get("home", ("?", "?")), names.get("away", ("?", "?"))
    if state == "post":
        w = next((c for c in comps if c.get("winner")), None)
        winner = (w.get("team", {}).get("shortDisplayName", "Someone")
                  if w else "Nobody")
        return f"Final: {away[0]} {away[1]}, {home[0]} {home[1]}. {winner} won."
    if state == "in":
        return f"{away[0]} {away[1]}, {home[0]} {home[1]} — {detail}."
    # pre
    try:
        import sys
        dt = datetime.datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
        hour_fmt = "%#I" if sys.platform == "win32" else "%-I"
        local = dt.astimezone().strftime(f"%A at {hour_fmt}:%M %p")
    except Exception:
        local = detail or "scheduled"
    return f"{away[0]} at {home[0]}, {local}."


def sports_answer(text: str) -> tuple[str, str]:
    """Spoken score/status for mentioned team(s). ("", reason) if none."""
    teams = detect_teams(text)
    if not teams:
        return "", "no known team detected"
    leagues: dict[tuple[str, str], list] = {}
    for entry in teams:
        leagues.setdefault((entry[0], entry[1]), []).append(entry)
    for (sport, league), entries in leagues.items():
        board = _scoreboard(sport, league)
        if not board:
            continue
        events = board.get("events", [])
        if len(entries) >= 2:
            for ev in events:
                if all(_team_in_event(e, ev) for e in entries[:2]):
                    spoken = _describe_event(ev, entries[0][2])
                    return spoken, spoken
        for entry in entries:
            mine = [ev for ev in events if _team_in_event(entry, ev)]
            if mine:
                # Prefer live/final today over future fixtures.
                def _rank(ev):
                    st = ev.get("status", {}).get("type", {}).get("state", "")
                    return {"in": 0, "post": 1, "pre": 2}.get(st, 3)
                ev = sorted(mine, key=_rank)[0]
                spoken = _describe_event(ev, entry[2])
                return spoken, spoken
            name = f"{entry[3]} {entry[4]}"
            return f"The {name} don't have a game on the board right now.", ""
    return "", "scoreboard unreachable (ESPN)" if teams else ""


# ---------------------------------------------------------------------------
# Music: "play <song>" -> YouTube Music autoplay link (free, no API key).
# yt-dlp resolves the query to a video id; the watch URL autoplays.
# ---------------------------------------------------------------------------
def _yt_dlp_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("yt_dlp") is not None


def resolve_song_candidates(query: str, n: int = 5,
                              timeout: int = 30) -> list[tuple[str, str]]:
    """Top-N (video_id, title) for `query` via yt-dlp search. [] on failure."""
    import subprocess
    import sys
    try:
        r = subprocess.run(
            [sys.executable, "-m", "yt_dlp", f"ytsearch{n}:{query}",
             "--print", "%(id)s | %(title)s", "--no-playlist",
             "--quiet", "--no-warnings"],
            capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.strip().splitlines():
            vid, _, title = line.partition(" | ")
            vid, title = vid.strip(), title.strip()
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid) and title:
                out.append((vid, title))
        return out
    except Exception:
        return []


def resolve_song_id(query: str, timeout: int = 30) -> str | None:
    """First YouTube video id for `query` via yt-dlp search. None on failure."""
    cands = resolve_song_candidates(query, n=1, timeout=timeout)
    return cands[0][0] if cands else None


def pick_song_id(query: str, candidates: list[tuple[str, str]]) -> str:
    """Ask Flash-Lite which candidate best matches the request.

    Returns a validated video id; falls back to candidates[0] (or "" if
    empty) on any failure. Cheap (~100 tokens) and fixes blind top-pick
    misses like lyric-fragments and cover versions.
    """
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0][0]
    import router
    try:
        lines = "\n".join(f"{vid} — {title}" for vid, title in candidates)
        prompt = ("A user asked a voice assistant to play this: "
                  f"\"{query}\".\nWhich YouTube result below is most likely "
                  "the exact song they mean (prefer official audio/video and "
                  "the named artist; avoid covers, lyrics videos, and "
                  "unrelated songs)? Reply with ONLY the video id, nothing else.\n"
                  + lines)
        client = router.make_client("gemini")
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        pick = (resp.choices[0].message.content or "").strip()
        ids = {vid for vid, _ in candidates}
        if pick in ids:
            return pick
    except Exception:
        pass
    return candidates[0][0]


def play_song_answer(text: str) -> tuple[str, str]:
    """Open a song so it starts playing. Returns (spoken, console_detail)."""
    import webbrowser
    clean = text.strip().rstrip(".!?").strip()
    m = re.match(r"^play\s+(.*?)(?:\s+on\s+(youtube music|youtube))?\s*$",
                 clean, re.I)
    query = (m.group(1) if m else re.sub(r"^play\s+", "", clean, flags=re.I)).strip()
    on_yt = (m.group(2) or "").lower() if m else ""
    if not query:
        return "", "nothing to play after 'play'"
    # Liked songs: YouTube Music pseudo-playlist (needs browser login).
    if any(p in query.lower() for p in (
            "liked playlist", "liked songs", "liked music", "my likes")):
        url = "https://music.youtube.com/playlist?list=LM"
        try:
            webbrowser.open(url)
        except Exception as e:
            return "", f"browser open failed: {e}"
        spoken = "Playing your liked songs on YouTube Music."
        return spoken, f"{spoken}\nURL: {url}"
    if _yt_dlp_available():
        cands = resolve_song_candidates(query)
        vid = pick_song_id(query, cands) if cands else ""
        if vid:
            picked = next((t for v, t in cands if v == vid), "")
            if on_yt == "youtube":
                url = f"https://www.youtube.com/watch?v={vid}&autoplay=1"
                where = "YouTube"
            else:
                url = f"https://music.youtube.com/watch?v={vid}&autoplay=1"
                where = "YouTube Music"
            try:
                webbrowser.open(url)
            except Exception as e:
                return "", f"browser open failed: {e}"
            spoken = f"Playing {query} on {where}."
            detail = f"{spoken}\nURL: {url}"
            if picked:
                detail += f"\nPicked: {picked}"
            return spoken, detail
    # Fallback: no yt-dlp or no match -> search-results page (user picks).
    q = urllib.parse.quote_plus(query + " music")
    url = f"https://music.youtube.com/search?q={q}"
    try:
        webbrowser.open(url)
    except Exception as e:
        return "", f"browser open failed: {e}"
    spoken = f"I couldn't pin down the exact track — showing results for {query}."
    return spoken, f"{spoken}\nURL: {url}"
def league_label(sport: str, league: str) -> str:
    return {"mlb": "baseball", "nfl": "football",
            "nba": "basketball", "nhl": "hockey"}.get(league, league)

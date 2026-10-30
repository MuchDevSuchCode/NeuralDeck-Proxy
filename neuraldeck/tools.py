"""Built-in tools the Chat tab can offer a model.

They run here, on the deck, not in the browser: a key stays in the
deck's settings, and the model only sees what each tool returns. The
chat turns the whole pack off with the ``tools_enabled`` setting, and a
user can untick any one of them in Chat Settings like any other tool.

  get_datetime      the date and time, in a named zone or the deck's own
  get_weather       current conditions and a forecast (Open-Meteo, no key)
  web_search        the web (Brave Search with a key, or a SearXNG instance)
  news              recent news on a topic, via the same provider
  fetch_url         a page as readable text, so a search result can be read
  wikipedia         an article summary, which beats a search snippet for facts
  calculate         arithmetic and math functions, exactly
  unit_convert      length, mass, temperature, volume, speed, data, …
  currency          exchange rates (frankfurter.app, no key)
  run_python        Python in a throwaway Docker container, no network
  read_github       a repo's tree, a file, an issue or a pull request
  search_chats      the account's own saved conversations
  list_documents, search_documents, read_document
                    the account's document folder (and a shared one)
  deck_status, list_models, launch_model, stop_model, benchmark
                    the deck itself; need the Dashboard / Benchmarks view
  generate_image    ComfyUI, through a workflow file with placeholders
  speak             the TTS endpoint; the clip comes back as a link
  schedule_task, list_tasks, cancel_task
                    prompts the deck runs on a schedule (scheduler.py)
  home_assistant    states and services on a Home Assistant instance
  send_email        mail through the SMTP server in Settings

``fetch_url`` refuses private, loopback and link-local addresses so a
model cannot be talked into reading the deck's own API, the proxy, or
anything else on the LAN. Tools that touch the deck are offered only to
accounts with the matching view, and integrations only once configured.
"""

import ast
import datetime as _dt
import html
import ipaddress
import json
import math
import operator
import os
import re
import secrets
import shutil
import smtplib
import socket
import subprocess
import tempfile
import threading
import time
import zoneinfo
from email.message import EmailMessage
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx

from . import chats, config

# deck.py fills this in at startup: live telemetry, the launcher, the
# scheduler, the event loop. Without it the deck-side tools are not offered.
CONTEXT: dict = {}

UA = "NeuralDeck/1.0 (+https://github.com/MuchDevSuchCode/NeuralDeck-Proxy)"
TIMEOUT = httpx.Timeout(20.0, connect=8.0)


class ToolError(Exception):
    """A message for the model (and the user), not a crash."""


# ── definitions (OpenAI function-calling shape) ─────────────────────────────

def _fn(name, description, props, required=()):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": list(required)}}}


DEFINITIONS = [
    _fn("get_datetime",
        "The current date and time. Use it before reasoning about 'today', 'tomorrow', "
        "deadlines or anything relative to now.",
        {"timezone": {"type": "string",
                      "description": "An IANA zone such as America/Chicago or Europe/Berlin. "
                                     "Omit for the deck's local zone."}}),
    _fn("get_weather",
        "Current weather and a short daily forecast for a place, from Open-Meteo. "
        "Give a city (and region or country when ambiguous), or latitude and longitude.",
        {"location": {"type": "string", "description": "A place name, e.g. 'Austin, TX' or 'Paris'."},
         "latitude": {"type": "number"}, "longitude": {"type": "number"},
         "days": {"type": "integer", "description": "Forecast days, 1–7 (default 3)."},
         "units": {"type": "string", "enum": ["metric", "imperial"],
                   "description": "metric (°C, km/h, mm) or imperial (°F, mph, in). Default metric."}}),
    _fn("web_search",
        "Search the web. Returns titles, URLs and snippets; call fetch_url on a result to read it. "
        "Use for anything recent, factual or outside your training data.",
        {"query": {"type": "string"},
         "count": {"type": "integer", "description": "Results to return, 1–10 (default 5)."}},
        required=["query"]),
    _fn("fetch_url",
        "Fetch a web page and return its readable text (scripts, menus and boilerplate removed). "
        "Long pages are cut at max_chars.",
        {"url": {"type": "string", "description": "An http(s) URL."},
         "max_chars": {"type": "integer", "description": "Characters to return, up to 40000 (default 8000)."}},
        required=["url"]),
]

def _opt(name, description, props, required=(), access=None):
    d = _fn(name, description, props, required)
    d["_access"] = access
    return d


DEFINITIONS += [
    _fn("news",
        "Recent news articles on a topic: titles, sources, dates, URLs and snippets.",
        {"query": {"type": "string"},
         "count": {"type": "integer", "description": "Articles to return, 1–10 (default 5)."}},
        required=["query"]),
    _fn("wikipedia",
        "Look a topic up on Wikipedia and return the article's summary. More reliable than a "
        "search snippet for facts, dates, definitions and people.",
        {"query": {"type": "string", "description": "A topic or article title."},
         "language": {"type": "string", "description": "Wikipedia language code (default en)."}},
        required=["query"]),
    _fn("calculate",
        "Evaluate a math expression exactly: + - * / ** % //, parentheses, and functions such as "
        "sqrt, sin, cos, tan, log, log10, log2, exp, abs, round, floor, ceil, factorial, gcd, "
        "and the constants pi and e. Use it for any arithmetic instead of guessing.",
        {"expression": {"type": "string", "description": "e.g. (1200*1.07**5) or sqrt(2)*pi"}},
        required=["expression"]),
    _fn("unit_convert",
        "Convert a value between units of length, mass, temperature, volume, area, speed, time, "
        "data size, energy, power or pressure, e.g. 5 km to mi, 350 F to C, 2 TB to GB.",
        {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
        required=["value", "from_unit", "to_unit"]),
    _fn("currency",
        "Convert money between currencies at the latest reference rate (ECB via frankfurter.app).",
        {"amount": {"type": "number"}, "from_currency": {"type": "string", "description": "ISO code, e.g. USD"},
         "to_currency": {"type": "string", "description": "ISO code, e.g. EUR"}},
        required=["amount", "from_currency", "to_currency"]),
    _fn("run_python",
        "Run Python 3.12 code in a throwaway container with numpy, pandas, sympy and matplotlib, "
        "no network and no access to the machine. stdout and stderr come back. Print what you "
        "want to see.",
        {"code": {"type": "string"},
         "timeout_s": {"type": "integer", "description": "Seconds allowed, up to 120 (default 30)."}},
        required=["code"]),
    _fn("read_github",
        "Read from a public GitHub repository without the web page's clutter: the file tree, "
        "a file's contents, the README, an issue or a pull request (with its comments).",
        {"action": {"type": "string", "enum": ["tree", "file", "readme", "issue", "pull"]},
         "repo": {"type": "string", "description": "owner/name, e.g. ggml-org/llama.cpp"},
         "path": {"type": "string", "description": "For tree and file: a path in the repo."},
         "ref": {"type": "string", "description": "Branch, tag or commit (default: the default branch)."},
         "number": {"type": "integer", "description": "For issue and pull."}},
        required=["action", "repo"]),
    _fn("search_chats",
        "Search this user's own saved conversations on the deck for earlier discussions, "
        "decisions or facts. Returns matching chats with the lines that matched.",
        {"query": {"type": "string"},
         "limit": {"type": "integer", "description": "Chats to return, 1–10 (default 5)."}},
        required=["query"]),
    _fn("list_documents",
        "List the files in the user's document folder on the deck (and the shared folder): "
        "text, markdown, code, CSV, JSON and PDF.",
        {}),
    _fn("search_documents",
        "Search the user's documents for a phrase or word; returns the files and matching lines.",
        {"query": {"type": "string"}}, required=["query"]),
    _fn("read_document",
        "Read a document from the user's folder, as returned by list_documents. Long files come "
        "back in pieces: use offset to continue.",
        {"path": {"type": "string"},
         "offset": {"type": "integer", "description": "Character to start at (default 0)."},
         "max_chars": {"type": "integer", "description": "Characters to return, up to 40000 (default 8000)."}},
        required=["path"]),
    _opt("deck_status",
         "What the NeuralDeck machine is doing now: models loaded and on which ports, RAM and VRAM "
         "used and free, GPU load and temperature, recent throughput, running services.",
         {}, access="dash"),
    _opt("list_models",
         "The models available to launch on this deck, with size and capabilities.",
         {}, access="dash"),
    _opt("launch_model",
         "Load a model onto the GPU so it can be chatted with. Takes a while; the result says "
         "whether it started. Use list_models for exact names.",
         {"model": {"type": "string", "description": "A name from list_models."},
          "backend": {"type": "string", "description": "A build label from deck_status; default: the deck's default."},
          "ctx": {"type": "integer", "description": "Context window in tokens; default: the deck's default."},
          "replace": {"type": "boolean", "description": "Stop running models first if memory is short."}},
         required=["model"], access="dash"),
    _opt("stop_model",
         "Unload a running model (free its memory).",
         {"model": {"type": "string", "description": "The running model's name, or its port."}},
         required=["model"], access="dash"),
    _opt("benchmark",
         "Run the deck's standard benchmark (a fixed 250-word prompt at temperature 0) against a "
         "loaded model and return prefill and decode tokens per second. Takes 10–60 seconds.",
         {"model": {"type": "string", "description": "A loaded model's name."}},
         required=["model"], access="bench"),
    _fn("generate_image",
        "Generate an image from a text prompt with the deck's image model. Returns markdown "
        "for the image; put that markdown in your reply so the user sees it.",
        {"prompt": {"type": "string"}, "negative": {"type": "string"},
         "width": {"type": "integer"}, "height": {"type": "integer"},
         "seed": {"type": "integer"}},
        required=["prompt"]),
    _fn("speak",
        "Turn text into speech with the deck's TTS service. Returns a link to the audio clip; "
        "put it in your reply so the user can play it.",
        {"text": {"type": "string"}, "voice": {"type": "string"}},
        required=["text"]),
    _fn("schedule_task",
        "Have the deck run a prompt on a schedule, with these tools, and save each result as a "
        "chat for the user. Give either every_minutes or daily_at.",
        {"prompt": {"type": "string", "description": "What to do each time, e.g. 'Check the weather in Austin and summarize it.'"},
         "every_minutes": {"type": "integer", "description": "Repeat interval, 5–10080 minutes."},
         "daily_at": {"type": "string", "description": "A time of day HH:MM (24h, deck local time) for a once-a-day task."},
         "model": {"type": "string", "description": "Model to run it with; default: the one in use."}},
        required=["prompt"]),
    _fn("list_tasks", "The user's scheduled tasks, with next and last run.", {}),
    _fn("cancel_task", "Cancel a scheduled task by id.", {"id": {"type": "string"}}, required=["id"]),
    _fn("home_assistant",
        "Read states or call services on the user's Home Assistant: list entities (optionally "
        "filtered by domain or text), read one entity's state, or call a service such as "
        "light.turn_on with data like {\"entity_id\": \"light.kitchen\", \"brightness\": 128}.",
        {"action": {"type": "string", "enum": ["list", "state", "call"]},
         "entity_id": {"type": "string"},
         "filter": {"type": "string", "description": "For list: a domain (light, switch, sensor…) or text to match."},
         "domain": {"type": "string", "description": "For call: the service domain, e.g. light."},
         "service": {"type": "string", "description": "For call: the service, e.g. turn_on."},
         "data": {"type": "object", "description": "For call: the service data."}},
        required=["action"]),
    _fn("send_email",
        "Send an email through the deck's configured SMTP server. Confirm the recipient and "
        "content with the user before sending.",
        {"to": {"type": "string", "description": "Recipient address(es), comma separated."},
         "subject": {"type": "string"}, "body": {"type": "string", "description": "Plain text."}},
        required=["to", "subject", "body"]),
]
NAMES = [d["function"]["name"] for d in DEFINITIONS]
ACCESS = {d["function"]["name"]: d.get("_access") for d in DEFINITIONS}


# ── helpers ─────────────────────────────────────────────────────────────────

def _client():
    return httpx.Client(timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": UA})


def _num(v, name, lo, hi, default):
    if v is None or v == "":
        return default
    try:
        n = float(v)
    except (TypeError, ValueError):
        raise ToolError(f"{name} must be a number")
    return min(hi, max(lo, n))


def _private_ip(ip: str) -> bool:
    a = ipaddress.ip_address(ip)
    return (a.is_private or a.is_loopback or a.is_link_local or a.is_reserved
            or a.is_multicast or a.is_unspecified)


def _check_public(url: str) -> str:
    """The URL, after checking it points at a public address."""
    u = urlsplit(url.strip())
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ToolError("the URL must start with http:// or https://")
    host = u.hostname
    if host.lower() in ("localhost",) or host.endswith(".local") or host.endswith(".internal"):
        raise ToolError("local addresses are not fetched")
    try:
        infos = socket.getaddrinfo(host, u.port or (443 if u.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise ToolError(f"could not resolve {host}")
    for info in infos:
        if _private_ip(info[4][0]):
            raise ToolError("private, loopback and link-local addresses are not fetched")
    return url.strip()


def _account(user) -> str:
    name = ((user or {}).get("name") or "").strip().lower()
    return name if name and re.match(r"^[a-z0-9][a-z0-9._-]{0,31}$", name) else chats.LOCAL


def _account_dir(kind: str, user) -> Path:
    return config.DATA_DIR / kind / _account(user)


_probes: dict = {}


def _reachable(url: str, every: float = 60.0) -> bool:
    """Whether a local service answers, remembered for a minute so the tool
    list does not probe it on every request."""
    now = time.time()
    hit = _probes.get(url)
    if hit and now - hit[0] < every:
        return hit[1]
    ok = False
    try:
        with httpx.Client(timeout=2.0) as c:
            ok = c.get(url).status_code < 500
    except httpx.HTTPError:
        ok = False
    _probes[url] = (now, ok)
    return ok


# ── get_datetime ────────────────────────────────────────────────────────────

def get_datetime(args: dict) -> dict:
    tz = (args.get("timezone") or "").strip()
    try:
        zone = zoneinfo.ZoneInfo(tz) if tz else None
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
        raise ToolError(f"unknown timezone '{tz}' — use an IANA name like America/New_York")
    now = _dt.datetime.now(zone) if zone else _dt.datetime.now().astimezone()
    return {"iso": now.isoformat(timespec="seconds"), "date": now.date().isoformat(),
            "time": now.strftime("%H:%M"), "weekday": now.strftime("%A"),
            "timezone": str(now.tzinfo), "utc_offset": now.strftime("%z"),
            "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "unix": int(time.time())}


# ── get_weather ─────────────────────────────────────────────────────────────

WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
       45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
       56: "freezing drizzle", 57: "heavy freezing drizzle", 61: "light rain", 63: "rain",
       65: "heavy rain", 66: "freezing rain", 67: "heavy freezing rain", 71: "light snow",
       73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
       82: "violent showers", 85: "light snow showers", 86: "heavy snow showers",
       95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail"}


def _geocode(c: httpx.Client, place: str) -> dict:
    r = c.get("https://geocoding-api.open-meteo.com/v1/search",
              params={"name": place, "count": 1, "language": "en", "format": "json"})
    r.raise_for_status()
    hits = r.json().get("results") or []
    if not hits:
        raise ToolError(f"no place called '{place}' was found — try adding a region or country")
    h = hits[0]
    name = ", ".join(x for x in (h.get("name"), h.get("admin1"), h.get("country")) if x)
    return {"name": name, "latitude": h["latitude"], "longitude": h["longitude"]}


def get_weather(args: dict) -> dict:
    days = int(_num(args.get("days"), "days", 1, 7, 3))
    imperial = (args.get("units") or "metric").lower().startswith("imp")
    with _client() as c:
        if args.get("latitude") is not None and args.get("longitude") is not None:
            lat = _num(args["latitude"], "latitude", -90, 90, None)
            lon = _num(args["longitude"], "longitude", -180, 180, None)
            where = {"name": f"{lat:.3f}, {lon:.3f}", "latitude": lat, "longitude": lon}
        elif (args.get("location") or "").strip():
            where = _geocode(c, args["location"].strip())
        else:
            raise ToolError("give a location, or latitude and longitude")
        params = {"latitude": where["latitude"], "longitude": where["longitude"],
                  "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                             "precipitation,weather_code,wind_speed_10m,wind_gusts_10m",
                  "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                           "precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
                  "timezone": "auto", "forecast_days": days}
        if imperial:
            params.update({"temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                           "precipitation_unit": "inch"})
        r = c.get("https://api.open-meteo.com/v1/forecast", params=params)
        if r.status_code != 200:
            raise ToolError(f"Open-Meteo answered {r.status_code}")
        d = r.json()
    cu, cuu = d.get("current", {}), d.get("current_units", {})
    out = {"location": where["name"], "timezone": d.get("timezone"),
           "units": "imperial" if imperial else "metric",
           "current": {
               "time": cu.get("time"), "conditions": WMO.get(cu.get("weather_code"), "unknown"),
               "temperature": f"{cu.get('temperature_2m')}{cuu.get('temperature_2m', '')}",
               "feels_like": f"{cu.get('apparent_temperature')}{cuu.get('apparent_temperature', '')}",
               "humidity": f"{cu.get('relative_humidity_2m')}%",
               "precipitation": f"{cu.get('precipitation')}{cuu.get('precipitation', '')}",
               "wind": f"{cu.get('wind_speed_10m')}{cuu.get('wind_speed_10m', '')}"
                       f" (gusts {cu.get('wind_gusts_10m')})"},
           "daily": []}
    dl, du = d.get("daily", {}), d.get("daily_units", {})
    for i, day in enumerate(dl.get("time", [])):
        out["daily"].append({
            "date": day, "conditions": WMO.get((dl.get("weather_code") or [None] * 99)[i], "unknown"),
            "high": f"{dl['temperature_2m_max'][i]}{du.get('temperature_2m_max', '')}",
            "low": f"{dl['temperature_2m_min'][i]}{du.get('temperature_2m_min', '')}",
            "precipitation": f"{dl['precipitation_sum'][i]}{du.get('precipitation_sum', '')}",
            "precipitation_chance": f"{dl['precipitation_probability_max'][i]}%",
            "wind_max": f"{dl['wind_speed_10m_max'][i]}{du.get('wind_speed_10m_max', '')}"})
    return out


# ── web_search ──────────────────────────────────────────────────────────────

def _provider():
    want = (getattr(config, "SEARCH_PROVIDER", "auto") or "auto").lower()
    key = (getattr(config, "BRAVE_API_KEY", "") or "").strip()
    sx = (getattr(config, "SEARXNG_URL", "") or "").strip().rstrip("/")
    if want == "brave" or (want == "auto" and key):
        if not key:
            raise ToolError("web search is set to Brave but no Brave API key is set "
                            "(Settings → Chat Tools)")
        return "brave", key
    if want == "searxng" or (want == "auto" and sx):
        if not sx:
            raise ToolError("web search is set to SearXNG but no SearXNG URL is set "
                            "(Settings → Chat Tools)")
        return "searxng", sx
    raise ToolError("no search provider is configured: set a Brave API key or a SearXNG URL "
                    "under Settings → Chat Tools")


def web_search(args: dict) -> dict:
    q = (args.get("query") or "").strip()
    if not q:
        raise ToolError("query is required")
    n = int(_num(args.get("count"), "count", 1, 10, 5))
    kind, val = _provider()
    results = []
    with _client() as c:
        if kind == "brave":
            r = c.get("https://api.search.brave.com/res/v1/web/search",
                      params={"q": q, "count": n, "text_decorations": "false"},
                      headers={"Accept": "application/json", "X-Subscription-Token": val})
            if r.status_code in (401, 403, 422):
                raise ToolError("Brave Search rejected the API key (Settings → Chat Tools)")
            if r.status_code == 429:
                raise ToolError("Brave Search rate limit reached; try again shortly")
            if r.status_code != 200:
                raise ToolError(f"Brave Search answered {r.status_code}")
            for w in (r.json().get("web") or {}).get("results") or []:
                results.append({"title": w.get("title"), "url": w.get("url"),
                                "snippet": html.unescape(re.sub(r"<[^>]+>", "", w.get("description") or "")),
                                "age": w.get("age")})
        else:
            r = c.get(f"{val}/search", params={"q": q, "format": "json"})
            if r.status_code != 200:
                raise ToolError(f"SearXNG at {val} answered {r.status_code} "
                                "(its settings.yaml must allow the json format)")
            for w in (r.json().get("results") or [])[:n]:
                results.append({"title": w.get("title"), "url": w.get("url"),
                                "snippet": w.get("content"), "engine": w.get("engine")})
    return {"query": q, "provider": kind, "results": results[:n]}


# ── fetch_url ───────────────────────────────────────────────────────────────

_SKIP = {"script", "style", "noscript", "template", "svg", "canvas", "iframe",
         "nav", "header", "footer", "aside", "form", "button"}
_BLOCK = {"p", "div", "br", "li", "ul", "ol", "tr", "td", "th", "table", "h1", "h2", "h3",
          "h4", "h5", "h6", "section", "article", "pre", "blockquote", "hr", "dd", "dt", "main"}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.title, self._in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK:
            self.out.append("\n")
        if tag.startswith("h") and len(tag) == 2 and tag[1].isdigit() and not self.skip:
            self.out.append("\n" + "#" * int(tag[1]) + " ")

    def handle_endtag(self, tag):
        if tag in _SKIP and self.skip:
            self.skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.out.append(data)


def _readable(raw: str) -> tuple:
    p = _Text()
    p.feed(raw)
    text = "".join(p.out)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return p.title.strip(), text


def fetch_url(args: dict) -> dict:
    url = _check_public(str(args.get("url") or ""))
    limit = int(_num(args.get("max_chars"), "max_chars", 200, 40000, 8000))
    with _client() as c:
        try:
            with c.stream("GET", url, headers={"Accept": "text/html,application/xhtml+xml,"
                                                         "text/plain;q=0.9,*/*;q=0.5"}) as r:
                if r.status_code != 200:
                    raise ToolError(f"{url} answered {r.status_code}")
                ctype = (r.headers.get("content-type") or "").lower()
                if not any(t in ctype for t in ("text/", "html", "xml", "json")):
                    raise ToolError(f"not a text page ({ctype.split(';')[0] or 'unknown type'})")
                buf, cap = b"", 2 * 1024 * 1024
                for chunk in r.iter_bytes():
                    buf += chunk
                    if len(buf) > cap:
                        break
                enc = r.encoding or "utf-8"
        except httpx.HTTPError as e:
            raise ToolError(f"fetch failed: {e.__class__.__name__}: {e}")
    try:
        raw = buf.decode(enc, errors="replace")
    except LookupError:
        raw = buf.decode("utf-8", errors="replace")
    if "html" in ctype or raw.lstrip()[:1] == "<":
        title, text = _readable(raw)
    else:
        title, text = "", raw.strip()
    cut = len(text) > limit
    return {"url": str(url), "title": title, "chars": len(text), "truncated": cut,
            "text": text[:limit] + ("\n…[cut]" if cut else "")}


# ── news ────────────────────────────────────────────────────────────────────

def news(args: dict) -> dict:
    q = (args.get("query") or "").strip()
    if not q:
        raise ToolError("query is required")
    n = int(_num(args.get("count"), "count", 1, 10, 5))
    kind, val = _provider()
    out = []
    with _client() as c:
        if kind == "brave":
            r = c.get("https://api.search.brave.com/res/v1/news/search",
                      params={"q": q, "count": n, "freshness": "pw"},
                      headers={"Accept": "application/json", "X-Subscription-Token": val})
            if r.status_code in (401, 403, 422):
                raise ToolError("Brave Search rejected the API key (Settings → Chat Tools)")
            if r.status_code != 200:
                raise ToolError(f"Brave News answered {r.status_code}")
            for w in r.json().get("results") or []:
                out.append({"title": w.get("title"), "url": w.get("url"), "age": w.get("age"),
                            "source": (w.get("meta_url") or {}).get("hostname"),
                            "snippet": html.unescape(re.sub(r"<[^>]+>", "", w.get("description") or ""))})
        else:
            r = c.get(f"{val}/search", params={"q": q, "format": "json", "categories": "news"})
            if r.status_code != 200:
                raise ToolError(f"SearXNG at {val} answered {r.status_code}")
            for w in (r.json().get("results") or [])[:n]:
                out.append({"title": w.get("title"), "url": w.get("url"),
                            "published": w.get("publishedDate"), "source": w.get("engine"),
                            "snippet": w.get("content")})
    return {"query": q, "provider": kind, "articles": out[:n]}


# ── wikipedia ───────────────────────────────────────────────────────────────

def wikipedia(args: dict) -> dict:
    q = (args.get("query") or "").strip()
    if not q:
        raise ToolError("query is required")
    lang = re.sub(r"[^a-z-]", "", (args.get("language") or "en").lower()) or "en"
    with _client() as c:
        r = c.get(f"https://{lang}.wikipedia.org/w/api.php",
                  params={"action": "query", "list": "search", "srsearch": q, "srlimit": 3,
                          "format": "json", "utf8": 1})
        if r.status_code != 200:
            raise ToolError(f"Wikipedia answered {r.status_code}")
        hits = (r.json().get("query") or {}).get("search") or []
        if not hits:
            raise ToolError(f"Wikipedia has no article matching '{q}'")
        title = hits[0]["title"]
        r = c.get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}")
        if r.status_code != 200:
            raise ToolError(f"Wikipedia summary answered {r.status_code}")
        d = r.json()
    return {"title": d.get("title"), "description": d.get("description"),
            "summary": d.get("extract"),
            "url": ((d.get("content_urls") or {}).get("desktop") or {}).get("page"),
            "other_matches": [h["title"] for h in hits[1:]]}


# ── calculate ───────────────────────────────────────────────────────────────

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {n: getattr(math, n) for n in ("sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
                                        "atan2", "sinh", "cosh", "tanh", "log", "log10", "log2",
                                        "exp", "floor", "ceil", "factorial", "gcd", "degrees",
                                        "radians", "hypot", "comb", "perm", "isqrt")}
_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max, "pow": pow, "sum": sum})
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}


def _ev(node):
    if isinstance(node, ast.Expression):
        return _ev(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _ev(node.left), _ev(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 10000:
            raise ToolError("exponent too large")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        if node.keywords:
            raise ToolError("keyword arguments are not supported")
        args = [_ev(a) for a in node.args]
        if node.func.id == "factorial" and (args and args[0] > 5000):
            raise ToolError("factorial argument too large")
        return _FUNCS[node.func.id](*args)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_ev(x) for x in node.elts]
    raise ToolError("only numbers, operators, parentheses and math functions are allowed")


def calculate(args: dict) -> dict:
    expr = (args.get("expression") or "").strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    if not expr:
        raise ToolError("expression is required")
    if len(expr) > 500:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
        val = _ev(tree)
    except ToolError:
        raise
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        raise ToolError(f"could not evaluate: {e}")
    if isinstance(val, float) and val.is_integer() and abs(val) < 1e15:
        val = int(val)
    return {"expression": expr, "result": val if isinstance(val, (int, list)) else float(val),
            "formatted": f"{val:,}" if isinstance(val, int) else (f"{val:,.6g}" if isinstance(val, float) else str(val))}


# ── unit_convert ────────────────────────────────────────────────────────────

# each family maps unit → factor to its base unit; temperature is special
_UNITS = {
    "length": {"m": 1, "meter": 1, "meters": 1, "metre": 1, "metres": 1, "km": 1000, "kilometer": 1000,
               "kilometers": 1000, "cm": .01, "centimeter": .01, "centimeters": .01, "mm": .001,
               "millimeter": .001, "millimeters": .001, "um": 1e-6, "micrometer": 1e-6, "nm": 1e-9,
               "mi": 1609.344, "mile": 1609.344, "miles": 1609.344, "yd": .9144, "yard": .9144,
               "yards": .9144, "ft": .3048, "foot": .3048, "feet": .3048, "in": .0254, "inch": .0254,
               "inches": .0254, "nmi": 1852, "nautical mile": 1852, "nautical miles": 1852,
               "ly": 9.4607e15, "light year": 9.4607e15, "light years": 9.4607e15, "au": 1.495978707e11},
    "mass": {"kg": 1, "kilogram": 1, "kilograms": 1, "g": .001, "gram": .001, "grams": .001,
             "mg": 1e-6, "milligram": 1e-6, "milligrams": 1e-6, "t": 1000, "tonne": 1000, "tonnes": 1000,
             "metric ton": 1000, "lb": .45359237, "lbs": .45359237, "pound": .45359237, "pounds": .45359237,
             "oz": .028349523125, "ounce": .028349523125, "ounces": .028349523125, "st": 6.35029318,
             "stone": 6.35029318, "ton": 907.18474, "tons": 907.18474},
    "volume": {"l": 1, "liter": 1, "liters": 1, "litre": 1, "litres": 1, "ml": .001, "milliliter": .001,
               "milliliters": .001, "m3": 1000, "cubic meter": 1000, "cubic meters": 1000,
               "gal": 3.785411784, "gallon": 3.785411784, "gallons": 3.785411784, "qt": .946352946,
               "quart": .946352946, "quarts": .946352946, "pt": .473176473, "pint": .473176473,
               "pints": .473176473, "cup": .2365882365, "cups": .2365882365, "fl oz": .0295735295625,
               "floz": .0295735295625, "fluid ounce": .0295735295625, "fluid ounces": .0295735295625,
               "tbsp": .01478676478125, "tablespoon": .01478676478125, "tablespoons": .01478676478125,
               "tsp": .00492892159375, "teaspoon": .00492892159375, "teaspoons": .00492892159375,
               "imperial gallon": 4.54609, "uk gallon": 4.54609},
    "area": {"m2": 1, "square meter": 1, "square meters": 1, "km2": 1e6, "square kilometer": 1e6,
             "square kilometers": 1e6, "ha": 1e4, "hectare": 1e4, "hectares": 1e4, "acre": 4046.8564224,
             "acres": 4046.8564224, "ft2": .09290304, "square foot": .09290304, "square feet": .09290304,
             "sqft": .09290304, "mi2": 2589988.110336, "square mile": 2589988.110336,
             "square miles": 2589988.110336, "cm2": 1e-4, "in2": 6.4516e-4, "square inch": 6.4516e-4,
             "square inches": 6.4516e-4},
    "speed": {"m/s": 1, "mps": 1, "km/h": 1 / 3.6, "kph": 1 / 3.6, "kmh": 1 / 3.6, "mph": .44704,
              "mi/h": .44704, "kn": .514444, "knot": .514444, "knots": .514444, "ft/s": .3048, "fps": .3048},
    "time": {"s": 1, "sec": 1, "second": 1, "seconds": 1, "ms": .001, "millisecond": .001,
             "milliseconds": .001, "min": 60, "minute": 60, "minutes": 60, "h": 3600, "hr": 3600,
             "hour": 3600, "hours": 3600, "d": 86400, "day": 86400, "days": 86400, "wk": 604800,
             "week": 604800, "weeks": 604800, "year": 31557600, "years": 31557600, "yr": 31557600},
    "data": {"b": 1, "byte": 1, "bytes": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12, "pb": 1e15,
             "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3, "tib": 1024 ** 4, "pib": 1024 ** 5,
             "bit": .125, "bits": .125, "kbit": 125, "mbit": 125e3, "gbit": 125e6, "kilobyte": 1e3,
             "megabyte": 1e6, "gigabyte": 1e9, "terabyte": 1e12, "kilobytes": 1e3, "megabytes": 1e6,
             "gigabytes": 1e9, "terabytes": 1e12},
    "energy": {"j": 1, "joule": 1, "joules": 1, "kj": 1000, "kilojoule": 1000, "kilojoules": 1000,
               "cal": 4.184, "calorie": 4.184, "calories": 4.184, "kcal": 4184, "kilocalorie": 4184,
               "kilocalories": 4184, "wh": 3600, "watt hour": 3600, "watt hours": 3600, "kwh": 3.6e6,
               "kilowatt hour": 3.6e6, "kilowatt hours": 3.6e6, "btu": 1055.05585, "ev": 1.602176634e-19},
    "power": {"w": 1, "watt": 1, "watts": 1, "kw": 1000, "kilowatt": 1000, "kilowatts": 1000,
              "mw": 1e6, "megawatt": 1e6, "hp": 745.69987158, "horsepower": 745.69987158},
    "pressure": {"pa": 1, "pascal": 1, "kpa": 1000, "bar": 1e5, "mbar": 100, "atm": 101325,
                 "psi": 6894.757293168, "mmhg": 133.322387415, "torr": 133.322387415, "inhg": 3386.389},
}
_TEMP = {"c": "c", "celsius": "c", "°c": "c", "degc": "c", "f": "f", "fahrenheit": "f", "°f": "f",
         "degf": "f", "k": "k", "kelvin": "k"}


def _to_c(v, u):
    return v if u == "c" else (v - 32) * 5 / 9 if u == "f" else v - 273.15


def _from_c(v, u):
    return v if u == "c" else v * 9 / 5 + 32 if u == "f" else v + 273.15


def unit_convert(args: dict) -> dict:
    v = _num(args.get("value"), "value", -1e300, 1e300, None)
    if v is None:
        raise ToolError("value is required")
    a = (args.get("from_unit") or "").strip().lower().replace("sq ", "square ")
    b = (args.get("to_unit") or "").strip().lower().replace("sq ", "square ")
    if not a or not b:
        raise ToolError("from_unit and to_unit are required")
    if a in _TEMP and b in _TEMP:
        out = _from_c(_to_c(v, _TEMP[a]), _TEMP[b])
        return {"value": v, "from": a, "to": b, "result": round(out, 4)}
    for fam, table in _UNITS.items():
        if a in table and b in table:
            out = v * table[a] / table[b]
            return {"value": v, "from": a, "to": b, "result": float(f"{out:.10g}"), "family": fam}
    fa = [f for f, t in _UNITS.items() if a in t] + (["temperature"] if a in _TEMP else [])
    fb = [f for f, t in _UNITS.items() if b in t] + (["temperature"] if b in _TEMP else [])
    if not fa or not fb:
        raise ToolError(f"unknown unit: {a if not fa else b}")
    raise ToolError(f"cannot convert {fa[0]} to {fb[0]}")


# ── currency ────────────────────────────────────────────────────────────────

def currency(args: dict) -> dict:
    amt = _num(args.get("amount"), "amount", -1e15, 1e15, None)
    if amt is None:
        raise ToolError("amount is required")
    a = (args.get("from_currency") or "").strip().upper()
    b = (args.get("to_currency") or "").strip().upper()
    if not re.match(r"^[A-Z]{3}$", a) or not re.match(r"^[A-Z]{3}$", b):
        raise ToolError("currencies are three-letter ISO codes, e.g. USD, EUR, GBP, JPY")
    if a == b:
        return {"amount": amt, "from": a, "to": b, "result": amt, "rate": 1.0}
    with _client() as c:
        r = c.get("https://api.frankfurter.app/latest", params={"amount": amt, "from": a, "to": b})
        if r.status_code == 404:
            raise ToolError(f"no rate for {a} → {b} (the ECB list covers ~30 major currencies)")
        if r.status_code != 200:
            raise ToolError(f"frankfurter.app answered {r.status_code}")
        d = r.json()
    res = (d.get("rates") or {}).get(b)
    if res is None:
        raise ToolError(f"no rate for {b}")
    return {"amount": amt, "from": a, "to": b, "result": res,
            "rate": round(res / amt, 6) if amt else None, "date": d.get("date")}


# ── run_python ──────────────────────────────────────────────────────────────

SANDBOX_IMAGE = "neuraldeck-sandbox"
_build_lock = threading.Lock()


def _docker() -> str:
    d = shutil.which("docker")
    if not d:
        raise ToolError("Docker is not installed on the deck; run_python needs it")
    return d


def _sandbox_ready(build: bool) -> bool:
    docker = _docker()
    if subprocess.run([docker, "image", "inspect", SANDBOX_IMAGE], capture_output=True).returncode == 0:
        return True
    if not build:
        return False
    with _build_lock:
        if subprocess.run([docker, "image", "inspect", SANDBOX_IMAGE], capture_output=True).returncode == 0:
            return True
        ctx = Path(__file__).parent / "sandbox"
        r = subprocess.run([docker, "build", "-q", "-t", SANDBOX_IMAGE, str(ctx)],
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise ToolError("could not build the sandbox image: " + (r.stderr or r.stdout)[-400:])
    return True


def run_python(args: dict) -> dict:
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code is required")
    limit = int(_num(args.get("timeout_s"), "timeout_s", 1, 120, 30))
    _sandbox_ready(build=True)
    name = "nd-sbx-" + secrets.token_hex(4)
    cmd = [_docker(), "run", "--rm", "-i", "--name", name, "--network", "none",
           "--cpus", "2", "--memory", "1g", "--memory-swap", "1g", "--pids-limit", "256",
           "--read-only", "--tmpfs", "/tmp:rw,size=256m,uid=65534",
           "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
           "-u", "65534", SANDBOX_IMAGE, "python", "-I", "-"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, input=code, capture_output=True, text=True, timeout=limit + 5)
    except subprocess.TimeoutExpired as e:
        subprocess.run([_docker(), "kill", name], capture_output=True)
        out = (e.stdout or b"")
        return {"timed_out": True, "seconds": limit,
                "stdout": (out.decode() if isinstance(out, bytes) else out)[-6000:],
                "stderr": f"killed after {limit} s"}
    cut = lambda t: (t[-12000:] if len(t) > 12000 else t)
    return {"exit_code": r.returncode, "seconds": round(time.time() - t0, 2),
            "stdout": cut(r.stdout), "stderr": cut(r.stderr)}


# ── read_github ─────────────────────────────────────────────────────────────

def _gh(c: httpx.Client, url: str, accept="application/vnd.github+json", **params):
    headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
    tok = (getattr(config, "GITHUB_TOKEN", "") or "").strip()
    if tok:
        headers["Authorization"] = "Bearer " + tok
    r = c.get(url, headers=headers, params=params or None)
    if r.status_code == 404:
        raise ToolError("not found on GitHub (check owner/name, the path and the ref)")
    if r.status_code == 403 and "rate limit" in r.text.lower():
        raise ToolError("GitHub rate limit reached; add a GitHub token under Settings → Chat Tools")
    if r.status_code != 200:
        raise ToolError(f"GitHub answered {r.status_code}")
    return r


def read_github(args: dict) -> dict:
    repo = (args.get("repo") or "").strip().strip("/")
    repo = re.sub(r"^https?://github\.com/", "", repo)
    if not re.match(r"^[\w.-]+/[\w.-]+$", repo):
        raise ToolError("repo is owner/name, e.g. ggml-org/llama.cpp")
    action = (args.get("action") or "").strip().lower()
    path = (args.get("path") or "").strip().strip("/")
    ref = (args.get("ref") or "").strip()
    base = f"https://api.github.com/repos/{repo}"
    with _client() as c:
        if action == "tree":
            r = _gh(c, f"{base}/contents/{quote(path)}", **({"ref": ref} if ref else {}))
            d = r.json()
            if isinstance(d, dict):
                return {"repo": repo, "path": path, "type": "file", "size": d.get("size"),
                        "note": "that path is a file; use action=file to read it"}
            return {"repo": repo, "path": path or "/", "entries": [
                {"name": x["name"], "type": x["type"], "size": x.get("size")} for x in d][:400]}
        if action in ("file", "readme"):
            url = f"{base}/readme" if action == "readme" else f"{base}/contents/{quote(path)}"
            if action == "file" and not path:
                raise ToolError("path is required for action=file")
            r = _gh(c, url, accept="application/vnd.github.raw+json", **({"ref": ref} if ref else {}))
            text = r.text
            cut = len(text) > 60000
            return {"repo": repo, "path": path or "README", "chars": len(text), "truncated": cut,
                    "content": text[:60000] + ("\n…[cut]" if cut else "")}
        if action in ("issue", "pull"):
            n = args.get("number")
            try:
                n = int(n)
            except (TypeError, ValueError):
                raise ToolError("number is required for issue and pull")
            d = _gh(c, f"{base}/issues/{n}").json()
            out = {"repo": repo, "number": n, "title": d.get("title"), "state": d.get("state"),
                   "author": (d.get("user") or {}).get("login"), "created": d.get("created_at"),
                   "url": d.get("html_url"), "labels": [l.get("name") for l in d.get("labels") or []],
                   "body": (d.get("body") or "")[:20000], "comments": []}
            if d.get("pull_request"):
                pr = _gh(c, f"{base}/pulls/{n}").json()
                out.update({"is_pull_request": True, "merged": pr.get("merged"),
                            "base": (pr.get("base") or {}).get("ref"), "head": (pr.get("head") or {}).get("ref"),
                            "changed_files": pr.get("changed_files"), "additions": pr.get("additions"),
                            "deletions": pr.get("deletions")})
            for cm in _gh(c, f"{base}/issues/{n}/comments", per_page=30).json():
                out["comments"].append({"author": (cm.get("user") or {}).get("login"),
                                        "created": cm.get("created_at"), "body": (cm.get("body") or "")[:4000]})
            return out
    raise ToolError("action must be tree, file, readme, issue or pull")


# ── search_chats ────────────────────────────────────────────────────────────

def search_chats(args: dict, user=None) -> dict:
    q = (args.get("query") or "").strip()
    if not q:
        raise ToolError("query is required")
    n = int(_num(args.get("limit"), "limit", 1, 10, 5))
    words = [w for w in re.split(r"\s+", q.lower()) if w]
    hits = []
    for c in chats.list_convs(_account(user)):
        lines, score = [], 0
        for m in c.get("messages") or []:
            if m.get("role") not in ("user", "assistant"):
                continue
            text = str(m.get("content") or "")
            for line in text.splitlines():
                low = line.lower()
                k = sum(1 for w in words if w in low)
                if k:
                    score += k
                    if len(lines) < 6:
                        lines.append(f"{m['role']}: {line.strip()[:240]}")
        title = str(c.get("title") or "")
        score += 3 * sum(1 for w in words if w in title.lower())
        if score:
            hits.append((score, {"title": title or "untitled", "id": c.get("id"),
                                 "updated": _dt.datetime.fromtimestamp((c.get("updated") or 0) / 1000).isoformat(timespec="minutes"),
                                 "matches": lines}))
    hits.sort(key=lambda h: -h[0])
    return {"query": q, "chats": [h[1] for h in hits[:n]]}


# ── documents ───────────────────────────────────────────────────────────────

DOC_EXT = {".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml",
           ".py", ".js", ".ts", ".html", ".css", ".sh", ".c", ".h", ".cpp", ".rs", ".go", ".java",
           ".log", ".ini", ".cfg", ".xml", ".pdf", ".tex", ".sql"}


def _doc_roots(user) -> list:
    return [("mine", _account_dir("documents", user)), ("shared", config.DATA_DIR / "documents" / "_shared")]


def _doc_files(user) -> list:
    out = []
    for label, root in _doc_roots(user):
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in DOC_EXT and not p.name.startswith("."):
                out.append((label, root, p))
    return out


def _doc_text(p: Path) -> str:
    if p.suffix.lower() == ".pdf":
        exe = shutil.which("pdftotext")
        if not exe:
            raise ToolError("reading PDFs needs pdftotext (poppler-utils) on the deck")
        r = subprocess.run([exe, "-layout", "-q", str(p), "-"], capture_output=True, text=True, timeout=120)
        return r.stdout
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise ToolError(f"could not read {p.name}: {e}")


def list_documents(args: dict, user=None) -> dict:
    mine, shared = _doc_roots(user)[0][1], _doc_roots(user)[1][1]
    files = [{"path": f"{label}/{p.relative_to(root)}", "size": p.stat().st_size,
              "modified": _dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes")}
             for label, root, p in _doc_files(user)]
    return {"files": files[:500], "count": len(files),
            "folders": {"mine": str(mine), "shared": str(shared)},
            "note": "" if files else "no documents yet — put files in the folders above"}


def _doc_resolve(user, path: str) -> Path:
    path = (path or "").strip().lstrip("/")
    label, _, rel = path.partition("/")
    roots = dict((l, r) for l, r in _doc_roots(user))
    if label not in roots or not rel:
        raise ToolError("path is as list_documents gives it: mine/<file> or shared/<file>")
    root = roots[label].resolve()
    p = (root / rel).resolve()
    if root not in p.parents or not p.is_file() or p.suffix.lower() not in DOC_EXT:
        raise ToolError(f"no document at {path}")
    return p


def read_document(args: dict, user=None) -> dict:
    p = _doc_resolve(user, args.get("path"))
    off = int(_num(args.get("offset"), "offset", 0, 1e9, 0))
    limit = int(_num(args.get("max_chars"), "max_chars", 200, 40000, 8000))
    text = _doc_text(p)
    piece = text[off:off + limit]
    return {"path": args.get("path"), "chars": len(text), "offset": off,
            "next_offset": off + limit if off + limit < len(text) else None, "text": piece}


def search_documents(args: dict, user=None) -> dict:
    q = (args.get("query") or "").strip().lower()
    if not q:
        raise ToolError("query is required")
    out = []
    for label, root, p in _doc_files(user):
        try:
            text = _doc_text(p)
        except ToolError:
            continue
        lines = [f"{i + 1}: {l.strip()[:240]}" for i, l in enumerate(text.splitlines()) if q in l.lower()]
        if lines:
            out.append({"path": f"{label}/{p.relative_to(root)}", "matches": len(lines), "lines": lines[:8]})
        if len(out) >= 20:
            break
    return {"query": q, "files": out}


# ── the deck itself ─────────────────────────────────────────────────────────

def _ctx(key):
    fn = CONTEXT.get(key)
    if fn is None:
        raise ToolError("this tool is not available: the deck did not register it")
    return fn


def _await(coro, timeout=600):
    """Run a coroutine on the deck's event loop from this worker thread."""
    import asyncio
    loop = CONTEXT.get("loop")
    if loop is None:
        raise ToolError("the deck's event loop is not available")
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)


def deck_status(args: dict) -> dict:
    snap = _ctx("snapshot")() or {}
    mem, gpu, cpu = snap.get("mem") or {}, snap.get("gpu") or {}, snap.get("cpu") or {}
    insts = []
    for i in snap.get("llama_instances") or []:
        insts.append({"model": i.get("alias"), "port": i.get("port"), "backend": i.get("backend"),
                      "kind": i.get("kind"), "ctx": i.get("ctx_total"), "slots": i.get("slots"),
                      "spec": i.get("spec"), "kv_used_pct": i.get("kv_pct")})
    llama = snap.get("llama") or {}
    return {"models_loaded": insts,
            "ram": {"used_gb": mem.get("used_gb"), "total_gb": mem.get("total_gb"),
                    "free_gb": round((mem.get("total_gb") or 0) - (mem.get("used_gb") or 0), 1)},
            "gpu": {"name": gpu.get("name"), "util_pct": gpu.get("util"), "temp_c": gpu.get("temp"),
                    "power_w": gpu.get("power"), "vram_used_gb": gpu.get("vram_used_gb"),
                    "vram_total_gb": gpu.get("vram_total_gb")},
            "cpu": {"util_pct": cpu.get("util"), "temp_c": cpu.get("temp")},
            "inference": {k: llama.get(k) for k in ("status", "model", "prompt_tps", "decode_tps",
                                                    "requests") if k in llama},
            "launch_in_progress": (snap.get("launch") or {}).get("running"),
            "services": {k: (v.get("running") if isinstance(v, dict) else v)
                         for k, v in (snap.get("services") or {}).items()},
            "backends": list((getattr(config, "BACKENDS", {}) or {}).keys()),
            "default_backend": getattr(config, "DEFAULT_BACKEND", None)}


def list_models(args: dict) -> dict:
    found = _ctx("models")()
    out = []
    for m in found:
        out.append({"name": m.get("name"), "format": m.get("format"), "quant": m.get("quant"),
                    "size_gb": round((m.get("vram") or m.get("size_gb") or 0), 1) or None,
                    "vision": bool(m.get("mmproj_path") or m.get("vision")),
                    "params": m.get("params") or m.get("n_params")})
    return {"models": out, "count": len(out)}


def launch_model(args: dict) -> dict:
    model = (args.get("model") or "").strip()
    if not model:
        raise ToolError("model is required")
    ctx = int(_num(args.get("ctx"), "ctx", 1024, 1048576, getattr(config, "CTX_DEFAULT", 32768)))
    backend = (args.get("backend") or "").strip() or getattr(config, "DEFAULT_BACKEND", None)
    try:
        res = _await(_ctx("launch")(model_name=model, backend=backend, ctx=ctx, slots=1, thinking="off",
                                    spec="auto", replace=bool(args.get("replace")), force=False), timeout=900)
    except (LookupError, FileNotFoundError, MemoryError, RuntimeError, ValueError) as e:
        raise ToolError(f"launch refused: {e}")
    except TimeoutError:
        raise ToolError("the launch is still running after 15 minutes; check deck_status")
    return {"launched": model, "backend": backend, "ctx": ctx, "result": res}


def stop_model(args: dict) -> dict:
    want = (args.get("model") or "").strip()
    if not want:
        raise ToolError("model (a name or a port) is required")
    for i in _ctx("instances")():
        if str(i.get("port")) == want or (i.get("alias") or "") == want:
            if not _ctx("stop_pid")(i["pid"]):
                raise ToolError(f"the process on :{i['port']} did not stop")
            return {"stopped": i.get("alias"), "port": i.get("port")}
    raise ToolError(f"nothing running is called {want}")


BENCH_PROMPT = ("Explain, in about 250 words, how a lithium-ion battery charges and discharges, "
                "covering the role of the anode, cathode, and electrolyte.")


def benchmark(args: dict) -> dict:
    model = (args.get("model") or "").strip()
    if not model:
        raise ToolError("model is required")
    insts = _ctx("instances")()
    inst = next((i for i in insts if i.get("alias") == model), None)
    if inst is None:
        raise ToolError(f"{model} is not loaded; launch it first (list_models / launch_model)")
    body = {"model": model, "messages": [{"role": "user", "content": BENCH_PROMPT}],
            "temperature": 0, "max_tokens": 320, "stream": False}
    t0 = time.time()
    with httpx.Client(timeout=httpx.Timeout(600.0, connect=5.0)) as c:
        r = c.post(config.PROXY_CHAT, json=body, headers=_ctx("proxy_headers")())
    wall = time.time() - t0
    if r.status_code != 200:
        raise ToolError(f"the proxy answered {r.status_code}: {r.text[:200]}")
    d = r.json()
    tm, usage = d.get("timings") or {}, d.get("usage") or {}
    rec = {"model": model, "kind": "std", "backend": inst.get("backend"), "backend_kind": inst.get("kind"),
           "max_tokens": 320, "temp": 0, "wall_s": wall,
           "prompt_n": tm.get("prompt_n", usage.get("prompt_tokens")),
           "predicted_n": tm.get("predicted_n", usage.get("completion_tokens")),
           "prompt_tps": tm.get("prompt_per_second"), "decode_tps": tm.get("predicted_per_second"),
           "ttft_ms": tm.get("prompt_ms"), "draft_n": tm.get("draft_n"), "draft_acc": tm.get("draft_n_accepted")}
    if rec["decode_tps"] is None and rec["predicted_n"]:
        rec["decode_tps"] = rec["predicted_n"] / wall
    saved = _ctx("bench_add")(rec, insts)
    return {"model": model, "backend": inst.get("backend"), "wall_s": round(wall, 2),
            "prompt_tokens": rec["prompt_n"], "generated_tokens": rec["predicted_n"],
            "prefill_tps": rec["prompt_tps"], "decode_tps": round(rec["decode_tps"], 2) if rec["decode_tps"] else None,
            "ttft_ms": rec["ttft_ms"], "recorded_as": saved.get("id"),
            "note": "recorded in the Benchmarks tab" + ("" if tm else "; timings estimated from the wall clock")}


# ── generated files: images and audio the chat can show ─────────────────────

def _generated(user, ext: str) -> tuple:
    d = _account_dir("generated", user)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    name = _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3) + ext
    return d / name, f"/api/generated/{name}"


def _comfy_workflow() -> dict:
    path = (getattr(config, "COMFY_WORKFLOW", "") or "").strip()
    if not path:
        raise ToolError("no ComfyUI workflow file is set (Settings → Chat Tools)")
    try:
        return json.loads(Path(os.path.expanduser(path)).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ToolError(f"could not read the ComfyUI workflow: {e}")


def _sd_base():
    base = (getattr(config, "SD_URL", "") or "").rstrip("/")
    return base if base and _reachable(base + "/v1/models") else None


def _sd_generate(prompt, args, user) -> dict:
    """stable-diffusion.cpp's sd-server, through its OpenAI-shaped API."""
    w = int(_num(args.get("width"), "width", 64, 2048, 1024))
    h = int(_num(args.get("height"), "height", 64, 2048, 1024))
    seed = int(_num(args.get("seed"), "seed", -1, 2 ** 31 - 1, -1))
    extra = {"seed": seed}
    neg = (args.get("negative") or "").strip()
    if neg:
        extra["negative_prompt"] = neg
    body = {"prompt": f"{prompt} <sd_cpp_extra_args>{json.dumps(extra)}</sd_cpp_extra_args>",
            "n": 1, "size": f"{w}x{h}", "output_format": "png"}
    with httpx.Client(timeout=httpx.Timeout(900.0, connect=5.0)) as c:
        r = c.post(_sd_base() + "/v1/images/generations", json=body)
    if r.status_code == 503:
        raise ToolError("sd-server has no model loaded yet; give it a moment or check its log")
    if r.status_code != 200:
        raise ToolError(f"sd-server answered {r.status_code}: {r.text[:300]}")
    data = (r.json().get("data") or [{}])[0]
    b64 = data.get("b64_json")
    if not b64:
        raise ToolError("sd-server returned no image")
    import base64
    path, url = _generated(user, ".png")
    path.write_bytes(base64.b64decode(b64))
    return {"url": url, "markdown": f"![{prompt[:60]}]({url})", "size": f"{w}x{h}",
            "seed": seed if seed >= 0 else "random", "backend": "sd-server",
            "note": "include the markdown in your reply to show the image"}


def generate_image(args: dict, user=None) -> dict:
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        raise ToolError("prompt is required")
    if _sd_base():
        return _sd_generate(prompt, args, user)
    wf = _comfy_workflow()
    fill = {"{prompt}": prompt, "{negative}": (args.get("negative") or "").strip(),
            "{seed}": str(int(_num(args.get("seed"), "seed", 0, 2 ** 53, secrets.randbelow(2 ** 32)))),
            "{width}": str(int(_num(args.get("width"), "width", 64, 4096, 1024))),
            "{height}": str(int(_num(args.get("height"), "height", 64, 4096, 1024)))}
    text = json.dumps(wf)
    for k, v in fill.items():
        text = text.replace(f'"{k}"', json.dumps(int(v)) if k in ("{seed}", "{width}", "{height}") else json.dumps(v))
        text = text.replace(k, v.replace('"', '\\"'))
    wf = json.loads(text)
    base = getattr(config, "COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
    cid = secrets.token_hex(8)
    with httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0)) as c:
        r = c.post(f"{base}/prompt", json={"prompt": wf, "client_id": cid})
        if r.status_code != 200:
            raise ToolError(f"ComfyUI refused the workflow ({r.status_code}): {r.text[:300]}")
        pid = r.json().get("prompt_id")
        if not pid:
            raise ToolError("ComfyUI returned no prompt id")
        t0 = time.time()
        while time.time() - t0 < 600:
            h = c.get(f"{base}/history/{pid}").json().get(pid)
            if h:
                st = h.get("status") or {}
                if st.get("status_str") == "error":
                    raise ToolError("ComfyUI reported an error: " + json.dumps(st.get("messages") or [])[:400])
                imgs = [im for node in (h.get("outputs") or {}).values() for im in node.get("images") or []]
                if imgs:
                    im = imgs[0]
                    g = c.get(f"{base}/view", params={"filename": im["filename"], "subfolder": im.get("subfolder", ""),
                                                      "type": im.get("type", "output")})
                    if g.status_code != 200:
                        raise ToolError("ComfyUI produced an image the deck could not fetch")
                    ext = os.path.splitext(im["filename"])[1] or ".png"
                    path, url = _generated(user, ext)
                    path.write_bytes(g.content)
                    return {"url": url, "markdown": f"![{prompt[:60]}]({url})", "seed": fill["{seed}"],
                            "note": "include the markdown in your reply to show the image"}
                if st.get("completed"):
                    raise ToolError("the workflow finished without an image output")
            time.sleep(1.0)
    raise ToolError("ComfyUI did not finish within 10 minutes")


def speak(args: dict, user=None) -> dict:
    text = (args.get("text") or "").strip()
    if not text:
        raise ToolError("text is required")
    if len(text) > 4000:
        raise ToolError("text is limited to 4000 characters")
    voice = (args.get("voice") or getattr(config, "TTS_DEFAULT_VOICE", "") or "").strip()
    body = {"model": "tts-1", "input": text, "response_format": "mp3"}
    if voice:
        body["voice"] = voice
    base = getattr(config, "TTS_ENDPOINT", "").rstrip("/")
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=5.0)) as c:
        r = c.post(f"{base}/v1/audio/speech", json=body)
    if r.status_code != 200:
        raise ToolError(f"the TTS endpoint answered {r.status_code}: {r.text[:200]}")
    ctype = (r.headers.get("content-type") or "").lower()
    ext = ".wav" if "wav" in ctype else ".ogg" if "ogg" in ctype else ".mp3"
    path, url = _generated(user, ext)
    path.write_bytes(r.content)
    return {"url": url, "markdown": f"[▶ listen]({url})", "seconds_estimate": round(len(text.split()) / 2.5, 1),
            "note": "include the markdown link in your reply so the user can play it"}


# ── scheduled tasks ─────────────────────────────────────────────────────────

def schedule_task(args: dict, user=None) -> dict:
    sched = _ctx("scheduler")
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        raise ToolError("prompt is required")
    every = args.get("every_minutes")
    at = (args.get("daily_at") or "").strip()
    if every not in (None, "") and at:
        raise ToolError("give every_minutes or daily_at, not both")
    if every not in (None, ""):
        every = int(_num(every, "every_minutes", 5, 10080, 60))
        at = None
    elif at:
        if not re.match(r"^([01]?\d|2[0-3]):[0-5]\d$", at):
            raise ToolError("daily_at is HH:MM, 24-hour")
        every = None
    else:
        raise ToolError("give every_minutes or daily_at")
    model = (args.get("model") or "").strip() or (CONTEXT.get("current_model") or (lambda: None))()
    task = sched.add(_account(user), prompt=prompt, every=every, at=at, model=model)
    return {"scheduled": task, "note": "results appear as new chats in the sidebar"}


def list_tasks(args: dict, user=None) -> dict:
    return {"tasks": _ctx("scheduler").list_tasks(_account(user))}


def cancel_task(args: dict, user=None) -> dict:
    tid = (args.get("id") or "").strip()
    if not _ctx("scheduler").remove(_account(user), tid):
        raise ToolError(f"no task {tid}")
    return {"cancelled": tid}


# ── home assistant ──────────────────────────────────────────────────────────

def _ha():
    url = (getattr(config, "HA_URL", "") or "").strip().rstrip("/")
    tok = (getattr(config, "HA_TOKEN", "") or "").strip()
    if not url or not tok:
        raise ToolError("Home Assistant is not configured (Settings → Tool Integrations)")
    return url, {"Authorization": "Bearer " + tok, "Content-Type": "application/json"}


def home_assistant(args: dict) -> dict:
    url, headers = _ha()
    action = (args.get("action") or "").lower()
    with httpx.Client(timeout=TIMEOUT) as c:
        if action == "list":
            r = c.get(f"{url}/api/states", headers=headers)
            if r.status_code != 200:
                raise ToolError(f"Home Assistant answered {r.status_code}")
            flt = (args.get("filter") or "").strip().lower()
            out = []
            for s in r.json():
                eid = s.get("entity_id", "")
                name = (s.get("attributes") or {}).get("friendly_name", "")
                if flt and not (eid.startswith(flt + ".") or flt in eid.lower() or flt in name.lower()):
                    continue
                out.append({"entity_id": eid, "name": name, "state": s.get("state"),
                            "unit": (s.get("attributes") or {}).get("unit_of_measurement")})
            return {"count": len(out), "entities": out[:200], "truncated": len(out) > 200}
        if action == "state":
            eid = (args.get("entity_id") or "").strip()
            if not eid:
                raise ToolError("entity_id is required")
            r = c.get(f"{url}/api/states/{eid}", headers=headers)
            if r.status_code == 404:
                raise ToolError(f"no entity {eid}")
            if r.status_code != 200:
                raise ToolError(f"Home Assistant answered {r.status_code}")
            s = r.json()
            return {"entity_id": eid, "state": s.get("state"), "attributes": s.get("attributes"),
                    "last_changed": s.get("last_changed")}
        if action == "call":
            dom, svc = (args.get("domain") or "").strip(), (args.get("service") or "").strip()
            data = args.get("data") if isinstance(args.get("data"), dict) else {}
            if args.get("entity_id") and "entity_id" not in data:
                data["entity_id"] = args["entity_id"]
            if not dom or not svc:
                raise ToolError("domain and service are required for call")
            r = c.post(f"{url}/api/services/{dom}/{svc}", headers=headers, json=data)
            if r.status_code not in (200, 201):
                raise ToolError(f"Home Assistant answered {r.status_code}: {r.text[:200]}")
            changed = [{"entity_id": s.get("entity_id"), "state": s.get("state")} for s in r.json()] \
                if isinstance(r.json(), list) else r.json()
            return {"called": f"{dom}.{svc}", "data": data, "changed": changed}
    raise ToolError("action must be list, state or call")


# ── email ───────────────────────────────────────────────────────────────────

def send_email(args: dict) -> dict:
    host = (getattr(config, "SMTP_HOST", "") or "").strip()
    sender = (getattr(config, "SMTP_FROM", "") or "").strip()
    if not host or not sender:
        raise ToolError("email is not configured (Settings → Tool Integrations)")
    to = [t.strip() for t in re.split(r"[,;\s]+", args.get("to") or "") if t.strip()]
    if not to or not all(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", t) for t in to):
        raise ToolError("to must be one or more email addresses")
    subject = (args.get("subject") or "").strip()[:200]
    body = args.get("body") or ""
    if not subject or not body.strip():
        raise ToolError("subject and body are required")
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, ", ".join(to), subject
    msg.set_content(body)
    port = int(getattr(config, "SMTP_PORT", 587) or 587)
    user_, pw = (getattr(config, "SMTP_USER", "") or "").strip(), (getattr(config, "SMTP_PASSWORD", "") or "")
    try:
        if port == 465:
            srv = smtplib.SMTP_SSL(host, port, timeout=30)
        else:
            srv = smtplib.SMTP(host, port, timeout=30)
            srv.ehlo()
            if srv.has_extn("starttls"):
                srv.starttls()
                srv.ehlo()
        with srv:
            if user_:
                srv.login(user_, pw)
            srv.send_message(msg)
    except (smtplib.SMTPException, OSError) as e:
        raise ToolError(f"sending failed: {e}")
    return {"sent": True, "to": to, "subject": subject}


# ── dispatch ────────────────────────────────────────────────────────────────

RUN = {"get_datetime": get_datetime, "get_weather": get_weather, "web_search": web_search,
       "news": news, "fetch_url": fetch_url, "wikipedia": wikipedia, "calculate": calculate,
       "unit_convert": unit_convert, "currency": currency, "run_python": run_python,
       "read_github": read_github, "search_chats": search_chats, "list_documents": list_documents,
       "search_documents": search_documents, "read_document": read_document,
       "deck_status": deck_status, "list_models": list_models, "launch_model": launch_model,
       "stop_model": stop_model, "benchmark": benchmark, "generate_image": generate_image,
       "speak": speak, "schedule_task": schedule_task, "list_tasks": list_tasks,
       "cancel_task": cancel_task, "home_assistant": home_assistant, "send_email": send_email}
# tools that take the calling account
PER_USER = {"search_chats", "list_documents", "search_documents", "read_document",
            "generate_image", "speak", "schedule_task", "list_tasks", "cancel_task"}


def _offered(name: str, user) -> bool:
    """Whether a tool makes sense to offer right now: configured, reachable,
    and allowed for this account. A tool that always fails wastes a turn."""
    try:
        if name in ("web_search", "news"):
            _provider()
        elif name == "run_python":
            return shutil.which("docker") is not None
        elif name in ("deck_status", "list_models", "launch_model", "stop_model", "benchmark"):
            if not CONTEXT:
                return False
        elif name == "generate_image":
            if _sd_base():
                return True
            _comfy_workflow()
            return _reachable(getattr(config, "COMFY_URL", "").rstrip("/") + "/system_stats")
        elif name == "speak":
            return _reachable(getattr(config, "TTS_ENDPOINT", "").rstrip("/") + "/v1/models")
        elif name in ("schedule_task", "list_tasks", "cancel_task"):
            return "scheduler" in CONTEXT
        elif name == "home_assistant":
            _ha()
        elif name == "send_email":
            return bool((getattr(config, "SMTP_HOST", "") or "").strip()
                        and (getattr(config, "SMTP_FROM", "") or "").strip())
    except ToolError:
        return False
    need = ACCESS.get(name)
    if need and user is not None and need not in (user.get("access") or []):
        return False
    return True


# what each tool needs before it is offered, for the chat's help panel
NEEDS = {
    "web_search": "a search provider: a Brave API key or a SearXNG URL under Settings → Chat Tools",
    "news": "the same search provider as web_search",
    "run_python": "Docker on the deck (the sandbox image builds itself on first use)",
    "deck_status": "the Dashboard view", "list_models": "the Dashboard view",
    "launch_model": "the Dashboard view", "stop_model": "the Dashboard view",
    "benchmark": "the Benchmarks view",
    "generate_image": "sd-server running (Settings → Image Generation), or ComfyUI with a workflow file",
    "speak": "the TTS service running",
    "schedule_task": "nothing", "list_tasks": "nothing", "cancel_task": "nothing",
    "home_assistant": "a Home Assistant URL and token under Settings → Tool Integrations",
    "send_email": "SMTP settings under Settings → Tool Integrations",
}


def catalog(user=None) -> list:
    """Every built-in tool, offered or not, with why not: the help panel."""
    out = []
    for d in DEFINITIONS:
        name = d["function"]["name"]
        need = ACCESS.get(name)
        out.append({"name": name, "description": d["function"]["description"],
                    "offered": _offered(name, user),
                    "needs": NEEDS.get(name, "nothing"),
                    "no_access": bool(need and user is not None and need not in (user.get("access") or []))})
    return out


def definitions(user=None) -> list:
    """What the chat offers the model for this account."""
    return [{k: v for k, v in d.items() if k != "_access"}
            for d in DEFINITIONS if _offered(d["function"]["name"], user)]


def run(name: str, args, user=None) -> str:
    """Run one tool; the text the model gets back (JSON or an error line)."""
    fn = RUN.get(name)
    if fn is None:
        raise ToolError(f"no built-in tool called {name}")
    need = ACCESS.get(name)
    if need and user is not None and need not in (user.get("access") or []):
        raise ToolError(f"this account has no access to the {need} view, which {name} needs")
    if not isinstance(args, dict):
        args = {}
    try:
        out = fn(args, user=user) if name in PER_USER else fn(args)
        return json.dumps(out, ensure_ascii=False, default=str)
    except ToolError:
        raise
    except httpx.HTTPError as e:
        raise ToolError(f"{name} failed: {e.__class__.__name__}: {e}")
    except subprocess.TimeoutExpired:
        raise ToolError(f"{name} timed out")

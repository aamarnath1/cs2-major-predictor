"""Polymarket match prices for the Singapore Major (public API, no key).

Each Swiss match has a moneyline market. For a match that has started, the price is the last one
traded before the scheduled start, so it can be compared fairly with the model's pre-match odds.
Those prices never change, so they are cached; only upcoming matches are refetched.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import requests

EVENTS = "https://gamma-api.polymarket.com/events"
HISTORY = "https://clob.polymarket.com/prices-history"
CS2_TAG = 100780
HEADERS = {"User-Agent": "cs2-major-predictor/1.0 (github.com/aamarnath1/cs2-major-predictor)"}
# Polymarket spellings that differ from the site's names (compared case-insensitively).
ALIASES = {"nip": "Ninjas in Pyjamas", "pvision": "PARIVISION", "dendele cs": "DENDELE",
           "navi": "Natus Vincere", "mongolz": "The MongolZ", "1win": "The One", "betboom team": "BetBoom"}


def team_name(raw: str, names: list[str]) -> str | None:
    key = raw.strip().casefold()
    by_key = {n.casefold(): n for n in names}
    return by_key.get(key) or (ALIASES.get(key) if ALIASES.get(key) in names else None)


def _get(url: str, **params) -> object:
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _events(since: str) -> list[dict]:
    out, offset = [], 0
    while True:
        page = _get(EVENTS, tag_id=CS2_TAG, end_date_min=since, limit=50, offset=offset)
        out += page
        if len(page) < 50:
            return out
        offset += 50


def _parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T")).astimezone(UTC)


def price_before(token: str, start: datetime) -> float | None:
    """Last traded price of `token` at or before `start`."""
    hist = _get(HISTORY, market=token, interval="max", fidelity=60).get("history", [])
    before = [h["p"] for h in hist if h["t"] <= start.timestamp()]
    return float(before[-1]) if before else None


def fetch_market_odds(names: list[str], cache: Path, since: str = "2026-11-23T00:00:00Z") -> list[dict]:
    """[{stage, a, b, p_a, start, final, slug}] for Swiss matches; falls back to the cache when offline."""
    old = {e["slug"]: e for e in json.loads(cache.read_text(encoding="utf-8"))} if cache.exists() else {}
    try:
        events = _events(since)
    except requests.RequestException as exc:
        print(f"market: Polymarket unavailable ({exc.__class__.__name__}), using cached prices")
        return list(old.values())
    now = datetime.now(UTC)
    out = dict(old)
    for ev in events:
        meta = ev.get("eventMetadata") or {}
        if meta.get("league") != "PGL Major" or meta.get("serie") != "Singapore":
            continue
        tour = meta.get("tournament", "")
        if not tour.startswith("Stage "):
            continue
        slug = ev["slug"]
        if old.get(slug, {}).get("final"):
            continue
        ml = [m for m in ev.get("markets", []) if m.get("sportsMarketType") == "moneyline"]
        if len(ml) != 1:
            continue
        m = ml[0]
        raw = json.loads(m["outcomes"])
        a, b = (team_name(x, names) for x in raw)
        if not a or not b:
            print(f"market: skipped {ev['title']!r} (team names not recognised)")
            continue
        start = _parse_time(ev.get("startTime") or m["gameStartTime"])
        try:
            if start <= now:
                p = price_before(json.loads(m["clobTokenIds"])[0], start)
            else:
                p = float(json.loads(m["outcomePrices"])[0])
        except (requests.RequestException, ValueError, KeyError):
            continue
        if p is None or not 0 < p < 1:
            continue
        out[slug] = {"stage": int(tour.split()[1]), "a": a, "b": b, "p_a": round(p, 4),
                     "start": start.isoformat(timespec="minutes"), "final": start <= now, "slug": slug}
    rows = sorted(out.values(), key=lambda e: e["start"])
    cache.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows

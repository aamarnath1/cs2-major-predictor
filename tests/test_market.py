import json
from datetime import UTC, datetime, timedelta

import requests

from cs2p import market

NAMES = ["Ninjas in Pyjamas", "Liquid", "PARIVISION", "FaZe"]


def event(slug, a, b, start, prices=("0.6", "0.4"), tournament="Stage 1"):
    return {"slug": slug, "title": f"Counter-Strike: {a} vs {b}", "startTime": start.isoformat(),
            "eventMetadata": {"league": "PGL Major", "serie": "Singapore", "tournament": tournament},
            "markets": [{"sportsMarketType": "child_moneyline"},
                        {"sportsMarketType": "moneyline", "outcomes": json.dumps([a, b]),
                         "outcomePrices": json.dumps(list(prices)), "clobTokenIds": json.dumps(["tok", "tok2"]),
                         "gameStartTime": start.isoformat()}]}


def test_aliases_and_unknown_names():
    assert market.team_name("NIP", NAMES) == "Ninjas in Pyjamas"
    assert market.team_name("PVISION", NAMES) == "PARIVISION"
    assert market.team_name("liquid", NAMES) == "Liquid"
    assert market.team_name("Someone Else", NAMES) is None


def test_started_match_uses_last_price_before_start_and_is_cached(tmp_path, monkeypatch):
    now = datetime.now(UTC)
    past, future = now - timedelta(hours=3), now + timedelta(hours=5)
    evs = [event("played", "NIP", "Liquid", past), event("next", "PVISION", "FaZe", future),
           event("other", "NIP", "FaZe", future, tournament="Playoffs"),
           {**event("x", "NIP", "Liquid", past), "eventMetadata": {"league": "IEM", "serie": "Cologne"}}]
    hist = [{"t": (past - timedelta(hours=1)).timestamp(), "p": 0.45},
            {"t": (past - timedelta(minutes=5)).timestamp(), "p": 0.48},
            {"t": (past + timedelta(minutes=30)).timestamp(), "p": 0.9}]  # live trading after start: ignored
    calls = []

    def fake_get(url, **params):
        calls.append(url)
        return evs if url == market.EVENTS else {"history": hist}

    monkeypatch.setattr(market, "_get", fake_get)
    cache = tmp_path / "m.json"
    rows = {r["slug"]: r for r in market.fetch_market_odds(NAMES, cache)}
    assert set(rows) == {"played", "next"}
    assert rows["played"] == {**rows["played"], "a": "Ninjas in Pyjamas", "b": "Liquid", "p_a": 0.48, "final": True, "stage": 1}
    assert rows["next"]["p_a"] == 0.6 and not rows["next"]["final"]

    calls.clear()
    market.fetch_market_odds(NAMES, cache)
    assert market.HISTORY not in calls  # the final price is not fetched again


def test_offline_falls_back_to_cache(tmp_path, monkeypatch):
    cache = tmp_path / "m.json"
    cache.write_text(json.dumps([{"slug": "s", "stage": 1, "a": "Liquid", "b": "FaZe", "p_a": 0.5,
                                  "start": "2026-11-24T08:00+00:00", "final": True}]))

    def down(url, **params):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(market, "_get", down)
    assert [r["slug"] for r in market.fetch_market_odds(NAMES, cache)] == ["s"]

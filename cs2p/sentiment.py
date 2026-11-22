"""Crowd sentiment about the 32 Singapore Major teams, from public social posts.

Method
------
1. Fetch (cached under ``data/raw/<source>/``) posts from the last ~30 days before Stage 1:
   * Bluesky public AppView search (``app.bsky.feed.searchPosts``; robots.txt allows crawling
     the public API) for general CS2 / Major / Pick'em queries plus two queries per team, and the
     reply trees (``getPostThread``) of Pick'em / prediction posts and of posts with replies.
     Unauthenticated search returns only the first page (<= 100 posts): cursor pages give 403.
   * Reddit (r/GlobalOffensive, r/cs2, r/csgobetting) ONLY via the official OAuth API when
     ``REDDIT_CLIENT_ID`` / ``REDDIT_CLIENT_SECRET`` are set (app-only grant, no user login).
     Unauthenticated JSON endpoints return 403 and robots.txt is ``Disallow: /``, so they are
     not used. HLTV is never fetched (ToS). YouTube skipped (needs an API key).
2. Keep only documents with CS context: a CS keyword (cs2, counter-strike, pick'em, major ...)
   in the text or in the thread root, or any Reddit doc from the CS subreddits. This removes
   e.g. NFL "Falcons", FIFA "FUT" and League-of-Legends-only "G2"/"FlyQuest" posts. Docs naming
   another esport (R6, Valorant, LoL, Dota ...) without "cs2"/"counter-strike" are dropped.
3. Mentions: per-team alias regexes with word boundaries. Ambiguous English words (Legacy,
   Imperial, Wildcard, Aurora, The One, ...) are matched case-sensitively; short ambiguous
   aliases ("one", "bb", "fq", "ef", "pv") are not used at all.
4. At most ``AUTHOR_CAP`` documents per (author, team) are counted, so news/tipster bots
   (which post most CS content on Bluesky) cannot dominate a team's numbers.
5. Sentiment: VADER compound on the whole document; positive >= 0.05, negative <= -0.05.
6. Crowd picks (windowed heuristic): for documents in a prediction context (doc or thread root
   matches pick'em / prediction / predict / "my picks" / "Singapore Major"), a team mention
   counts toward ``crowd_3_0`` / ``crowd_0_3`` / ``crowd_champion`` if a "3-0", "0-3" or
   champion phrase ("win(s) the major", "champion(s)", "winner", "lift the trophy", "take it
   all") occurs within ``WINDOW`` characters of that mention. Each document counts at most once
   per team and bucket.

Known biases
------------
* Popular teams (Vitality, NaVi, Spirit, FaZe, s1mple's GOSH) are over-mentioned regardless of
  expected performance; share of voice measures attention, not belief.
* Sarcasm, banter and memes ("Vitality 0-3 trust") are scored literally.
* VADER is not gaming-aware: "insane", "sick", "kill", "destroy" read as negative, and
  "3-0" / "0-3" carry no sentiment for it.
* Much of Bluesky CS content is automated: news mirrors, betting-tip and Russian "prognoz"
  bots. The author cap limits this but the remaining crowd is still not "fans".
* VADER is English-only; Portuguese / Russian posts (a large share) score ~0, pulling
  teams with Brazilian/CIS coverage (FURIA, Imperial, Spirit, BetBoom) towards neutral.
* Bluesky's CS community is small and skews English / Western; Reddit (when enabled) skews
  the same way. Asian and CIS fan bases are under-represented.
* The crowd window is crude: "NaVi 3-0 over FaZe" credits both teams; map scores ("won 3-0
  on maps") are confused with Swiss records.
"""

import argparse
import base64
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from cs2p import teams

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_JSON = ROOT / "outputs" / "sentiment.json"
OUT_CSV = ROOT / "data" / "processed" / "sentiment_comments.csv"

UA = "cs2-major-predictor/0.1 (personal research)"
SLEEP = 2.0
SINCE = "2026-10-24T00:00:00Z"  # ~30 days before Stage 1 (2026-11-24); fixed so the cache hits
WINDOW = 60
MAX_THREADS = 200
AUTHOR_CAP = 3

ALL_TEAMS = [d for d, _ in teams.STAGE1 + teams.STAGE2_INVITES + teams.STAGE3_INVITES]

# Aliases: plain strings are case-insensitive, ("text", True) entries are case-sensitive.
ALIASES: dict[str, list] = {
    "Ninjas in Pyjamas": ["ninjas in pyjamas", "ninjas in pajamas", ("NiP", True)],
    "PARIVISION": ["parivision", "pvision"],
    "HEROIC": [("HEROIC", True), ("Heroic", True)],
    "FaZe": ["faze"],
    "Liquid": ["team liquid", ("Liquid", True), ("TL", True)],
    "Eternal Fire": ["eternal fire"],
    "100 Thieves": ["100 thieves", "100t"],
    "9z": ["9z"],
    "M80": ["m80"],
    "DENDELE": ["dendele"],
    "Imperial": [("Imperial", True), "imperial esports"],
    "Wildcard": [("Wildcard", True), "wildcard gaming"],
    "GOSH": [("GOSH", True), "bc.game", "bcgame", r"s1mple'?s team"],
    "FlyQuest": ["flyquest"],
    "Chinggis Warriors": ["chinggis warriors", "chinggis"],
    "Lynn Vision": ["lynn vision", "lvg"],
    "FUT": [("FUT", True), "fut esports"],
    "Nemiga": ["nemiga"],
    "G2": ["g2"],
    "The One": ["1win", ("The One", True)],
    "Natus Vincere": ["natus vincere", "navi", "na'vi", "na`vi"],
    "Astralis": ["astralis"],
    "The MongolZ": ["mongolz"],
    "MIBR": ["mibr"],
    "MOUZ": ["mouz", "mousesports"],
    "Vitality": ["vitality", "vita", "zywoo'?s team", "zywoo team"],
    "FURIA": ["furia"],
    "Aurora": [("Aurora", True), ("AURORA", True)],
    "Spirit": ["team spirit", ("Spirit", True), "donk"],
    "BetBoom": ["betboom"],
    "Falcons": ["falcons", "niko"],
    "Legacy": [("Legacy", True), ("LEGACY", True)],
}
assert set(ALIASES) == set(ALL_TEAMS)


def _alias_re(aliases: list) -> tuple[re.Pattern, re.Pattern | None]:
    ci = [a for a in aliases if isinstance(a, str)]
    cs = [a[0] for a in aliases if isinstance(a, tuple)]
    wrap = r"(?<![\w.])(?:{})(?![\w])"
    ci_re = re.compile(wrap.format("|".join(ci)), re.I) if ci else None
    cs_re = re.compile(wrap.format("|".join(cs))) if cs else None
    return ci_re, cs_re


TEAM_RES = {t: _alias_re(a) for t, a in ALIASES.items()}

CS_CONTEXT = re.compile(
    r"\b(cs2|cs ?go|cs:go|counter[- ]?strike|pick ?'?em|pickem|hltv|singapore major|pgl|"
    r"cs major|major stage|stage [123]|swiss stage|iem|blast premier|esl pro league|faceit|"
    r"zywoo|s1mple|m0nesy|ropz|donk|frozen|jks|broky|apex|xantares)\b",
    re.I,
)
OTHER_GAME = re.compile(r"\b(r6|rainbow ?six|siege|valorant|vct|league of legends|lck|lec|"
                        r"dota|rocket league|overwatch|apex legends)\b", re.I)
CS_EXPLICIT = re.compile(r"\bcs2\b|counter[- ]?strike", re.I)
PRED = re.compile(
    r"pick ?'?em|pickem|predict|prediction|my picks|singapore major|\b3-0\b.*\b0-3\b", re.I | re.S
)
CROWD = {
    "crowd_3_0": re.compile(r"(?<![\d.])3\s?[-–:]\s?0(?![\d.])"),
    "crowd_0_3": re.compile(r"(?<![\d.])0\s?[-–:]\s?3(?![\d.])"),
    "crowd_champion": re.compile(
        r"win(s|ning)? (the|this|it all|singapore)|champions?\b|winners?\b|lift(s|ing)? the "
        r"(trophy|cup)|take(s)? it all|title",
        re.I,
    ),
}

log = logging.getLogger("cs2p.sentiment")


# ---------------------------------------------------------------- fetching (cached)
class Fetcher:
    """GET with an on-disk JSON cache, polite sleeps and 429 backoff."""

    def __init__(self) -> None:
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.hits = self.misses = 0
        self.failures: list[str] = []

    def get(self, source: str, url: str, params: dict | None = None,
            headers: dict | None = None) -> dict | None:
        key = hashlib.sha1(f"{url}?{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
        path = RAW / source / f"{key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        self.misses += 1
        for attempt in range(5):
            time.sleep(SLEEP)
            try:
                r = self.s.get(url, params=params, headers=headers, timeout=30)
            except requests.RequestException as e:
                log.warning("request error %s: %s", url, e)
                continue
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 0) or 0) or 10 * 2**attempt
                log.warning("429 on %s, backing off %ss", url, wait)
                time.sleep(wait)
                continue
            if r.status_code != 200:
                self.failures.append(f"{r.status_code} {url} {params or ''}")
                return None
            data = r.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
            return data
        self.failures.append(f"gave up {url} {params or ''}")
        return None


def _doc(src, id_, date, community, title, text, likes, pred_root=False, author=""):
    return {"source": src, "id": id_, "date": date[:10], "community": community, "author": author,
            "thread_title": title, "text": text or "", "likes": likes or 0,
            "root_text": title, "pred_root": pred_root}


def fetch_bluesky(f: Fetcher) -> list[dict]:
    """Search posts + reply trees of selected threads from the public Bluesky AppView."""
    base = "https://api.bsky.app/xrpc/"
    general = ["cs2", "counter-strike", "csgo", "cs2 major", "singapore major", "pgl major",
               "pickem", "pick'em", "cs2 predictions", "hltv"]
    queries = general + [f"{t} cs2" for t in ALL_TEAMS] + [f"{t} major" for t in ALL_TEAMS]
    posts: dict[str, dict] = {}
    for q in queries:  # first page only: unauthenticated cursor requests return 403
        params = {"q": q, "limit": 100, "sort": "latest", "since": SINCE}
        d = f.get("bluesky", base + "app.bsky.feed.searchPosts", params)
        for p in (d or {}).get("posts", []):
            posts[p["uri"]] = p

    docs = []
    for p in posts.values():
        text = p["record"].get("text", "")
        docs.append(_doc("bluesky", p["uri"], p["record"].get("createdAt", p["indexedAt"]),
                         "bluesky", text[:100], text, p.get("likeCount"), bool(PRED.search(text)),
                         p["author"]["handle"]))

    # Reply trees: prediction posts first, then most-replied CS posts.
    cands = [p for p in posts.values() if p.get("replyCount", 0) > 0
             and CS_CONTEXT.search(p["record"].get("text", ""))]
    cands.sort(key=lambda p: (not PRED.search(p["record"].get("text", "")), -p["replyCount"]))
    for p in cands[:MAX_THREADS]:
        d = f.get("bluesky", base + "app.bsky.feed.getPostThread", {"uri": p["uri"], "depth": 10})
        if not d:
            continue
        root_text = p["record"].get("text", "")
        stack = list(d.get("thread", {}).get("replies", []) or [])
        while stack:
            n = stack.pop()
            post = n.get("post")
            if not post:
                continue
            stack.extend(n.get("replies", []) or [])
            if post["uri"] in posts:
                continue
            t = post["record"].get("text", "")
            docs.append(_doc("bluesky", post["uri"], post["record"].get("createdAt", ""),
                             "bluesky", root_text[:100], t, post.get("likeCount"),
                             bool(PRED.search(root_text)), post["author"]["handle"]))
            docs[-1]["root_text"] = root_text
    return docs


def fetch_reddit(f: Fetcher) -> tuple[list[dict], str]:
    """Reddit via the official OAuth API (app-only). Skipped without credentials.

    Credentials come from the environment or a gitignored `.env` file in the project root.
    """
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip('"'))
    cid, secret = os.environ.get("REDDIT_CLIENT_ID"), os.environ.get("REDDIT_CLIENT_SECRET")
    if not (cid and secret):
        return [], ("skipped: unauthenticated JSON returns 403 and robots.txt disallows "
                    "crawling; set REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET to enable OAuth")
    auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    r = requests.post("https://www.reddit.com/api/v1/access_token",
                      data={"grant_type": "client_credentials"},
                      headers={"Authorization": f"Basic {auth}", "User-Agent": UA}, timeout=30)
    if r.status_code != 200:
        return [], f"oauth failed: {r.status_code}"
    hdr = {"Authorization": f"bearer {r.json()['access_token']}"}
    api = "https://oauth.reddit.com"
    threads: dict[str, dict] = {}
    for sub in ["GlobalOffensive", "cs2", "csgobetting"]:
        for q in ["pick'em", "pickem", "predictions", "Singapore Major", "match thread"]:
            d = f.get("reddit", f"{api}/r/{sub}/search",
                      {"q": q, "restrict_sr": 1, "sort": "new", "t": "month", "limit": 100}, hdr)
            for c in (d or {}).get("data", {}).get("children", []):
                threads[c["data"]["id"]] = c["data"]
    docs = []
    for tid, t in list(threads.items())[:MAX_THREADS]:
        title = t.get("title", "")
        body = f"{title}\n{t.get('selftext', '')}"
        pred = bool(PRED.search(title))
        date = time.strftime("%Y-%m-%d", time.gmtime(t.get("created_utc", 0)))
        docs.append(_doc("reddit", tid, date, t["subreddit"], title, body, t.get("score"), pred,
                         t.get("author", "")))
        d = f.get("reddit", f"{api}/r/{t['subreddit']}/comments/{tid}", {"limit": 500}, hdr)
        stack = list(d[1]["data"]["children"]) if d and len(d) > 1 else []
        while stack:
            c = stack.pop()
            if c.get("kind") != "t1":
                continue
            cd = c["data"]
            reps = cd.get("replies")
            if isinstance(reps, dict):
                stack.extend(reps["data"]["children"])
            date = time.strftime("%Y-%m-%d", time.gmtime(cd.get("created_utc", 0)))
            docs.append(_doc("reddit", cd["id"], date, t["subreddit"], title, cd.get("body", ""),
                             cd.get("score"), pred, cd.get("author", "")))
    return docs, f"ok: {len(threads)} threads"


# ---------------------------------------------------------------- analysis
def find_mentions(text: str) -> dict[str, list[tuple[int, int]]]:
    """Team -> list of (start, end) alias spans in ``text``."""
    out = {}
    for team, (ci, cs) in TEAM_RES.items():
        spans = [m.span() for r in (ci, cs) if r for m in r.finditer(text)]
        if spans:
            out[team] = sorted(spans)
    return out


def snippet(text: str, span: tuple[int, int], n: int = 190) -> str:
    a = max(0, span[0] - n // 2)
    s = " ".join(text[a:a + n].split())
    return ("..." if a else "") + s


def analyse(docs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (cs-relevant docs, mention rows)."""
    vader = SentimentIntensityAnalyzer()
    seen, keep, rows = set(), [], []
    per_author: dict[tuple[str, str], int] = defaultdict(int)
    for d in docs:
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        if d["source"] != "reddit" and not (CS_CONTEXT.search(d["text"])
                                            or CS_CONTEXT.search(d["root_text"])):
            continue
        if OTHER_GAME.search(d["text"]) and not CS_EXPLICIT.search(d["text"]):
            continue
        keep.append(d)
        ments = find_mentions(d["text"])
        if not ments:
            continue
        comp = vader.polarity_scores(d["text"])["compound"]
        pred = d["pred_root"] or bool(PRED.search(d["text"]))
        for team, spans in ments.items():
            per_author[d["author"], team] += 1
            if d["author"] and per_author[d["author"], team] > AUTHOR_CAP:
                continue
            row = {"date": d["date"], "subreddit": d["community"],
                   "thread_title": " ".join(d["thread_title"].split())[:100], "team": team,
                   "compound": comp, "snippet": snippet(d["text"], spans[0]),
                   "alias": d["text"][spans[0][0]:spans[0][1]], "likes": d["likes"],
                   "prediction": pred, "source": d["source"]}
            for k, rx in CROWD.items():
                row[k] = pred and any(
                    rx.search(d["text"][max(0, a - WINDOW):b + WINDOW]) for a, b in spans)
            rows.append(row)
    return keep, rows


def summarise(keep: list[dict], rows: list[dict], meta: dict) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["team"]].append(r)
    total = len(rows) or 1
    out = {}
    for t in ALL_TEAMS:
        rs = by.get(t, [])
        n = len(rs)
        comps = [r["compound"] for r in rs]
        ex, texts = [], set()
        for r in sorted(rs, key=lambda r: -r["likes"]):
            if r["snippet"] not in texts and len(ex) < 5:
                texts.add(r["snippet"])
                ex.append({"date": r["date"], "source": r["source"], "compound": r["compound"],
                           "text": r["snippet"][:199]})
        out[t] = {
            "mentions": n,
            "share_of_voice": round(n / total, 4),
            "mean_compound": round(sum(comps) / n, 4) if n else None,
            "pct_positive": round(sum(c >= 0.05 for c in comps) / n, 4) if n else None,
            "pct_negative": round(sum(c <= -0.05 for c in comps) / n, 4) if n else None,
            **{k: sum(r[k] for r in rs) for k in CROWD},
            "examples": ex,
        }
    dates = sorted(d["date"] for d in keep if d["date"])
    meta |= {"total_documents": len(keep), "total_mentions": len(rows),
             "prediction_documents": sum(1 for d in keep if d["pred_root"]),
             "date_range": [dates[0], dates[-1]] if dates else None,
             "window_chars": WINDOW, "since": SINCE, "author_cap": AUTHOR_CAP}
    return {"metadata": meta, "teams": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=int, default=20, help="random mentions to print (precision)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    f = Fetcher()
    docs = fetch_bluesky(f)
    rdocs, rstatus = fetch_reddit(f)
    keep, rows = analyse(docs + rdocs)
    meta = {"sources": {"bluesky": f"ok: {sum(d['source'] == 'bluesky' for d in docs)} docs",
                        "reddit": rstatus, "hltv": "not used (ToS)",
                        "youtube": "not used (API key required)"},
            "fetched_documents": len(docs) + len(rdocs),
            "cache_hits": f.hits, "cache_misses": f.misses, "failed_requests": f.failures[:50]}
    res = summarise(keep, rows, meta)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["date", "subreddit", "thread_title", "team", "compound", "snippet"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    m = res["metadata"]
    print(f"\ndocs kept {m['total_documents']} / fetched {m['fetched_documents']}, "
          f"mentions {m['total_mentions']}, dates {m['date_range']}, "
          f"cache hits {f.hits} misses {f.misses}, failures {len(f.failures)}")
    print(f"{'team':<20}{'ment':>6}{'share':>8}{'sent':>8}{'3-0':>6}{'0-3':>6}{'champ':>7}")
    for t, s in sorted(res["teams"].items(), key=lambda kv: -kv[1]["mentions"]):
        mc = f"{s['mean_compound']:+.3f}" if s["mean_compound"] is not None else "-"
        print(f"{t:<20}{s['mentions']:>6}{s['share_of_voice']:>8.1%}{mc:>8}"
              f"{s['crowd_3_0']:>6}{s['crowd_0_3']:>6}{s['crowd_champion']:>7}")

    if args.sample and rows:
        print(f"\nPrecision sample ({args.sample} random mentions, seed 0):")
        for i, r in enumerate(random.Random(0).sample(rows, min(args.sample, len(rows))), 1):
            print(f"{i:>2}. [{r['team']} via '{r['alias']}'] {r['snippet'][:180]}")


if __name__ == "__main__":
    main()

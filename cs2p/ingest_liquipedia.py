"""Ingest recent CS2 match results and past Major Swiss stages from Liquipedia.

Uses only ``action=query&prop=revisions`` (1 request / 2 s, up to 50 titles per request).
Every page is cached as JSON under ``data/raw/liquipedia/pages/`` so reruns are offline.
"""

import hashlib
import html
import json
import re
import time
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "liquipedia"
PAGES = RAW / "pages"
OUT = ROOT / "data" / "processed"

API = "https://liquipedia.net/counterstrike/api.php"
HEADERS = {
    "User-Agent": "cs2-major-predictor/0.1 (personal research)",
    "Accept-Encoding": "gzip",
}
MIN_INTERVAL = 2.1
_last_request = 0.0
N_REQUESTS = 0


def _cache_path(title: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", title)[:80]
    digest = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
    return PAGES / f"{safe}__{digest}.json"


def _api_get(params: dict) -> dict:
    global _last_request, N_REQUESTS
    wait = MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    for attempt in range(4):
        _last_request = time.monotonic()
        r = requests.get(API, params=params, headers=HEADERS, timeout=60)
        N_REQUESTS += 1
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def fetch_pages(titles: list[str], max_age_h: float | None = None) -> dict[str, dict]:
    """Return {requested title: {"title": resolved, "content": wikitext or None}}.

    Cached per requested title (missing pages are cached with content None). With
    ``max_age_h``, cache entries older than that are refetched.
    """
    PAGES.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict] = {}
    todo = []
    for t in dict.fromkeys(titles):
        p = _cache_path(t)
        fresh = max_age_h is None or (p.exists() and time.time() - p.stat().st_mtime < max_age_h * 3600)
        if p.exists() and fresh:
            out[t] = json.loads(p.read_text(encoding="utf-8"))
        else:
            todo.append(t)
    for i in range(0, len(todo), 50):
        chunk = todo[i : i + 50]
        data = _api_get(
            {
                "action": "query",
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "titles": "|".join(chunk),
                "redirects": 1,
                "format": "json",
                "formatversion": 2,
            }
        )
        q = data.get("query", {})
        # Map requested -> normalized -> redirect target.
        alias = {t: t for t in chunk}
        for n in q.get("normalized", []):
            for k, v in alias.items():
                if v == n["from"]:
                    alias[k] = n["to"]
        for rd in q.get("redirects", []):
            for k, v in alias.items():
                if v == rd["from"]:
                    alias[k] = rd["to"]
        by_title = {}
        for pg in q.get("pages", []):
            content = None
            if not pg.get("missing") and pg.get("revisions"):
                content = pg["revisions"][0]["slots"]["main"]["content"]
            by_title[pg["title"]] = content
        for t in chunk:
            rec = {"title": alias[t], "content": by_title.get(alias[t])}
            _cache_path(t).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            out[t] = rec
    return out


def cached_query(params: dict, max_age_h: float | None = None) -> dict:
    """Run an ``action=query`` request (e.g. list=categorymembers), cached by its params."""
    params = {"action": "query", "format": "json", "formatversion": 2, **params}
    key = json.dumps(params, sort_keys=True)
    p = RAW / "queries" / f"{hashlib.md5(key.encode()).hexdigest()}.json"
    if p.exists() and (max_age_h is None or time.time() - p.stat().st_mtime < max_age_h * 3600):
        return json.loads(p.read_text(encoding="utf-8"))["response"]
    data = _api_get(params)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"params": params, "response": data}, ensure_ascii=False), "utf-8")
    return data


def fetch_page(title: str) -> str | None:
    return fetch_pages([title])[title]["content"]


# --------------------------------------------------------------------------- wikitext helpers

COMMENT = re.compile(r"<!--.*?-->", re.S)
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
DATE_LONG = re.compile(r"([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})")
DATE_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
YEAR = re.compile(r"(?<!\d)(19|20)\d\d(?!\d)")


def parse_date(s: str) -> str | None:
    """'October 3, 2026 - 11:00 {{Abbr/CEST}}' or '2026-10-03' -> '2026-10-03'."""
    m = DATE_ISO.search(s)
    if m:
        return m.group(0)
    m = DATE_LONG.search(s)
    if m and m.group(1).lower()[:3] in {k[:3] for k in MONTHS}:
        month = next(v for k, v in MONTHS.items() if k[:3] == m.group(1).lower()[:3])
        return f"{int(m.group(3)):04d}-{month:02d}-{int(m.group(2)):02d}"
    return None


def find_templates(text: str, name: str) -> list[tuple[int, int, str]]:
    """Return (start, end, inner) of every top-level-or-nested ``{{name ...}}`` call."""
    out = []
    pat = re.compile(r"\{\{\s*" + re.escape(name) + r"\s*(?=[|}])", re.I)
    for m in pat.finditer(text):
        depth, i = 0, m.start()
        while i < len(text) - 1:
            two = text[i : i + 2]
            if two == "{{":
                depth += 1
                i += 2
            elif two == "}}":
                depth -= 1
                i += 2
                if depth == 0:
                    out.append((m.start(), i, text[m.start() + 2 : i - 2]))
                    break
            else:
                i += 1
    return out


def split_params(inner: str) -> tuple[list[str], dict[str, str]]:
    """Split a template's inner text at depth-0 pipes -> (positional, named)."""
    parts, depth, buf, i = [], 0, [], 0
    while i < len(inner):
        two = inner[i : i + 2]
        if two in ("{{", "[["):
            depth += 1
            buf.append(two)
            i += 2
        elif two in ("}}", "]]"):
            depth -= 1
            buf.append(two)
            i += 2
        elif inner[i] == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
        else:
            buf.append(inner[i])
            i += 1
    parts.append("".join(buf))
    pos, named = [], {}
    for p in parts[1:]:  # parts[0] is the template name
        k, eq, v = p.partition("=")
        if eq and re.fullmatch(r"\s*[\w ]+\s*", k):
            named[k.strip()] = v.strip()
        else:
            pos.append(p.strip())
    return pos, named


def infobox(text: str) -> dict[str, str]:
    for name in ("Infobox league", "HiddenDataBox"):
        t = find_templates(text, name)
        if t:
            return split_params(t[0][2])[1]
    return {}


TRUE = {"true", "1", "yes", "y"}


def _int(v: str | None) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return 0


def parse_map(inner: str) -> dict | None:
    """Return {"map", "s1", "s2"} for a played map, else None (skipped / unplayed)."""
    _, p = split_params(inner)
    if p.get("finished", "").lower() == "skip":
        return None
    if p.get("score1", "") != "" or p.get("score2", "") != "":
        s1, s2 = _int(p.get("score1")), _int(p.get("score2"))
    else:
        s1 = _int(p.get("t1t")) + _int(p.get("t1ct"))
        s2 = _int(p.get("t2t")) + _int(p.get("t2ct"))
        for n in range(1, 30):
            if f"o{n}t1t" not in p and f"o{n}t1ct" not in p:
                break
            s1 += _int(p.get(f"o{n}t1t")) + _int(p.get(f"o{n}t1ct"))
            s2 += _int(p.get(f"o{n}t2t")) + _int(p.get(f"o{n}t2ct"))
    w = p.get("winner", "").strip()
    if s1 == s2 and w not in ("1", "2"):
        return None
    if w not in ("1", "2"):
        w = "1" if s1 > s2 else "2"
    return {"map": re.sub(r"\s+", " ", p.get("map", "")).strip() or "Unknown", "s1": s1, "s2": s2,
            "winner": int(w)}


def opponent(v: str) -> tuple[str | None, str]:
    """``{{TeamOpponent|code|score=2}}`` -> (code, score)."""
    t = find_templates(v, "TeamOpponent")
    if not t:
        return None, ""
    pos, named = split_params(t[0][2])
    code = named.get("template") or (pos[0] if pos else "")
    code = re.sub(r"\s+", " ", code).strip().lower()
    return (code or None), named.get("score", "")


def parse_matches(text: str) -> tuple[list[dict], int]:
    """Parse every finished ``{{Match}}`` on a page. Returns (matches, n_skipped_unclear)."""
    text = COMMENT.sub("", text)
    lists = []
    for s, e, inner in find_templates(text, "Matchlist"):
        _, p = split_params(inner)
        lists.append((s, e, p.get("id", ""), p.get("matchsection", "") or p.get("title", "")))
    out, skipped = [], 0
    for s, e, inner in find_templates(text, "Match"):
        _, p = split_params(inner)
        c1, sc1 = opponent(p.get("opponent1", ""))
        c2, sc2 = opponent(p.get("opponent2", ""))
        flagged = p.get("finished", "").lower() in TRUE or p.get("winner", "") in ("1", "2")
        if not (c1 and c2):
            continue
        if p.get("walkover") or str(p.get("winner", "")).lower() in ("ff", "wo") or \
                {sc1.upper(), sc2.upper()} & {"W", "FF", "L", "DQ"}:
            continue  # forfeit
        map_keys = sorted((k for k in p if re.fullmatch(r"map\d+", k)), key=lambda k: int(k[3:]))
        maps = []
        for k in map_keys:
            t = find_templates(p[k], "Map")
            if t:
                m = parse_map(t[0][2])
                if m:
                    maps.append(m)
        w1 = sum(m["winner"] == 1 for m in maps)
        w2 = sum(m["winner"] == 2 for m in maps)
        if not maps and (sc1 or sc2):
            w1, w2 = _int(sc1), _int(sc2)
        if not flagged and not maps:
            continue
        wf = p.get("winner", "")
        if wf in ("1", "2"):
            winner = int(wf)
        elif w1 != w2:
            winner = 1 if w1 > w2 else 2
        else:
            skipped += 1  # draw (e.g. Bo2 1-1) or no result data
            continue
        best_of = _int(p.get("bestof")) or len(map_keys) or 1
        # Some finished matches lack finished=true; accept them once a side has a majority.
        if not flagged and max(w1, w2) < best_of // 2 + 1:
            continue
        best_of = max(best_of, 2 * max(w1, w2) - 1)
        ml = next((x for x in lists if x[0] < s < x[1]), None)
        out.append({
            "date": parse_date(p.get("date", "")),
            "team1": c1, "team2": c2, "winner": winner,
            "score1": w1, "score2": w2, "best_of": best_of, "maps": maps,
            "matchlist_id": ml[2] if ml else "", "section": ml[3] if ml else "",
        })
    return out, skipped


# --------------------------------------------------------------------------- discovery

def category_members(cat: str) -> list[str]:
    titles, cont = [], {}
    while True:
        d = cached_query({"list": "categorymembers", "cmtitle": cat, "cmlimit": "max",
                          "cmnamespace": 0, "cmprop": "title", **cont}, max_age_h=24)
        titles += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            return titles
        cont = d["continue"]


def subpages(title: str, max_age_h: float | None) -> list[str]:
    titles, cont = [], {}
    while True:
        d = cached_query({"list": "allpages", "apprefix": title + "/", "aplimit": "max",
                          "apfilterredir": "nonredirects", **cont}, max_age_h=max_age_h)
        titles += [p["title"] for p in d["query"]["allpages"]]
        if "continue" not in d:
            return titles
        cont = d["continue"]


TIER_CATS = {"S-Tier": "Category:S-Tier Tournaments", "A-Tier": "Category:A-Tier Tournaments",
             "B-Tier": "Category:B-Tier Tournaments", "C-Tier": "Category:C-Tier Tournaments"}
# C-Tier is only pulled for the recent window: several Major teams (Imperial, Lynn Vision,
# Chinggis Warriors) played only C-Tier events in Oct-Nov 2026. Filter on `tier` if unwanted.
C_TIER_FROM = "2026-10-01"
TODAY = date.today().isoformat()


def recent(edate: str | None, days: int = 21) -> bool:
    """True for tournaments that ended recently or not yet (their pages may still change)."""
    if not edate:
        return True
    return (date.fromisoformat(TODAY) - date.fromisoformat(edate)).days <= days


def discover_tournaments(since: str) -> list[dict]:
    """Main tournament pages of S/A/B tier overlapping [since, today]."""
    y1 = int(TODAY[:4]) + 1
    cands: dict[str, str] = {}
    for tier, cat in TIER_CATS.items():
        lo = max(since, C_TIER_FROM) if tier == "C-Tier" else since
        for t in category_members(cat):
            years = [int(m.group(0)) for m in YEAR.finditer(t)]
            if tier == "C-Tier" and not years:
                continue  # keep the C-Tier sweep small: dated titles only
            if not years or any(int(lo[:4]) <= y <= y1 for y in years):
                cands.setdefault(t, tier)
    pages = fetch_pages(list(cands))
    out = []
    for t, tier in cands.items():
        text = pages[t]["content"]
        if not text:
            continue
        ib = infobox(COMMENT.sub("", text))
        sd, ed = parse_date(ib.get("sdate", "")), parse_date(ib.get("edate", ""))
        if not sd:
            continue
        ed = ed or sd
        if ed < (max(since, C_TIER_FROM) if tier == "C-Tier" else since) or sd > TODAY:
            continue
        game = ib.get("game", "").lower()
        if game and game != "cs2":
            continue
        if not game and sd < "2023-10-15":  # CS:GO -> CS2 switchover; untagged pages are CS:GO
            continue
        out.append({"title": t, "tier": ib.get("liquipediatier", tier) or tier,
                    "name": ib.get("name", t), "sdate": sd, "edate": ed,
                    "type": ib.get("type", "").lower()})
    return out


# --------------------------------------------------------------------------- team mapping

# Liquipedia code -> VRS name, for names the fuzzy matcher can't resolve.
MANUAL = {
    "vit": "Vitality", "vitality": "Vitality", "team vitality": "Vitality",
    "1w": "1win", "1win": "1win", "the one": "1win",
    "gosh": "BC.Game", "bc.game": "BC.Game", "bcg": "BC.Game", "bc.game esports": "BC.Game",
    "pvision": "PARIVISION", "parivision": "PARIVISION",
    "betboom": "BETBOOM", "bb": "BETBOOM", "bb team": "BETBOOM", "betboom team": "BETBOOM",
    "tl": "Liquid", "ef": "Eternal Fire", "100t": "100 Thieves", "fq": "FlyQuest",
    "chinggis": "Chinggis Warriors", "lvg": "Lynn Vision", "imp": "Imperial",
    "imperial esports": "Imperial", "nip": "Ninjas in Pyjamas", "navi": "Natus Vincere",
    "mouz": "MOUZ", "mousesports": "MOUZ", "mongolz": "The MongolZ", "the mongolz": "The MongolZ",
    "faze": "FaZe", "g2": "G2", "gl": "GamerLegion", "fnc": "fnatic", "fnatic": "fnatic",
    "vp": "Virtus.pro", "c9": "Cloud9", "col": "Complexity", "complexity": "Complexity",
    "ence": "ENCE", "pain": "paiN", "mibr": "MIBR", "big": "BIG", "saw": "SAW",
    "huns": "The Huns", "amkal esports": "AMKAL", "koi": "KOI", "apeks": "Apeks",
    "gaimin": "Gaimin Gladiators", "jjh": "JiJieHao", "ic": "Inner Circle",
    "ic academy": "Inner Circle Academy", "mnxt": "MOUZ NXT", "bestia.a": "BESTIA Academy",
    "ex-keyd": "ex-Keyd Stars", "bounty.br": "Bounty Hunters",
    "theone 1337": "the ONE",  # Mongolian team, not 1win
}
# VRS renamed some teams mid-history; Liquipedia keeps one code. (vrs_new, vrs_old, last day old)
DATED_ALIASES = [("BETBOOM", "BetBoom", "2026-03-24")]
STRIP = re.compile(r"^(team )|( (esports?|gaming|clan|club|e-sports|gg|team))+$")


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()
    s = re.sub(r"\s*\((cs|counter-strike|team|[a-z ]+ team)\)$", "", s)
    s = STRIP.sub("", s).strip()
    return re.sub(r"[^a-z0-9]+", "", s)


def load_vrs_names() -> dict[str, str]:
    """{normalized: VRS name}, preferring the name with the most matches since 2025."""
    vrs = pd.read_csv(OUT / "matches.csv")
    names = pd.concat([vrs.winner, vrs.loser])
    recent_counts = pd.concat([vrs.winner[vrs.date >= "2025-01-01"],
                               vrs.loser[vrs.date >= "2025-01-01"]]).value_counts()
    best: dict[str, str] = {}
    for n in names.unique():
        k = norm(n)
        if k and (k not in best or recent_counts.get(n, 0) > recent_counts.get(best[k], 0)):
            best[k] = n
    return best


def build_team_map(codes: set[str], vrs: dict[str, str]) -> pd.DataFrame:
    """Resolve Liquipedia team-template codes to VRS names.

    Order: manual table -> code itself -> legacy ``Template:TeamShort/<code>`` redirect target
    -> wiki page titled like the code (if it is a team page).
    """
    vrs_set = set(pd.read_csv(OUT / "matches.csv", usecols=["winner", "loser"]).stack())
    rows, todo = {}, []
    for c in sorted(codes):
        if c in MANUAL and MANUAL[c] in vrs_set:
            rows[c] = (MANUAL[c], c, "manual")
        elif norm(c) in vrs:
            rows[c] = (vrs[norm(c)], c, "code")
        elif re.search(r"\.[a-z]{2}$", c) and norm(c[:-3]) in vrs:  # country tag, e.g. unity.sk
            rows[c] = (vrs[norm(c[:-3])], c, "code")
        else:
            todo.append(c)
    tmpl = fetch_pages([f"Template:TeamShort/{c}" for c in todo])
    left = []
    for c in todo:
        target = tmpl[f"Template:TeamShort/{c}"]["title"].split("/", 1)[1]
        if target != c and (target in MANUAL or norm(target) in vrs):
            rows[c] = (MANUAL.get(target) or vrs[norm(target)], target, "template")
        else:
            left.append(c)
    pages = fetch_pages([c[0].upper() + c[1:] for c in left])
    for c in left:
        rec = pages[c[0].upper() + c[1:]]
        is_team = bool(rec["content"]) and "{{Infobox team" in rec["content"]
        lp_name = rec["title"] if is_team else c
        if is_team and norm(lp_name) in vrs:
            rows[c] = (vrs[norm(lp_name)], lp_name, "team page")
        else:
            rows[c] = (None, lp_name, "unmapped")
    return pd.DataFrame(
        [{"code": c, "liquipedia_name": n, "vrs_name": v, "method": m} for c, (v, n, m) in rows.items()]
    )


def to_vrs(code: str, day: str | None, tmap: dict, lp: dict) -> tuple[str, bool]:
    name = tmap.get(code)
    if not name:
        return lp.get(code, code), False
    for new, old, last in DATED_ALIASES:
        if name == new and day and day <= last:
            return old, True
    return name, True


# --------------------------------------------------------------------------- pages -> rows

def collect_pages(tournaments: list[dict], extra: list[str]) -> list[dict]:
    """Expand tournaments to [main page + subpages], each with event metadata."""
    out = []
    for t in tournaments:
        age = 6 if recent(t["edate"]) else None
        titles = [t["title"]] + subpages(t["title"], max_age_h=age)
        pages = fetch_pages(titles, max_age_h=age)
        for title in titles:
            text = pages[title]["content"]
            if not text:
                continue
            if title != t["title"] and find_templates(text, "Infobox league"):
                continue  # separate tournament (e.g. qualifier) with its own tier
            ib = infobox(COMMENT.sub("", text)) if title != t["title"] else {}
            out.append({**t, "page": title, "text": text,
                        "tier": ib.get("liquipediatier") or t["tier"],
                        "name": ib.get("name") or t["name"],
                        "sdate": parse_date(ib.get("sdate", "")) or t["sdate"]})
    seen = {p["page"] for p in out}
    for title in extra:
        if title not in seen:
            text = fetch_page(title)
            ib = infobox(COMMENT.sub("", text))
            out.append({"title": title, "page": title, "text": text, "tier": ib.get("liquipediatier"),
                        "name": ib.get("name", title), "sdate": parse_date(ib.get("sdate", "")),
                        "edate": parse_date(ib.get("edate", "")), "type": "offline"})
    return out


def clean(name: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(name)).strip()


def lan_flag(kind: str) -> bool | None:
    if kind == "offline":
        return True
    if kind == "online":
        return False
    return None


def build_rows(pages: list[dict], since: str) -> tuple[list[dict], int]:
    rows, seen, skipped = [], set(), 0
    for pg in pages:
        matches, sk = parse_matches(pg["text"])
        skipped += sk
        for m in matches:
            day = m["date"] or pg["sdate"]
            if not day or day < since or day > TODAY:
                continue
            w, l = (1, 2) if m["winner"] == 1 else (2, 1)
            maps = ";".join(f"{x['map']}:{x[f's{w}']}-{x[f's{l}']}" for x in m["maps"])
            key = (day, m[f"team{w}"], m[f"team{l}"], m[f"score{w}"], m[f"score{l}"], maps)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"date": day, "winner_code": m[f"team{w}"], "loser_code": m[f"team{l}"],
                         "winner_score": m[f"score{w}"], "loser_score": m[f"score{l}"],
                         "best_of": m["best_of"], "maps": maps, "event": clean(pg["name"]),
                         "tier": pg["tier"], "lan": lan_flag(pg["type"]), "source_page": pg["page"]})
    return rows, skipped


# --------------------------------------------------------------------------- Major Swiss stages

MAJOR_STAGES = [
    ("PGL Major Copenhagen 2024", "PGL/2024/Copenhagen/Opening Stage"),
    ("PGL Major Copenhagen 2024", "PGL/2024/Copenhagen/Elimination Stage"),
    ("Perfect World Shanghai Major 2024", "Perfect World/Major/2024/Shanghai/Opening Stage"),
    ("Perfect World Shanghai Major 2024", "Perfect World/Major/2024/Shanghai/Elimination Stage"),
    ("BLAST.tv Austin Major 2025", "BLAST/Major/2025/Austin/Stage 1"),
    ("BLAST.tv Austin Major 2025", "BLAST/Major/2025/Austin/Stage 2"),
    ("BLAST.tv Austin Major 2025", "BLAST/Major/2025/Austin/Stage 3"),
    ("StarLadder Budapest Major 2025", "StarLadder/2025/Major/Stage 1"),
    ("StarLadder Budapest Major 2025", "StarLadder/2025/Major/Stage 2"),
    ("StarLadder Budapest Major 2025", "StarLadder/2025/Major/Stage 3"),
    ("IEM Cologne Major 2026", "Intel Extreme Masters/2026/Cologne/Stage 1"),
    ("IEM Cologne Major 2026", "Intel Extreme Masters/2026/Cologne/Stage 2"),
    ("IEM Cologne Major 2026", "Intel Extreme Masters/2026/Cologne/Stage 3"),
]


def stage_seeds(text: str) -> list[str]:
    """Team codes in seed order: SwissTableLeague teamN, or SwissStandings TeamOpponent order."""
    t = find_templates(text, "SwissTableLeague")
    if t:
        _, p = split_params(t[0][2])
        n = sorted((int(k[4:]), v) for k, v in p.items() if re.fullmatch(r"team\d+", k))
        return [re.sub(r"\s+", " ", v).strip().lower() for _, v in n]
    t = find_templates(text, "SwissStandings")
    if t:
        pos, _ = split_params(t[0][2])
        return [c for c in (opponent(x)[0] for x in pos) if c]
    return []


def parse_stage(event: str, title: str, tmap: dict, lp: dict) -> dict:
    text = COMMENT.sub("", fetch_page(title))
    ib = infobox(text)
    seeds = stage_seeds(text)
    rounds_by_id = {}
    ss = find_templates(text, "SwissStandings")
    if ss:
        _, p = split_params(ss[0][2])
        for k, v in p.items():
            if re.fullmatch(r"round\d+", k):
                for gid in re.findall(r"matchgroups=([^|}]+)", v):
                    for g in gid.split(","):
                        rounds_by_id[g.strip()] = int(k[5:])
    matches, _ = parse_matches(text)
    out_matches = []
    for m in matches:
        r = rounds_by_id.get(m["matchlist_id"])
        if r is None:
            mm = re.search(r"Round\s*(\d+)", m["section"])
            r = int(mm.group(1)) if mm else None
        n1 = to_vrs(m["team1"], m["date"], tmap, lp)[0]
        n2 = to_vrs(m["team2"], m["date"], tmap, lp)[0]
        out_matches.append({"round": r, "team1": n1, "team2": n2,
                            "winner": n1 if m["winner"] == 1 else n2, "best_of": m["best_of"]})
    day = ib.get("sdate")
    teams = []
    for i, c in enumerate(seeds, 1):
        name = to_vrs(c, day, tmap, lp)[0]
        wins = sum(x["winner"] == name for x in out_matches)
        losses = sum(name in (x["team1"], x["team2"]) and x["winner"] != name for x in out_matches)
        teams.append({"name": name, "seed": i, "wins": wins, "losses": losses})
    # Verify seeding: round-1 pairings must be seed i vs seed i+8.
    seed_of = {t["name"]: t["seed"] for t in teams}
    r1 = [x for x in out_matches if x["round"] == 1]
    ok = len(r1) == 8 and all(
        abs(seed_of.get(x["team1"], 0) - seed_of.get(x["team2"], 99)) == 8 for x in r1)
    return {"event": event, "stage": clean(ib.get("name", title).split(":")[-1]),
            "start_date": day, "source_page": title, "seeding_verified": ok,
            "teams": teams, "matches": out_matches}


# --------------------------------------------------------------------------- main

def major_teams() -> list[str]:
    from cs2p.teams import STAGE1, STAGE2_INVITES, STAGE3_INVITES

    return [v for _, v in STAGE1 + STAGE2_INVITES + STAGE3_INVITES]


def log(msg: str) -> None:
    print(msg, flush=True)


def main(since: str = "2023-09-27") -> None:
    t0 = time.monotonic()
    tournaments = discover_tournaments(since)
    log(f"{len(tournaments)} S/A/B-tier tournaments overlap {since}..{TODAY}")
    pages = collect_pages(tournaments, [t for _, t in MAJOR_STAGES])
    log(f"{len(pages)} pages (main + subpages)")
    raw, skipped = build_rows(pages, since)
    codes = {r["winner_code"] for r in raw} | {r["loser_code"] for r in raw}
    for _, t in MAJOR_STAGES:
        text = COMMENT.sub("", fetch_page(t))
        codes |= set(stage_seeds(text))
        codes |= {c for m in parse_matches(text)[0] for c in (m["team1"], m["team2"])}
    vrs = load_vrs_names()
    team_map = build_team_map(codes, vrs)
    team_map.sort_values("code").to_csv(OUT / "liquipedia_team_map.csv", index=False)
    tmap = {c: (v if isinstance(v, str) else None) for c, v in zip(team_map.code, team_map.vrs_name)}
    lp = dict(zip(team_map.code, team_map.liquipedia_name))

    rows = []
    for r in raw:
        w, wok = to_vrs(r.pop("winner_code"), r["date"], tmap, lp)
        l, lok = to_vrs(r.pop("loser_code"), r["date"], tmap, lp)
        rows.append({"date": r["date"], "winner": w, "loser": l, **r, "_mapped": wok and lok})
    df = pd.DataFrame(rows).sort_values(["date", "event", "winner"]).reset_index(drop=True)
    mapped = df.pop("_mapped")
    df.to_csv(OUT / "matches_liquipedia.csv", index=False)

    stages = [parse_stage(e, t, tmap, lp) for e, t in MAJOR_STAGES]
    (OUT / "past_major_stages.json").write_text(json.dumps(stages, indent=1, ensure_ascii=False), "utf-8")

    # ---- summary
    log(f"\nmatches_liquipedia.csv: {len(df)} rows ({skipped} draws/no-result matches skipped)")
    log("matches by month:\n" + df.date.str[:7].value_counts().sort_index().to_string())
    log(f"rows with both teams mapped to VRS names: {mapped.mean():.1%}")
    log(f"tier counts: {df.tier.value_counts().to_dict()}")
    un = team_map[team_map.vrs_name.isna()]
    counts = pd.concat([df.winner, df.loser]).value_counts()
    un = un.assign(rows=un.liquipedia_name.map(counts).fillna(0).astype(int))
    un = un.sort_values("rows", ascending=False)
    log(f"unmapped codes: {len(un)} (top by rows):\n" + un.head(40).to_string(index=False))
    rdf = df[df.date >= "2026-10-01"]
    rc = pd.concat([rdf.winner, rdf.loser]).value_counts()
    log("\nMajor teams, matches since 2026-10-01:")
    log(", ".join(f"{t}={rc.get(t, 0)}" for t in major_teams()))
    log("\npast Major stages:")
    for s in stages:
        rec = " ".join(f"{t['seed']}.{t['name']}({t['wins']}-{t['losses']})" for t in s["teams"])
        log(f"  {s['event']} / {s['stage']} {s['start_date']} matches={len(s['matches'])} "
            f"seeding_verified={s['seeding_verified']}\n    {rec}")
    log(f"\nAPI requests this run: {N_REQUESTS}, elapsed {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2023-09-27", help="earliest match date (YYYY-MM-DD)")
    main(ap.parse_args().since)

"""Rebuild a series-level match history from Valve Regional Standings detail pages.

Each monthly snapshot lists, for every ranked roster, the matches of the last ~6 months.
Match IDs are only unique within a snapshot, so we dedupe across snapshots on
(date, winner, loser), keeping the max multiplicity seen in any single snapshot.
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

VRS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "vrs" / "live"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed"

ROW = re.compile(r"^\|\s*\d+\s*\|\s*(\d+)\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(.+?)\s*\|\s*([WL])\s*\|")


def parse_detail(path: Path) -> tuple[str, str, list[tuple[str, str, str, str]]]:
    """Return (team, roster, [(match_id, date, opponent, W/L), ...]) for one detail page."""
    text = path.read_text(encoding="utf-8")
    team = re.search(r"Team Name:\s*(.+?)<br", text).group(1).strip()
    roster = re.search(r"Roster:\s*(.+?)<br", text).group(1).strip()
    rows = []
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            rows.append(m.groups())
    return team, roster, rows


def _disambiguate(team: str, roster: str, seen: dict[str, str]) -> str:
    """VRS sometimes lists two different rosters under one name in the same snapshot.

    The higher-ranked roster keeps the name; later ones get a suffix from their first player.
    Pages/rows must be visited in rank order for this to be consistent across files.
    """
    key = team.casefold()
    if key in seen and seen[key] != roster:
        return f"{team} ({roster.split(',')[0].strip()})"
    seen.setdefault(key, roster)
    return team


def parse_snapshot(snap_dir: Path) -> dict[str, dict]:
    """Return {match_id: record} for one snapshot, merging both sides of each match.

    A team's own detail page is authoritative for its name; the opponent column is only
    used when the other side has no page in this snapshot.
    """
    raw: dict[str, dict] = {}
    seen: dict[str, str] = {}
    for f in sorted(snap_dir.glob("*.md")):  # file names start with the rank
        team, roster, rows = parse_detail(f)
        team = _disambiguate(team, roster, seen)
        for mid, date, opp, wl in rows:
            rec = raw.setdefault(mid, {"date": date, "own": {}, "opp": {}, "roster": {}})
            other = "L" if wl == "W" else "W"
            rec["own"][wl] = team
            rec["opp"].setdefault(other, opp)
            rec["roster"][wl] = roster
    matches = {}
    for mid, rec in raw.items():
        w = rec["own"].get("W", rec["opp"].get("W"))
        l = rec["own"].get("L", rec["opp"].get("L"))
        if w and l:
            matches[mid] = {"date": rec["date"], "winner": w, "loser": l,
                            "winner_roster": rec["roster"].get("W"), "loser_roster": rec["roster"].get("L")}
    return matches


def _base(name: str) -> str:
    """Case-insensitive name without the disambiguation suffix, used as the dedupe key."""
    return re.sub(r" \(.*\)$", "", name).casefold()


def build_matches() -> pd.DataFrame:
    """Union all snapshots into one deduplicated match table.

    Dedupe key is case-insensitive (VRS changes capitalisation, e.g. BetBoom/BETBOOM);
    the most recent spelling of each name is kept.
    """
    best: dict[tuple, int] = {}
    rosters: dict[tuple, dict] = {}
    spelling: dict[str, str] = {}
    names: dict[tuple, tuple[str, str]] = {}
    for snap_dir in sorted(VRS_DIR.glob("*/details/*"), key=lambda p: p.name):
        snap = parse_snapshot(snap_dir)
        for r in snap.values():
            spelling[r["winner"].casefold()] = r["winner"]
            spelling[r["loser"].casefold()] = r["loser"]
        key = lambda r: (r["date"], _base(r["winner"]), _base(r["loser"]))  # noqa: E731
        counts = Counter(key(r) for r in snap.values())
        for k, n in counts.items():
            best[k] = max(best.get(k, 0), n)
        for r in snap.values():
            names[key(r)] = (r["winner"], r["loser"])  # most recent snapshot's naming wins
            slot = rosters.setdefault(key(r), {})
            for side in ("winner_roster", "loser_roster"):
                if r.get(side):
                    slot[side] = r[side]
    rows = []
    for k, n in best.items():
        w, l = (spelling[x.casefold()] for x in names[k])
        rows += [{"date": k[0], "winner": w, "loser": l, **rosters.get(k, {})}] * n
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def build_vrs_points() -> pd.DataFrame:
    """Return VRS points per (snapshot date, team), used as the baseline model."""
    rows = []
    for f in sorted(VRS_DIR.glob("*/standings_global_*.md")):
        snap = pd.to_datetime(re.search(r"(\d{4}_\d{2}_\d{2})", f.name).group(1), format="%Y_%m_%d")
        seen: dict[str, str] = {}
        for line in f.read_text(encoding="utf-8").splitlines():  # rows are in rank order
            parts = [p.strip() for p in line.split("|")]
            if len(parts) > 4 and parts[1].isdigit():
                team = _disambiguate(parts[3], parts[4], seen)
                rows.append({"snapshot": snap, "team": team, "points": float(parts[2])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    m = build_matches()
    m.to_csv(OUT / "matches.csv", index=False)
    p = build_vrs_points()
    p.to_csv(OUT / "vrs_points.csv", index=False)
    print(f"{len(m)} matches {m.date.min().date()}..{m.date.max().date()}, {len(p)} VRS rows")

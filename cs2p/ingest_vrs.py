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


def parse_snapshot(snap_dir: Path) -> dict[str, dict]:
    """Return {match_id: record} for one snapshot, merging both sides of each match."""
    matches: dict[str, dict] = {}
    for f in snap_dir.glob("*.md"):
        team, roster, rows = parse_detail(f)
        for mid, date, opp, wl in rows:
            rec = matches.setdefault(mid, {"date": date})
            winner, loser = (team, opp) if wl == "W" else (opp, team)
            rec["winner"], rec["loser"] = winner, loser
            rec["winner_roster" if wl == "W" else "loser_roster"] = roster
    return matches


def build_matches() -> pd.DataFrame:
    """Union all snapshots into one deduplicated match table."""
    best: dict[tuple, int] = {}
    rosters: dict[tuple, dict] = {}
    for snap_dir in sorted(VRS_DIR.glob("*/details/*")):
        snap = parse_snapshot(snap_dir)
        counts = Counter((r["date"], r["winner"], r["loser"]) for r in snap.values())
        for key, n in counts.items():
            best[key] = max(best.get(key, 0), n)
        for r in snap.values():
            k = (r["date"], r["winner"], r["loser"])
            slot = rosters.setdefault(k, {})
            for side in ("winner_roster", "loser_roster"):
                if r.get(side):
                    slot[side] = r[side]
    rows = []
    for (date, w, l), n in best.items():
        rows += [{"date": date, "winner": w, "loser": l, **rosters.get((date, w, l), {})}] * n
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def build_vrs_points() -> pd.DataFrame:
    """Return VRS points per (snapshot date, team), used as the baseline model."""
    rows = []
    for f in sorted(VRS_DIR.glob("*/standings_global_*.md")):
        snap = pd.to_datetime(re.search(r"(\d{4}_\d{2}_\d{2})", f.name).group(1), format="%Y_%m_%d")
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) > 4 and parts[1].isdigit():
                rows.append({"snapshot": snap, "team": parts[3], "points": float(parts[2])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    m = build_matches()
    m.to_csv(OUT / "matches.csv", index=False)
    p = build_vrs_points()
    p.to_csv(OUT / "vrs_points.csv", index=False)
    print(f"{len(m)} matches {m.date.min().date()}..{m.date.max().date()}, {len(p)} VRS rows")

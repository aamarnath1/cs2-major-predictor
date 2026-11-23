"""Test shrinking odds for recent VRS roster changes and teams with few prior results."""

import argparse
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from cs2p.elo import PROC, load_matches
from cs2p.features import build_features
from cs2p.ingest_vrs import VRS_DIR
from cs2p.zoo import add_bt, logit, sig

START = pd.Timestamp("2026-03-01")
THROUGH = pd.Timestamp("2026-11-22")
SHRINK = 0.20
FEW_GAMES = 15
MAX_SNAPSHOT_AGE = pd.Timedelta(days=45)


@dataclass(frozen=True)
class Snapshot:
    """VRS roster and points published together in one archived roster table."""

    effective: pd.Timestamp
    available: pd.Timestamp
    rosters: dict[str, frozenset[str]]
    points: dict[str, int]


def load_snapshots(manifest_path: Path) -> list[Snapshot]:
    """Use the last-changing commit for each file at the pinned Valve revision."""
    manifest = json.loads(manifest_path.read_bytes())
    current = subprocess.check_output(
        ["git", "-C", str(VRS_DIR.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    if current != manifest["source_commit"]:
        raise ValueError("The local VRS revision must match the availability manifest.")
    paths = sorted(VRS_DIR.glob("*/standings_global_*.md"))
    records = {row["file"]: row for row in manifest["files"]}
    if len(records) != len(paths) or records.keys() != {p.name for p in paths}:
        raise ValueError("The availability manifest must cover every local global snapshot.")

    snapshots = []
    for path in paths:
        effective = pd.to_datetime(path.stem.removeprefix("standings_global_"), format="%Y_%m_%d")
        committed = pd.Timestamp(records[path.name]["committed_at"])
        available = max(effective, committed.tz_convert("UTC").tz_localize(None).normalize())
        roster_sets: dict[str, set[frozenset[str]]] = defaultdict(set)
        points: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            cells = [cell.strip() for cell in line.split("|")]
            if len(cells) < 5 or not cells[1].isdigit() or not cells[2].isdigit():
                continue
            team = cells[3].casefold()
            roster = frozenset(player.strip().casefold() for player in cells[4].split(","))
            roster_sets[team].add(roster)
            points[team] = max(points.get(team, 0), int(cells[2]))
        rosters = {
            team: next(iter(values))
            for team, values in roster_sets.items()
            if len(values) == 1 and len(next(iter(values))) == 5
            and "" not in next(iter(values))
        }
        if not points:
            raise ValueError(f"Global VRS snapshot has no rankings: {path.name}")
        snapshots.append(Snapshot(effective, available, rosters, points))
    return sorted(snapshots, key=lambda snapshot: snapshot.effective)


def label_roster_context(features: pd.DataFrame, snapshots: list[Snapshot]) -> pd.DataFrame:
    """Compare two fresh, unambiguous rosters that were available before match day."""
    out = features.reset_index(drop=True)
    movement = np.full(len(out), np.nan)
    top = np.zeros(len(out), dtype=bool)
    for day, matches in out.groupby("date"):
        previous = [s for s in snapshots if s.available < day]
        if not previous:
            continue
        latest = previous[-1]
        if day - latest.effective > MAX_SNAPSHOT_AGE:
            continue
        points_a = matches.a.map(latest.points).fillna(0)
        points_b = matches.b.map(latest.points).fillna(0)
        top[matches.index] = (points_a >= 1400) & (points_b >= 1400)
        if len(previous) < 2 or latest.effective - previous[-2].effective > MAX_SNAPSHOT_AGE:
            continue
        older = previous[-2]
        changes = {
            team: len(players - older.rosters[team])
            for team, players in latest.rosters.items() if team in older.rosters
        }
        movement[matches.index] = np.maximum(matches.a.map(changes), matches.b.map(changes))
    out["changes"] = movement
    out["top_vrs"] = top
    out["low_history"] = out[["games_a", "games_b"]].min(axis=1) < FEW_GAMES
    return out


def evaluate(rows: pd.DataFrame) -> dict:
    """Compare the fixed shrink on identical matches, with month-block intervals."""
    if rows.empty:
        return {"n": 0}
    baseline = rows.p.to_numpy()
    adjusted = 0.5 + (1 - SHRINK) * (baseline - 0.5)
    base_loss = -np.log(np.clip(baseline, 1e-6, 1))
    adjusted_loss = -np.log(np.clip(adjusted, 1e-6, 1))
    delta = adjusted_loss - base_loss
    by_month = pd.DataFrame({"month": rows.date.dt.to_period("M"), "delta": delta})
    blocks = by_month.groupby("month").delta.agg(["sum", "size"])
    result = {
        "n": len(rows), "months": len(blocks),
        "base_logloss": float(base_loss.mean()), "shrunk_logloss": float(adjusted_loss.mean()),
        "base_brier": float(((1 - baseline) ** 2).mean()),
        "shrunk_brier": float(((1 - adjusted) ** 2).mean()),
        "delta_logloss": float(delta.mean()),
    }
    if len(blocks) > 1:
        draw = np.random.default_rng(403).integers(len(blocks), size=(5000, len(blocks)))
        means = blocks["sum"].to_numpy()[draw].sum(1) / blocks["size"].to_numpy()[draw].sum(1)
        result["month_block_ci_95"] = np.quantile(means, [0.025, 0.975]).tolist()
    return result


def main(manifest_path: Path) -> None:
    """Run the historical sensitivity and print the results without writing forecasts."""
    snapshots = load_snapshots(manifest_path)
    history = load_matches()
    history = history[history.date <= THROUGH].assign(winner_roster=None, loser_roster=None)
    features, _ = build_features(history, start=START, batch_by_day=True)
    params = json.loads((PROC / "bt_params.json").read_bytes())
    add_bt(features, history, params["half_life_days"], params["prior"])
    bt = sig((features.bt_a - features.bt_b).to_numpy())
    forecasts = np.stack([features.elo_p, features.glicko_p, features.ts_p, bt])
    features["p"] = sig(logit(forecasts).mean(axis=0))
    if features.empty or not np.isfinite(features.p).all():
        raise ValueError("The historical predictions are empty or incomplete.")

    rows = label_roster_context(features, snapshots)
    coverage = rows.changes.notna()
    changed = rows.changes.gt(0)
    groups = {
        "all": rows,
        "known_stable": rows[rows.changes.eq(0)],
        "published_roster_change": rows[changed],
        "top_vrs_roster_change": rows[changed & rows.top_vrs],
        "low_history": rows[rows.low_history],
        "top_vrs_low_history": rows[rows.low_history & rows.top_vrs],
    }
    manifest_lf = manifest_path.read_bytes().replace(b"\r\n", b"\n")
    print(json.dumps({
        "start": str(START.date()), "through": str(THROUGH.date()),
        "shrink": SHRINK, "low_history_prior_series_below": FEW_GAMES,
        "match_count": len(rows), "snapshot_count": len(snapshots),
        "roster_coverage_both_teams": int(coverage.sum()),
        "top_vrs_roster_coverage_both_teams": int((coverage & rows.top_vrs).sum()),
        "top_vrs_matches": int(rows.top_vrs.sum()),
        "availability_lf_sha256": sha256(manifest_lf).hexdigest(),
        "input_sha256": {
            filename: sha256((PROC / filename).read_bytes()).hexdigest()
            for filename in (
                "matches.csv", "matches_liquipedia.csv", "elo_params.json", "bt_params.json"
            )
        },
        "cohorts": {name: evaluate(group) for name, group in groups.items()},
    }, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--availability", type=Path, required=True)
    main(cli.parse_args().availability)

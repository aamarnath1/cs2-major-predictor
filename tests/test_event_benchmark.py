import json
from pathlib import Path

import numpy as np

from cs2p.event_benchmark import published_vrs
from cs2p.swiss import ELIM_03, simulate


def test_vrs_availability_uses_the_later_publication_date(tmp_path: Path) -> None:
    points = tmp_path / "vrs_points.csv"
    points.write_text("snapshot,team,points\n2026-06-01,Team A,1500\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{
        "file": "standings_global_2026_06_01.md",
        "committed_at": "2026-06-02T22:00:00Z",
    }]}), encoding="utf-8")

    rows = published_vrs(points, manifest)

    assert rows.loc[0, "team"] == "team a"
    assert rows.loc[0, "available"].isoformat() == "2026-06-02T00:00:00"


def test_swiss_uses_the_supplied_bo1_probabilities() -> None:
    p = np.full((16, 16), 0.5)
    single_map = np.triu(np.ones((16, 16)), 1)

    outcomes = simulate(p, n_sims=100, seed=1, p_bo1=single_map)

    assert not (outcomes[:, :8] == ELIM_03).any()

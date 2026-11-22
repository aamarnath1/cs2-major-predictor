import numpy as np
import pytest

from cs2p import live
from cs2p.swiss import ADV, ADV_30, simulate
from cs2p.teams import STAGE1


SEEDS = [name for name, _ in STAGE1]


def result(round_number: int, a: int, b: int) -> dict:
    """Build one result with the first team as winner."""
    return {"round": round_number, "a": SEEDS[a], "b": SEEDS[b], "winner": SEEDS[a]}


def test_accepts_complete_round_then_partial_round() -> None:
    first_round = [result(1, i, i + 8) for i in range(8)]
    rounds = live.fixed_rounds(SEEDS, [result(2, 0, 1), *first_round])
    assert [len(round_results) for round_results in rounds] == [8, 1]
    assert rounds[1] == [(0, 1, 0)]
    outcomes = simulate(np.full((16, 16), 0.5), n_sims=40, fixed_rounds=rounds)
    assert np.all(((outcomes == ADV_30) | (outcomes == ADV)).sum(axis=1) == 8)


@pytest.mark.parametrize(
    ("matches", "message"),
    [
        ([result(2, 0, 1)], "Round 1 is incomplete"),
        ([result(1, 0, 8)] * 2, "more than once"),
        ([result(1, 0, 9)], "pairing disagrees with the seeds"),
        ([{**result(1, 0, 8), "winner": SEEDS[1]}], "invalid teams or winner"),
        ([{**result(1, 0, 8), "b": "Foreign team"}], "invalid teams or winner"),
        ([{**result(1, 0, 8), "round": None}], "valid Swiss round"),
        (
            [*[result(1, i, i + 8) for i in range(8)], result(2, 0, 9)],
            "incompatible Swiss records",
        ),
    ],
)
def test_rejects_contradictory_feed(matches: list[dict], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        live.fixed_rounds(SEEDS, matches)


def test_fetch_does_not_silently_discard_an_unmapped_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live, "fetch_pages", lambda *args, **kwargs: {})
    monkeypatch.setattr(live, "_team_maps", lambda: ({}, {}))
    monkeypatch.setattr(
        live,
        "parse_stage",
        lambda *args: {
            "teams": [],
            "start_date": "2026-11-24",
            "matches": [{
                "round": 1,
                "date": "2026-11-24",
                "team1": STAGE1[0][1],
                "team2": STAGE1[8][1],
                "winner": "Unmapped",
            }],
        },
    )
    with pytest.raises(ValueError, match="Invalid Singapore Stage 1 results:"):
        live.fetch_state()


@pytest.mark.parametrize("field", ["pending", "early", "valid", "different"])
def test_fetch_reconciles_later_field_with_previous_stage(
    monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    first_stage_pairs = (
        [(i, i + 8) for i in range(8)],
        [(0, 7), (1, 6), (2, 5), (3, 4), (8, 15), (9, 14), (10, 13), (11, 12)],
        [(0, 3), (1, 2), (4, 11), (5, 10), (6, 9), (7, 8), (12, 15), (13, 14)],
        [(2, 7), (3, 6), (4, 5), (8, 13), (9, 12), (10, 11)],
        [(5, 9), (6, 8), (7, 10)],
    )
    first_stage_results = [
        {"round": rnd, "date": "2026-11-24", "team1": STAGE1[a][1],
         "team2": STAGE1[b][1], "winner": STAGE1[a][1]}
        for rnd, pairs in enumerate(first_stage_pairs, start=1)
        for a, b in pairs
    ]
    invites = live.INVITES[2]
    qualifiers = STAGE1[:8] if field != "different" else STAGE1[1:9]

    def fake_stage(event: str, title: str, tmap: dict, lp: dict) -> dict:
        if title == live.STAGE_PAGES[1]:
            return {
                "teams": [{"name": name} for _, name in STAGE1],
                "start_date": "2026-11-24",
                "matches": first_stage_results if field != "early" else [],
            }
        if title == live.STAGE_PAGES[2] and field != "pending":
            return {
                "teams": [{"name": name} for _, name in invites + qualifiers],
                "start_date": "2026-11-28",
                "matches": [],
            }
        return {"teams": [], "start_date": None, "matches": []}

    monkeypatch.setattr(live, "fetch_pages", lambda *args, **kwargs: {
        live.PLAYOFF_PAGE: {"content": ""},
    })
    monkeypatch.setattr(live, "_team_maps", lambda: ({}, {}))
    monkeypatch.setattr(live, "parse_stage", fake_stage)
    if field == "early":
        with pytest.raises(ValueError, match="previous Swiss stage is incomplete"):
            live.fetch_state()
    elif field == "different":
        with pytest.raises(ValueError, match="field disagrees with invites and previous qualifiers"):
            live.fetch_state()
    else:
        state = live.fetch_state()
        assert state[1]["status"] == "complete"
        assert state[2]["status"] == "upcoming"
        if field == "pending":
            assert state[2]["seeds"] is None
        else:
            assert state[2]["seeds"] == [name for name, _ in invites + qualifiers]

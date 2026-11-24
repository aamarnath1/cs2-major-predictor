from cs2p.run import movement_base


def data(n_matches: int, champ: float, generated: str = "t") -> dict:
    return {
        "generated": generated,
        "stages": [{"stage": 1, "matches": [{}] * n_matches, "teams": [{"team": "A", "p_qualify": champ}]}],
        "playoffs": {"slots": {}},
        "major": [{"team": "A", "champion": champ}],
    }


def test_baseline_advances_only_with_new_results():
    old = data(0, 0.2, "before")
    new = data(8, 0.3, "after")
    base = movement_base(old, new)
    assert base["generated"] == "before" and base["champion"] == {"A": 0.2} and base["qualify"] == {"1": {"A": 0.2}}
    new["movement"] = base
    rerun = data(8, 0.31, "rerun")
    assert movement_base(new, rerun) == base  # no new results: keep the old baseline


def test_no_previous_file():
    assert movement_base(None, data(0, 0.2)) is None

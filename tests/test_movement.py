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


def test_shift_teams_keeps_matrix_consistent():
    import numpy as np
    import pandas as pd

    from cs2p.run import shift_teams
    P = pd.DataFrame([[0.5, 0.6, 0.7], [0.4, 0.5, 0.55], [0.3, 0.45, 0.5]], index=list("ABC"), columns=list("ABC"))
    Q = shift_teams(P, ["A"], -50)
    assert np.allclose(Q.values + Q.values.T, 1)
    assert Q.loc["A", "B"] < P.loc["A", "B"] and Q.loc["B", "C"] == P.loc["B", "C"]

import json
from pathlib import Path

from cs2p import teams

FIELDS = {"1": {d for d, _ in teams.STAGE1}}
ALL = {d for d, _ in teams.STAGE1 + teams.STAGE2_INVITES + teams.STAGE3_INVITES}


def test_pundit_cards_are_valid():
    data = json.loads((Path(__file__).parent.parent / "research" / "pundits.json").read_text(encoding="utf-8"))
    for stage, cards in data["stages"].items():
        for card in cards:
            picks = card["picks"]
            assert [len(picks[k]) for k in ("3-0", "3-1/3-2", "0-3")] == [2, 6, 2], card["source"]
            chosen = [t for k in picks for t in picks[k]]
            assert len(set(chosen)) == 10, card["source"]
            if stage in FIELDS:
                assert set(chosen) <= FIELDS[stage], (card["source"], set(chosen) - FIELDS[stage])
            assert set(chosen) <= ALL, (card["source"], set(chosen) - ALL)  # spelling matches the site
            assert card["url"].startswith("https://")

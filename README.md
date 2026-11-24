# CS2 Major Predictor

**[Open the interactive PGL Singapore Major report](https://aamarnath1.github.io/cs2-major-predictor/)**

How would you fill out a Pick'em card if you could play the Major thousands of times? This project
starts with public match results, estimates how likely each team is to win a series, and plays out
the Swiss stages and playoffs. The report lets you compare cards, follow a team's path through
Swiss, and see how the forecast changes as real results arrive.

## From match history to a Pick'em card

```mermaid
flowchart TD
    V["Valve VRS<br/>Match history and ranking snapshots"]
    L["Liquipedia<br/>Extra match history, past Majors and live results"]
    R["Valve and PGL rules<br/>Seeds, pairings and match formats"]
    H["Historical results<br/>Normalize team names and combine coverage"]
    M["Team ratings<br/>Elo · Glicko-2 · TrueSkill · Bradley-Terry"]
    P["Matchup probabilities<br/>Average the four models in log-odds"]
    S["Tournament simulation<br/>Play out Swiss and the playoff bracket"]
    C["Pick'em search<br/>Find cards likely to clear the challenges"]
    E["Evaluation<br/>Walk-forward match tests and historical Pick'em replay"]
    W["Static report<br/>Python exports JSON for GitHub Pages"]

    V --> H
    L --> H
    H --> M --> P --> S --> C --> W
    R --> S
    L -->|"Replay completed results"| S
    P --> E
    L --> E
    E --> W
```

[Valve's VRS repository](https://github.com/ValveSoftware/counter-strike_regional_standings)
supplies the core history and the ranking baseline. [Liquipedia](https://liquipedia.net/counterstrike/PGL/2026/Singapore)
extends coverage outside the VRS history and supplies tournament results. The simulator uses
[Valve's Major rules](https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/main/major-supplemental-rulebook.md)
and [PGL's Singapore format](https://www.pglesports.com/cs2/pgl-major-singapore-2026/).
Once a stage reaches its scheduled opener, the report scores only a card saved beforehand.

### What we learned

The four-rater ensemble beat a tuned VRS baseline on 560 historical matches between top-rated
teams. Elo plus Bradley–Terry came close enough that the extra raters have yet to show a clear
benefit. In the historical Pick'em replay (raw Bo1 odds, as in production), the ensemble averaged
4.36 correct picks across 11 stages, the same as seed picks (5 wins, 2 ties, 4 losses; 95% CI -1.0 to
+1.0 picks per stage). Its cards expected 43.0 correct picks in total and got 48, about one standard
deviation above, so the Pick'em odds are not clearly miscalibrated. Some
stages also predate the tuning cutoff. A map-pool and veto experiment showed no measurable gain in its test, so it
stays outside the live forecast.

## Run it yourself

### View the included snapshot

Clone the repo and serve the checked-in report with Python. Open <http://127.0.0.1:8600>.

```bash
git clone https://github.com/aamarnath1/cs2-major-predictor.git
cd cs2-major-predictor
python -m http.server 8600 --bind 127.0.0.1 --directory web
```

### Rebuild the forecast

From the repo root, create an environment with Python 3.11 or newer.

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate`
on macOS or Linux. Then run these commands in order. The first Liquipedia backfill is rate-limited
and can take a while. It caches pages locally. A rebuild uses the latest available source data.

```bash
python -m pip install numpy pandas scipy scikit-learn trueskill requests pytest
git clone --depth 1 https://github.com/ValveSoftware/counter-strike_regional_standings data/raw/vrs
python -m cs2p.ingest_vrs
python -m cs2p.ingest_liquipedia
python -m cs2p.backtest
python -m cs2p.backtest_bt
python -m cs2p.run
python -m pytest -q
```

The forecast updates `web/data.json` and writes tables and available pre-cutoff cards to
`outputs/`. Rerun `python -m cs2p.run` after a round to pick up Singapore results. To regenerate
the research sections, run `python -m cs2p.zoo` and `python -m cs2p.replay` before the forecast.

The main entry points are [the forecast pipeline](cs2p/run.py), [the Swiss simulator](cs2p/swiss.py),
and [the Bo1 experiment](cs2p/bo1_experiment.py). Run `python -m cs2p.bo1_experiment` after the data
and parameters are built to compare the current Bo1 conversion with raw and calibrated odds.
Use `--card PATH` to reproduce challenger forecasts from an archived Swiss card. The [frozen
Singapore Round 1 forecasts](outputs/bo1_singapore_stage1.json) are ready to score after play.
After refreshing the report, run
`python -m cs2p.bo1_experiment --score-round1 web/data.json`. It reports progress until all eight
opening results are present, then scores each frozen variant with log-loss and Brier score.

The research checks below print JSON without publishing a forecast.

| Check | Command |
| --- | --- |
| Chronological Major rating and Pick’em benchmark | `python -m cs2p.event_benchmark` |
| Broader Bo1 walk-forward comparison | `python -m cs2p.bo1_walkforward` |
| Roster and low-history sensitivity | `python -m cs2p.roster_research --availability research/vrs_snapshot_publication.json` |

The roster check uses the Valve revision pinned in the
[publication manifest](research/vrs_snapshot_publication.json). It tests changes in published
rosters, since the historical data does not reliably record who stood in at a specific match.

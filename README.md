# CS2 Major Predictor — PGL Singapore Major 2026

Statistical Pick'em + champion predictor built on free public data.

## Pipeline
```powershell
uv venv .venv; uv pip install --python .venv/Scripts/python.exe pandas numpy scikit-learn trueskill vaderSentiment pytest
git clone --depth 1 https://github.com/ValveSoftware/counter-strike_regional_standings data/raw/vrs
.venv/Scripts/python.exe -m cs2p.ingest_vrs     # 37k series from Valve VRS detail pages
.venv/Scripts/python.exe -m cs2p.backtest       # tune Elo vs VRS baseline
.venv/Scripts/python.exe -m cs2p.backtest_bt    # Bradley-Terry + ensemble backtest
.venv/Scripts/python.exe -m cs2p.ingest_liquipedia  # Nov results + past Major stages (rate-limited, cached)
.venv/Scripts/python.exe -m cs2p.sentiment      # Reddit crowd sentiment -> outputs/sentiment.json
.venv/Scripts/python.exe -m cs2p.zoo            # 10-model walk-forward comparison + bootstrap CIs
.venv/Scripts/python.exe -m cs2p.replay         # Pick'em replay on past Majors
.venv/Scripts/python.exe -m cs2p.run            # Stage 1 sim, Pick'em picks, champion odds -> web/data.json
.venv/Scripts/python.exe -m http.server 8600 --directory web   # open http://127.0.0.1:8600
.venv/Scripts/python.exe -m pytest tests
```

## Model
- Matchup probability = logit average of Elo, Glicko-2, TrueSkill and time-weighted Bradley-Terry.
- Model zoo (538 top-tier matches since Mar 2026, walk-forward): all raters ~0.645-0.652 log-loss,
  differences within bootstrap noise; all significantly beat Valve VRS (0.674).
- Swiss stage Monte Carlo (Buchholz pairings, Bo1/Bo3 rules) -> pick optimiser maximising P(>=5 correct).

## Data
Valve VRS repo (match history), Liquipedia API (participants, seeds, stand-ins; CC-BY-SA).

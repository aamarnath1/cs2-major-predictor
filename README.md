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
.venv/Scripts/python.exe -m cs2p.run            # live: pulls Singapore results from Liquipedia, re-rates, re-simulates
                                                # every stage + playoffs, writes web/data.json
.venv/Scripts/python.exe -m http.server 8600 --directory web   # open http://127.0.0.1:8600
.venv/Scripts/python.exe -m pytest tests
```

## Model
- Matchup probability = logit average of Elo, Glicko-2, TrueSkill and time-weighted Bradley-Terry.
- Model zoo (560 matches between top-rated teams since March 2026, prior-day ratings): four raters
  scored 0.6437 log-loss, Elo plus Bradley–Terry 0.6446, and the tuned Valve VRS baseline 0.6719.
  The test does not establish an advantage for four raters over two.
- Swiss stage Monte Carlo (Buchholz pairings, Bo1/Bo3 rules) -> pick optimiser maximising P(>=5 correct).

## Data
Valve VRS repo (match history), Liquipedia API (participants, seeds, stand-ins; CC-BY-SA).

## During the Major
Rerun `python -m cs2p.run` after each round. Stage pages are refreshed from Liquipedia (cached 15 min),
finished matches are replayed and the rest of the event is re-simulated. The report scores only a
card saved before PGL's scheduled stage opener. Missing cards stay unscored when the results feed
arrives late. Pre-cutoff Swiss snapshots include the matchup matrix, seed and Elo baseline cards,
and provenance for the data and model.
Stage 2/3 seeding follows the official rule (invites 1-8 by VRS, qualifiers 9-16 by record + Buchholz).

## Reddit sentiment (optional)
Create a "script" app at https://www.reddit.com/prefs/apps, then put the credentials in a `.env` file in
the project root (gitignored):
```
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
```
and run `python -m cs2p.sentiment`.

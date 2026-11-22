"""Model zoo: score many predictors on identical walk-forward splits.

Every model predicts only matches in month M using information available before month M
(online raters use strictly pre-match state; batch/learned models refit on the 1st of M).
Eval set = top-tier matches (both teams >= 1400 VRS points) since EVAL_FROM.
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cs2p.backtest_bt import regions
from cs2p.bt import fit_bt
from cs2p.elo import PROC, EloParams, load_matches, run
from cs2p.features import build_features

EVAL_FROM = pd.Timestamp("2026-03-01")
TRAIN_FROM = pd.Timestamp("2025-01-01")
OUT = PROC.parent.parent / "outputs"


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    """p = prob of side a winning, y = 1 if a won."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    return {"logloss": ll, "brier": float(np.mean((p - y) ** 2)), "acc": float(np.mean((p > .5) == y)), "ece": float(ece)}


def reliability(p: np.ndarray, y: np.ndarray) -> list[dict]:
    bins = np.clip((p * 10).astype(int), 0, 9)
    return [{"pred": float(p[bins == b].mean()), "obs": float(y[bins == b].mean()), "n": int((bins == b).sum())}
            for b in range(10) if (bins == b).sum() >= 5]


def add_bt(F: pd.DataFrame, m: pd.DataFrame, hl: float, prior: float) -> None:
    F["bt_a"], F["bt_b"] = np.nan, np.nan
    months = pd.date_range(F.date.min().to_period("M").to_timestamp(), F.date.max() + pd.offsets.MonthBegin(1), freq="MS")
    for start, end in zip(months[:-1], months[1:]):
        idx = (F.date >= start) & (F.date < end)
        if idx.any():
            s = fit_bt(m, start, hl, prior)
            F.loc[idx, "bt_a"] = F.loc[idx, "a"].map(s).fillna(0.0)
            F.loc[idx, "bt_b"] = F.loc[idx, "b"].map(s).fillna(0.0)


def design(F: pd.DataFrame) -> pd.DataFrame:
    """Antisymmetric feature diffs (a minus b) plus symmetric context."""
    X = pd.DataFrame(index=F.index)
    X["elo"] = F.elo_a - F.elo_b
    X["glicko"] = F.glicko_a - F.glicko_b
    X["glicko_rd"] = F.glicko_rd_a - F.glicko_rd_b
    X["ts_mu"] = F.ts_mu_a - F.ts_mu_b
    X["ts_sig"] = F.ts_sig_a - F.ts_sig_b
    X["bt"] = F.bt_a - F.bt_b
    X["vrs"] = (F.vrs_a.fillna(1000) - F.vrs_b.fillna(1000)) / 100
    X["form10"] = F.form10_a - F.form10_b
    X["form60"] = F.form60_a - F.form60_b
    X["sos10"] = (F.sos10_a - F.sos10_b) / 100
    X["games"] = np.log1p(F.games_a) - np.log1p(F.games_b)
    X["idle"] = np.log1p(F.idle_a) - np.log1p(F.idle_b)
    X["h2h"] = (2 * F.h2h_a - F.h2h_n) / (F.h2h_n + 2)
    return X


def flip(X: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Randomly swap sides on half the rows so labels are balanced. All features are antisymmetric."""
    s = rng.random(len(X)) < 0.5
    Xf = X.copy()
    Xf.loc[s] = -Xf.loc[s]
    return Xf, (~s).astype(int)


def history_test(m: pd.DataFrame) -> list[dict]:
    """Does older data help? Elo trained from different start dates, scored on the same eval set."""
    from cs2p.backtest import log_loss, vrs_baseline

    wp, lp = vrs_baseline(m)
    ev = (m.date >= EVAL_FROM).values & (wp >= 1400) & (lp >= 1400)
    params = EloParams(**json.loads((PROC / "elo_params.json").read_text()))
    out = []
    for cut in sorted({str(m.date.min().date()), "2024-08-01", "2025-02-01", "2025-08-01"}):
        keep = (m.date >= cut).values
        _, p = run(m[keep].reset_index(drop=True), params)
        out.append({"from": cut, "n_train": int(keep.sum()), "logloss": log_loss(p[ev[keep]])})
    return out


def main() -> None:
    m = load_matches()
    print("building features ...")
    F, _ = build_features(m, start=TRAIN_FROM)
    btp = json.loads((PROC / "bt_params.json").read_text())
    add_bt(F, m, btp["half_life_days"], btp["prior"])
    F.to_csv(PROC / "features.csv", index=False)

    reg = regions()
    top = (F.vrs_a >= 1400) & (F.vrs_b >= 1400)
    ev = (top & (F.date >= EVAL_FROM)).values
    ra, rb = F.a.map(reg), F.b.map(reg)
    cross = ev & ra.notna().values & rb.notna().values & (ra != rb).values

    rng = np.random.default_rng(0)
    X = design(F)
    Xf, y = flip(X, rng)
    s = y == 0  # rows where sides were swapped: probs must be flipped too

    def oriented(p_winner: np.ndarray) -> np.ndarray:
        return np.where(s, 1 - p_winner, p_winner)

    preds: dict[str, np.ndarray] = {}
    # --- single online raters (prob that original winner wins)
    preds["Elo"] = oriented(F.elo_p.values)
    preds["Glicko-2"] = oriented(F.glicko_p.values)
    preds["TrueSkill"] = oriented(F.ts_p.values)
    preds["Bradley-Terry"] = oriented(sig(F.bt_a.values - F.bt_b.values))
    # VRS baseline with scale fit before eval window.
    vd = (F.vrs_a - F.vrs_b).values
    pre = top.values & (F.date < EVAL_FROM).values & ~np.isnan(vd)
    scales = np.arange(200, 2001, 50)
    sc = scales[np.argmin([metrics(sig(vd[pre] * np.log(10) / s_), np.ones(pre.sum()))["logloss"] for s_ in scales])]
    preds["VRS points"] = oriented(sig(np.nan_to_num(vd) * np.log(10) / sc))

    # --- ensembles of raters (logit averages)
    preds["Ensemble Elo+BT"] = sig((logit(preds["Elo"]) + logit(preds["Bradley-Terry"])) / 2)
    preds["Ensemble 4 raters"] = sig(np.mean([logit(preds[k]) for k in ("Elo", "Glicko-2", "TrueSkill", "Bradley-Terry")], 0))

    # --- learned models, walk-forward monthly refit
    learned = {
        "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, fit_intercept=False)),
        "Gradient boosting": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                                     min_samples_leaf=80, l2_regularization=1.0),
    }
    min_games = (F.games_a >= 5).values & (F.games_b >= 5).values
    months = pd.date_range(EVAL_FROM, F.date.max() + pd.offsets.MonthBegin(1), freq="MS")
    importances = {}
    for name, make in learned.items():
        p = np.full(len(F), np.nan)
        for start, end in zip(months[:-1], months[1:]):
            tr = (F.date < start).values & min_games
            te = (F.date >= start).values & (F.date < end).values
            if not te.any():
                continue
            model = make().fit(Xf.values[tr], y[tr])
            p[te] = model.predict_proba(Xf.values[te])[:, 1]
            if name == "Logistic regression":
                importances[name] = dict(zip(X.columns, model[-1].coef_[0].round(3).tolist()))
        preds[name] = p
    preds["Stack: LR + 4 raters"] = sig((logit(preds["Logistic regression"]) + logit(preds["Ensemble 4 raters"])) / 2)

    # Paired bootstrap of per-match log-loss vs the current production model.
    def per_match_ll(p: np.ndarray) -> np.ndarray:
        q = np.clip(p[ev], 1e-6, 1 - 1e-6)
        return -(y[ev] * np.log(q) + (1 - y[ev]) * np.log(1 - q))

    base = per_match_ll(preds["Ensemble Elo+BT"])
    boot = np.random.default_rng(1).integers(0, ev.sum(), (2000, ev.sum()))
    results = []
    for name, p in preds.items():
        d = per_match_ll(p) - base
        bs = d[boot].mean(1)
        r = {"model": name, **metrics(p[ev], y[ev]), "cross_logloss": metrics(p[cross], y[cross])["logloss"],
             "delta_vs_prod": float(d.mean()), "delta_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
             "reliability": reliability(p[ev], y[ev])}
        results.append(r)
    results.sort(key=lambda r: r["logloss"])
    print(f"eval n={ev.sum()} cross n={cross.sum()}")
    print(pd.DataFrame(results).drop(columns="reliability").round(4).to_string(index=False))
    print("LR coefficients (standardised):", importances.get("Logistic regression"))

    out = {"eval_n": int(ev.sum()), "cross_n": int(cross.sum()), "eval_from": str(EVAL_FROM.date()),
           "results": results, "lr_coef": importances.get("Logistic regression"), "history": history_test(m)}
    (PROC / "zoo.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

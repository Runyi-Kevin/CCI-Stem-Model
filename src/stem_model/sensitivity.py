"""Sensitivity analysis of weights and robustness of the composite model."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .scoring import assign_band, composite, weight_vector


def _top_overlap(a: np.ndarray, b: np.ndarray, frac: float = C.TOP_FRACTION) -> float:
    k = int(round(len(a) * frac))
    return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k


def compare(base: pd.Series, alt: pd.Series) -> dict:
    rb, ra = base.rank(ascending=False, method="min"), alt.rank(ascending=False, method="min")
    shift = (rb - ra).abs()
    return {"Spearman rho": stats.spearmanr(base, alt).statistic,
            "Kendall tau": stats.kendalltau(base, alt).statistic,
            "Top-10% overlap": _top_overlap(base.to_numpy(), alt.to_numpy()),
            "Mean |rank shift|": shift.mean(), "Max |rank shift|": shift.max(),
            "Band changed %": 100 * (np.asarray(assign_band(base)) != np.asarray(assign_band(alt))).mean()}


def scenario_analysis(scored: pd.DataFrame) -> pd.DataFrame:
    base = scored["composite"]
    rows = []
    for name, w in C.SCENARIOS.items():
        rows.append({"Scenario": name, **compare(base, composite(scored, w))})
    return pd.DataFrame(rows)


def oat_analysis(scored: pd.DataFrame) -> pd.DataFrame:
    """One-at-a-time: shift one weight by delta, rescale the others proportionally."""
    base_w = C.BASELINE_WEIGHTS; base = scored["composite"]; rows = []
    for k in C.INDICATORS:
        for d in C.OAT_DELTAS:
            new = base_w[k] + d
            if new < 0 or new > 1:
                continue
            others = 1 - base_w[k]
            w = {j: (new if j == k else base_w[j] * (1 - new) / others) for j in C.INDICATORS}
            rows.append({"Indicator": C.LABELS[k], "Weight change (pp)": d * 100,
                         "New weight %": new * 100, **compare(base, composite(scored, w))})
    return pd.DataFrame(rows)


def monte_carlo(scored: pd.DataFrame, seed: int = C.SEED) -> dict:
    """Dirichlet-perturbed weights centred on the baseline (random weight uncertainty)."""
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(weight_vector(C.BASELINE_WEIGHTS) * C.MC_CONCENTRATION, size=C.MC_DRAWS)
    X = scored[C.INDICATORS].to_numpy()
    S = W @ X.T                                            # draws x participants
    ranks = (-S).argsort(axis=1).argsort(axis=1) + 1
    base_rank = scored["rank"].to_numpy()
    rho = np.array([stats.spearmanr(scored["composite"], s).statistic for s in S])
    base_band = np.asarray(assign_band(scored["composite"]))
    stay = np.zeros(len(scored))
    for s in S:
        stay += (np.asarray(assign_band(pd.Series(s))) == base_band)
    stay /= C.MC_DRAWS
    lo, hi = np.percentile(ranks, [5, 95], axis=0)
    return {"rho": rho, "weights": W, "rank_lo": lo, "rank_hi": hi, "base_rank": base_rank,
            "band_stability": stay,
            "summary": pd.DataFrame({
                "Metric": ["Draws", "Mean Spearman rho", "5th percentile rho", "Min rho",
                           "Median 90% rank-interval width", "Mean band stability %",
                           "Participants with >=90% band stability %"],
                "Value": [C.MC_DRAWS, rho.mean(), np.percentile(rho, 5), rho.min(),
                          float(np.median(hi - lo)), 100 * stay.mean(), 100 * (stay >= 0.9).mean()]})}


def noise_robustness(scored: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """Add Gaussian measurement error to indicator scores and re-rank."""
    rng = np.random.default_rng(seed); w = weight_vector(C.BASELINE_WEIGHTS)
    X = scored[C.INDICATORS].to_numpy(); base = scored["composite"].to_numpy()
    base_band = np.asarray(assign_band(pd.Series(base))); rows = []
    for sd in C.NOISE_SDS:
        rho, stay, top = [], [], []
        for _ in range(C.NOISE_REPS):
            Xn = np.clip(X + rng.normal(0, sd, X.shape), 0, 100); s = Xn @ w
            rho.append(stats.spearmanr(base, s).statistic)
            stay.append((np.asarray(assign_band(pd.Series(s))) == base_band).mean())
            top.append(_top_overlap(base, s))
        rows.append({"Noise SD (points)": sd, "Mean Spearman rho": np.mean(rho),
                     "5th pct rho": np.percentile(rho, 5), "Mean band agreement %": 100 * np.mean(stay),
                     "Mean top-10% overlap": np.mean(top)})
    return pd.DataFrame(rows)


def boundary_fragility(scored: pd.DataFrame) -> pd.DataFrame:
    cuts = [lo for lo, _ in C.BANDS if np.isfinite(lo)]
    d = scored["composite"].apply(lambda s: min(abs(s - c) for c in cuts))
    return pd.DataFrame({"Within (points) of a band boundary": [1, 2, 3],
                         "Participants": [int((d <= t).sum()) for t in (1, 2, 3)],
                         "Percent": [100 * float((d <= t).mean()) for t in (1, 2, 3)]})


def reproducibility_check(run_fn, repeats: int = 3) -> dict:
    """Execute the scoring pipeline repeatedly and compare SHA-256 hashes of outputs."""
    hashes = []
    for _ in range(repeats):
        out = run_fn()
        hashes.append(hashlib.sha256(pd.util.hash_pandas_object(out, index=True).values.tobytes()).hexdigest())
    return {"repeats": repeats, "identical": len(set(hashes)) == 1, "hash": hashes[0][:16]}

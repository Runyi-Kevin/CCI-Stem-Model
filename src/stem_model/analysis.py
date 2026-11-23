"""Descriptive statistics, correlation, PCA and K-means clustering."""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

from . import config as C

log = logging.getLogger(__name__)
COLS = C.INDICATORS + ["composite"]


# ---------------------------------------------------------------- descriptives
def descriptives(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in COLS:
        x = scored[c].to_numpy(float)
        n = len(x); m = x.mean(); sd = x.std(ddof=1)
        ci = stats.t.interval(0.95, n - 1, loc=m, scale=sd / np.sqrt(n))
        rows.append({"Variable": C.LABELS.get(c, "Composite score"), "n": n, "Mean": m, "SD": sd,
                     "Median": np.median(x), "Min": x.min(), "Max": x.max(),
                     "IQR": np.subtract(*np.percentile(x, [75, 25])),
                     "Skewness": stats.skew(x), "Excess kurtosis": stats.kurtosis(x),
                     "95% CI low": ci[0], "95% CI high": ci[1],
                     "Shapiro-Wilk p": stats.shapiro(x).pvalue})
    return pd.DataFrame(rows)


def band_frequencies(scored: pd.DataFrame) -> pd.DataFrame:
    f = scored["band"].value_counts().reindex(C.BAND_ORDER[::-1]).fillna(0).astype(int)
    out = f.rename_axis("Band").reset_index(name="Count")
    out["Percent"] = 100 * out.Count / out.Count.sum()
    out["Cumulative %"] = out.Percent.cumsum()
    return out


# ----------------------------------------------------------------- correlation
def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(1.0, run)
    return adj


def pearson_analysis(scored: pd.DataFrame):
    X = scored[C.INDICATORS]
    r_mat = X.corr(method="pearson")
    n = len(X); rows = []
    for a, b in itertools.combinations(C.INDICATORS, 2):
        r, p = stats.pearsonr(X[a], X[b])
        z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        rows.append({"Indicator A": C.LABELS[a], "Indicator B": C.LABELS[b], "r": r,
                     "95% CI low": lo, "95% CI high": hi, "p": p})
    pairs = pd.DataFrame(rows)
    pairs["p (Holm)"] = holm(pairs.p.to_numpy())
    pairs["Significant (Holm, 0.05)"] = np.where(pairs["p (Holm)"] < 0.05, "Yes", "No")
    return r_mat, pairs


# ------------------------------------------------------------------------- PCA
def bartlett_kmo(X: pd.DataFrame) -> dict:
    """Bartlett's test of sphericity and the Kaiser-Meyer-Olkin measure (is PCA appropriate?)."""
    R = np.corrcoef(X.to_numpy().T); n, p = X.shape
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R)); df = p * (p - 1) / 2
    inv = np.linalg.inv(R); d = np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    partial = -inv / d
    off = ~np.eye(p, dtype=bool)
    kmo = (R[off] ** 2).sum() / ((R[off] ** 2).sum() + (partial[off] ** 2).sum())
    return {"Bartlett chi-square": chi2, "Bartlett df": df,
            "Bartlett p": stats.chi2.sf(chi2, df), "KMO (overall)": kmo}


def parallel_analysis(n: int, p: int, seed: int, sims: int = C.PARALLEL_SIMS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eig = np.empty((sims, p))
    for s in range(sims):
        eig[s] = np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)).T)))[::-1]
    return np.percentile(eig, 95, axis=0)


def pca_analysis(scored: pd.DataFrame, seed: int = C.SEED) -> dict:
    X = scored[C.INDICATORS]
    Z = StandardScaler().fit_transform(X)
    pca = PCA(random_state=seed).fit(Z)
    comps = pca.components_.copy()
    for i in range(comps.shape[0]):               # deterministic sign convention
        if comps[i].sum() < 0:
            comps[i] *= -1
    scores = Z @ comps.T
    names = [f"PC{i+1}" for i in range(len(C.INDICATORS))]
    eig = pca.explained_variance_
    loadings = pd.DataFrame(comps.T * np.sqrt(eig), index=[C.LABELS[c] for c in C.INDICATORS], columns=names)
    pa = parallel_analysis(*Z.shape, seed)
    var = pd.DataFrame({"Component": names, "Eigenvalue": eig,
                        "Variance %": pca.explained_variance_ratio_ * 100,
                        "Cumulative %": np.cumsum(pca.explained_variance_ratio_) * 100,
                        "Parallel-analysis 95th pct": pa,
                        "Retain (Kaiser >1)": np.where(eig > 1, "Yes", "No"),
                        "Retain (parallel)": np.where(eig > pa, "Yes", "No")})
    return {"variance": var, "loadings": loadings, "scores": pd.DataFrame(scores, columns=names),
            "suitability": bartlett_kmo(X), "Z": Z}


# --------------------------------------------------------------------- K-means
def evaluate_k(Z: np.ndarray, seed: int = C.SEED) -> pd.DataFrame:
    rows = []
    for k in C.K_RANGE:
        km = KMeans(n_clusters=k, n_init=C.KMEANS_N_INIT, random_state=seed).fit(Z)
        rows.append({"k": k, "Inertia (WCSS)": km.inertia_,
                     "Silhouette": silhouette_score(Z, km.labels_),
                     "Calinski-Harabasz": calinski_harabasz_score(Z, km.labels_),
                     "Davies-Bouldin": davies_bouldin_score(Z, km.labels_)})
    return pd.DataFrame(rows)


def choose_k(ev: pd.DataFrame) -> int:
    """Highest mean silhouette over the candidate range."""
    return int(ev.loc[ev["Silhouette"].idxmax(), "k"])


def fit_clusters(scored: pd.DataFrame, Z: np.ndarray, k: int, seed: int = C.SEED) -> dict:
    km = KMeans(n_clusters=k, n_init=C.KMEANS_N_INIT, random_state=seed).fit(Z)
    # relabel clusters by ascending mean composite so labels are stable and meaningful
    order = scored.groupby(km.labels_)["composite"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order, start=1)}
    labels = np.array([remap[l] for l in km.labels_])
    prof = scored.assign(cluster=labels).groupby("cluster")[C.INDICATORS + ["composite"]].mean()
    prof.insert(0, "n", pd.Series(labels).value_counts().sort_index())
    prof.insert(1, "share %", prof.n / prof.n.sum() * 100)
    z = pd.DataFrame(Z, columns=C.INDICATORS).assign(cluster=labels).groupby("cluster").mean()
    # bootstrap stability (ARI of refits on resamples vs reference partition)
    rng = np.random.default_rng(seed); aris = []
    for _ in range(C.BOOTSTRAPS):
        idx = rng.integers(0, len(Z), len(Z))
        kb = KMeans(n_clusters=k, n_init=5, random_state=int(rng.integers(1e9))).fit(Z[idx])
        aris.append(adjusted_rand_score(km.labels_, kb.predict(Z)))
    return {"labels": labels, "profile": prof, "z_profile": z, "k": k,
            "silhouette": silhouette_score(Z, labels), "ari": np.array(aris)}

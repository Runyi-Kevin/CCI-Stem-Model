"""Figures (all stamped as synthetic/illustrative)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from . import config as C

sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans")
PAL = sns.color_palette("viridis", 5)
BAND_COL = dict(zip(C.BAND_ORDER, ["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#1f618d"]))


def _save(fig, out: Path, name: str) -> Path:
    fig.text(0.5, 0.005, C.FOOTER, ha="center", va="bottom", fontsize=7, style="italic", color="gray")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    path = out / f"{name}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def distribution(scored, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.histplot(scored.composite, bins=25, kde=True, color=PAL[1], ax=ax[0])
    for lo, lab in C.BANDS[:-1]:
        ax[0].axvline(lo, ls="--", c="gray", lw=0.8)
    ax[0].set(title="Composite score distribution", xlabel="Composite score (0-100)", ylabel="Participants")
    stats.probplot(scored.composite, dist="norm", plot=ax[1])
    ax[1].set_title("Normal Q-Q plot of composite score")
    return _save(fig, out, "fig01_composite_distribution")


def indicator_boxplots(scored, out):
    long = scored.melt(value_vars=C.INDICATORS, var_name="Indicator", value_name="Score")
    long["Indicator"] = long.Indicator.map(C.LABELS)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(data=long, x="Indicator", y="Score", palette="viridis", hue="Indicator", legend=False, ax=ax)
    ax.set(title="Distribution of the five indicator scores", xlabel="")
    plt.setp(ax.get_xticklabels(), rotation=15)
    return _save(fig, out, "fig02_indicator_boxplots")


def band_bars(freq, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(freq.Band, freq.Count, color=[BAND_COL[b] for b in freq.Band])
    for i, (c, p) in enumerate(zip(freq.Count, freq.Percent)):
        ax.text(i, c + 2, f"{c} ({p:.1f}%)", ha="center", fontsize=9)
    ax.set(title="Performance band classification", ylabel="Participants", ylim=(0, freq.Count.max() * 1.15))
    return _save(fig, out, "fig03_performance_bands")


def correlation_heatmap(r_mat, out):
    m = r_mat.rename(index=C.SHORT, columns=C.SHORT)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(m, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, mask=np.triu(np.ones_like(m, bool), 1),
                square=True, cbar_kws={"label": "Pearson r"}, ax=ax)
    ax.set_title("Pearson correlations between indicators")
    return _save(fig, out, "fig04_correlation_heatmap")


def scree(pca, out):
    v = pca["variance"]; x = np.arange(1, len(v) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, v.Eigenvalue, "o-", label="Observed eigenvalue")
    ax.plot(x, v["Parallel-analysis 95th pct"], "s--", c="gray", label="Parallel analysis (95th pct)")
    ax.axhline(1, c="red", lw=0.8, ls=":", label="Kaiser criterion (1)")
    ax.set(title="Scree plot with parallel analysis", xlabel="Component", ylabel="Eigenvalue", xticks=x)
    ax.legend()
    return _save(fig, out, "fig05_pca_scree")


def biplot(pca, labels, out):
    sc, ld = pca["scores"], pca["loadings"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.scatterplot(x=sc.PC1, y=sc.PC2, hue=pd.Series(labels, name="Cluster"), palette="viridis", s=22, alpha=.7, ax=ax)
    k = 3.0
    for name, row in ld.iterrows():
        ax.arrow(0, 0, row.PC1 * k, row.PC2 * k, color="crimson", head_width=0.08)
        ax.text(row.PC1 * k * 1.1, row.PC2 * k * 1.1, name, color="crimson", fontsize=8)
    v = pca["variance"]
    ax.set(title="PCA biplot (standardised indicators)", xlabel=f"PC1 ({v['Variance %'][0]:.1f}%)",
           ylabel=f"PC2 ({v['Variance %'][1]:.1f}%)")
    return _save(fig, out, "fig06_pca_biplot")


def cluster_selection(ev, k, out):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for a, col, t in zip(ax, ["Inertia (WCSS)", "Silhouette", "Davies-Bouldin"],
                         ["Elbow (WCSS)", "Mean silhouette (higher better)", "Davies-Bouldin (lower better)"]):
        a.plot(ev.k, ev[col], "o-"); a.axvline(k, c="red", ls="--", lw=.8); a.set(title=t, xlabel="k")
    return _save(fig, out, "fig07_cluster_selection")


def cluster_profiles(cl, out):
    z = cl["z_profile"].rename(columns=C.SHORT)
    z.index = [f"Cluster {i} (n={int(n)})" for i, n in zip(z.index, cl["profile"].n)]
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.9 * len(z)))
    sns.heatmap(z, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, cbar_kws={"label": "Mean z-score"}, ax=ax)
    ax.set_title(f"Cluster profiles (k={cl['k']}, standardised means)")
    return _save(fig, out, "fig08_cluster_profiles")


def sensitivity_scenarios(sc, out):
    d = sc.iloc[1:].sort_values("Spearman rho")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    ax[0].barh(d.Scenario, d["Spearman rho"], color=PAL[2]); ax[0].set_xlim(d["Spearman rho"].min() - .01, 1)
    ax[0].set(title="Spearman rho vs baseline ranking", xlabel="rho")
    ax[1].barh(d.Scenario, d["Top-10% overlap"] * 100, color=PAL[3])
    ax[1].set(title="Top-10% membership retained", xlabel="% of baseline top 10%")
    return _save(fig, out, "fig09_sensitivity_scenarios")


def oat_lines(oat, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, g in oat.groupby("Indicator", sort=False):
        ax.plot(g["Weight change (pp)"], g["Spearman rho"], "o-", label=name)
    ax.set(title="One-at-a-time weight perturbation", xlabel="Change in weight (percentage points)",
           ylabel="Spearman rho vs baseline")
    ax.legend(fontsize=8)
    return _save(fig, out, "fig10_oat_sensitivity")


def monte_carlo(mc, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    sns.histplot(mc["rho"], bins=30, color=PAL[1], ax=ax[0])
    ax[0].set(title=f"Spearman rho across {len(mc['rho'])} random weight sets", xlabel="rho vs baseline")
    o = np.argsort(mc["base_rank"])
    x = np.arange(1, len(o) + 1)
    ax[1].fill_between(x, mc["rank_lo"][o], mc["rank_hi"][o], color=PAL[2], alpha=.5, label="5th-95th percentile rank")
    ax[1].plot(x, mc["base_rank"][o], c="k", lw=.8, label="Baseline rank")
    ax[1].set(title="Rank uncertainty under weight perturbation", xlabel="Baseline rank", ylabel="Rank")
    ax[1].legend()
    return _save(fig, out, "fig11_monte_carlo")


def noise_curve(noise, out):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(noise["Noise SD (points)"], noise["Mean Spearman rho"], "o-", label="Mean rho")
    ax.fill_between(noise["Noise SD (points)"], noise["5th pct rho"], noise["Mean Spearman rho"], alpha=.25)
    ax.set(title="Ranking stability under measurement noise", xlabel="Noise SD added to each indicator (points)",
           ylabel="Spearman rho vs noise-free ranking")
    ax2 = ax.twinx(); ax2.plot(noise["Noise SD (points)"], noise["Mean band agreement %"], "s--", c="darkorange")
    ax2.set_ylabel("Band agreement (%)", color="darkorange"); ax2.grid(False)
    return _save(fig, out, "fig12_noise_robustness")

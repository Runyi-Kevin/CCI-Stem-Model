"""Indicator scoring, weighted composite index, ranking and classification."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def validate_weights(weights: dict) -> dict:
    if set(weights) != set(C.INDICATORS):
        raise ValueError(f"Weights must cover exactly: {C.INDICATORS}")
    if any(v < 0 for v in weights.values()):
        raise ValueError("Weights must be non-negative")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"Weights must sum to 1 (got {sum(weights.values()):.6f})")
    return weights


def weight_vector(weights: dict) -> np.ndarray:
    return np.array([validate_weights(weights)[k] for k in C.INDICATORS])


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw inputs into the five 0-100 indicator scores.

    Attendance & Punctuality = 100*(0.6*attended/10 + 0.4*on_time/attended)
    Knowledge Gain (normalised gain) = max(0, (post-pre)/(100-pre))*100
    The other three indicators are used as recorded.
    """
    out = pd.DataFrame({"participant_id": df.participant_id, "team_id": df.team_id})
    att_rate = df.sessions_attended / C.TOTAL_SESSIONS
    punct = np.where(df.sessions_attended > 0, df.sessions_on_time / df.sessions_attended.replace(0, np.nan), 0.0)
    punct = pd.Series(punct, index=df.index).fillna(0.0)
    out["attendance_punctuality"] = 100 * (C.ATTENDANCE_SUBWEIGHT * att_rate + (1 - C.ATTENDANCE_SUBWEIGHT) * punct)
    room = (100 - df.pre_test).replace(0, np.nan)
    out["knowledge_gain"] = ((df.post_test - df.pre_test) / room * 100).clip(lower=0).fillna(0.0)
    for c in ("participation", "teamwork", "group_discussion"):
        out[c] = df[c].astype(float)
    out["gain_points"] = df.post_test - df.pre_test
    return out


def composite(indicators: pd.DataFrame, weights: dict = C.BASELINE_WEIGHTS) -> pd.Series:
    """S_i = sum_k w_k * x_ik  (weights sum to 1; scores on 0-100).

    Rounded to 8 decimals so mathematically identical scores are exact ties in every
    engine (avoids floating-point noise changing ranks or band boundaries).
    """
    raw = indicators[C.INDICATORS].to_numpy() @ weight_vector(weights)
    return pd.Series(np.round(raw, C.SCORE_DECIMALS), index=indicators.index, name="composite")


def assign_band(scores: pd.Series) -> pd.Series:
    conds = [scores >= lo for lo, _ in C.BANDS]
    labels = [lab for _, lab in C.BANDS]
    band = pd.Series(np.select(conds, labels, default=labels[-1]), index=scores.index, name="band")
    return pd.Categorical(band, categories=C.BAND_ORDER, ordered=True)


def rank_scores(scores: pd.Series) -> pd.Series:
    """Competition ranking (1 = best; ties share the lowest rank)."""
    return scores.rank(method="min", ascending=False).astype(int)


def build_scored_table(df: pd.DataFrame, weights: dict = C.BASELINE_WEIGHTS) -> pd.DataFrame:
    ind = compute_indicators(df)
    ind["composite"] = composite(ind, weights)
    ind["rank"] = rank_scores(ind["composite"])
    ind["percentile"] = ind["composite"].rank(pct=True) * 100
    ind["band"] = assign_band(ind["composite"])
    return ind.sort_values("rank", kind="stable").reset_index(drop=True)

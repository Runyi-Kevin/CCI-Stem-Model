"""Data loading and quality assurance."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when the input data fail a blocking quality check."""


@dataclass
class Check:
    name: str
    status: str   # PASS | WARN | FAIL
    detail: str


def load_raw(path) -> pd.DataFrame:
    """Read the participant sheet and return only the raw input columns."""
    try:
        df = pd.read_excel(path, sheet_name=C.SHEET_NAME, header=C.HEADER_ROW, engine="openpyxl")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input workbook not found: {path}") from exc
    except ValueError as exc:
        raise DataValidationError(f"Cannot read sheet '{C.SHEET_NAME}': {exc}") from exc
    missing = [c for c in C.RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    df = df[list(C.RAW_COLUMNS)].rename(columns=C.RAW_COLUMNS).dropna(how="all").reset_index(drop=True)
    log.info("Loaded %d rows from %s", len(df), path)
    return df


def validate(df: pd.DataFrame, expected_n: int = C.EXPECTED_N, strict: bool = True):
    """Run quality checks. Returns (clean_df, checks_df). Raises on FAIL if strict."""
    checks: list[Check] = []
    add = lambda n, s, d: checks.append(Check(n, s, d))
    df = df.copy()

    add("Row count", "PASS" if len(df) == expected_n else "FAIL",
        f"{len(df)} rows (expected {expected_n})")

    ids = df["participant_id"].astype(str)
    dup = int(ids.duplicated().sum())
    add("Unique participant IDs", "PASS" if dup == 0 else "FAIL", f"{dup} duplicate IDs")
    bad_fmt = int((~ids.str.match(r"^STEM-\d{3}$")).sum())
    add("ID format STEM-nnn", "PASS" if bad_fmt == 0 else "FAIL", f"{bad_fmt} malformed IDs")

    num_cols = ["sessions_attended", "sessions_on_time"] + C.SCORE_COLUMNS
    non_numeric = 0
    for col in num_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        non_numeric += int(coerced.isna().sum() - df[col].isna().sum())
        df[col] = coerced
    add("Numeric data types", "PASS" if non_numeric == 0 else "FAIL",
        f"{non_numeric} non-numeric entries in numeric fields")

    n_missing = int(df[["participant_id", "team_id"] + num_cols].isna().sum().sum())
    add("Missing values", "PASS" if n_missing == 0 else "FAIL", f"{n_missing} missing cells")

    bad_sess = int(((df.sessions_attended < 0) | (df.sessions_attended > C.TOTAL_SESSIONS)
                    | (df.sessions_attended % 1 != 0)).sum())
    add("Sessions attended in 0-10 (integer)", "PASS" if bad_sess == 0 else "FAIL", f"{bad_sess} invalid rows")
    bad_ot = int(((df.sessions_on_time < 0) | (df.sessions_on_time > df.sessions_attended)).sum())
    add("On-time sessions <= attended", "PASS" if bad_ot == 0 else "FAIL", f"{bad_ot} invalid rows")
    bad_rng = int(sum(((df[c] < 0) | (df[c] > 100)).sum() for c in C.SCORE_COLUMNS))
    add("Scores within 0-100", "PASS" if bad_rng == 0 else "FAIL", f"{bad_rng} out-of-range values")

    zero_att = int((df.sessions_attended == 0).sum())
    add("Zero attendance (punctuality undefined)", "PASS" if zero_att == 0 else "WARN",
        f"{zero_att} participants; punctuality set to 0")
    decline = int((df.post_test < df.pre_test).sum())
    add("Post-test below pre-test", "PASS" if decline == 0 else "WARN",
        f"{decline} participants; negative gain floored at 0 in the model")
    ceil = int((df.pre_test >= 100).sum())
    add("Pre-test at ceiling (gain undefined)", "PASS" if ceil == 0 else "WARN",
        f"{ceil} participants; gain score set to 0")

    zero_var = [c for c in num_cols if df[c].nunique(dropna=True) <= 1]
    add("Non-constant indicators", "PASS" if not zero_var else "FAIL",
        "all vary" if not zero_var else f"constant: {zero_var}")

    out_total = 0
    for c in C.SCORE_COLUMNS:
        q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
        out_total += int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
    add("Univariate outliers (1.5xIQR)", "PASS" if out_total == 0 else "WARN",
        f"{out_total} flagged values retained (valid scores on a bounded scale)")

    sizes = df["team_id"].value_counts()
    add("Team structure", "PASS", f"{sizes.size} teams, size {sizes.min()}-{sizes.max()} (median {int(sizes.median())})")

    out = pd.DataFrame([c.__dict__ for c in checks]).rename(
        columns={"name": "Check", "status": "Status", "detail": "Detail"})
    for _, r in out.iterrows():
        log.log(logging.ERROR if r.Status == "FAIL" else logging.INFO if r.Status == "PASS" else logging.WARNING,
                "[%s] %s - %s", r.Status, r.Check, r.Detail)
    if strict and (out.Status == "FAIL").any():
        raise DataValidationError("Blocking validation failures: " + "; ".join(out[out.Status == "FAIL"].Check))
    return df, out


def reconcile_with_excel(scored: pd.DataFrame, path) -> Check:
    """Compare Python composite scores with cached values of the source workbook's own formulas."""
    ref = pd.read_excel(path, sheet_name=C.SHEET_NAME, header=C.HEADER_ROW, engine="openpyxl")
    col = "Overall Performance Score"
    ref = ref.dropna(subset=["Participant ID"]).set_index("Participant ID")[col]
    ref = pd.to_numeric(ref, errors="coerce")
    if ref.isna().all():
        return Check("Cross-check vs Excel formulas", "WARN", "source workbook has no cached formula values")
    diff = (scored.set_index("participant_id")["composite"] - ref).abs().max()
    return Check("Cross-check vs Excel formulas", "PASS" if diff < 1e-6 else "FAIL",
                 f"max |Python - Excel| = {diff:.2e} over {ref.notna().sum()} participants")

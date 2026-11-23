"""Automated Excel workbook (live formulas) and Markdown technical report."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from . import config as C

log = logging.getLogger(__name__)
F = lambda **k: Font(name="Arial", size=k.pop("size", 10), **k)
HDR_FILL = PatternFill("solid", fgColor="1F3864")
WARN_FILL = PatternFill("solid", fgColor="FFF2CC")
BLUE = Font(name="Arial", size=10, color="0000FF")


def _fmt_for(col: str, series: pd.Series) -> str | None:
    if pd.api.types.is_integer_dtype(series):
        return "0"
    if not pd.api.types.is_float_dtype(series):
        return None
    c = col.lower()
    if c.startswith("p ") or c in ("p", "p (holm)") or "shapiro" in c or "bartlett p" in c:
        return "0.0000"
    if "%" in c or "percent" in c or "share" in c:
        return "0.0"
    return "0.000"


def _title(ws, text, note=None):
    ws["A1"] = text; ws["A1"].font = F(size=14, bold=True)
    ws["A2"] = C.DISCLAIMER; ws["A2"].font = F(italic=True, color="C00000")
    if note:
        ws["A3"] = note; ws["A3"].font = F(italic=True, color="595959")


def put_df(ws, df: pd.DataFrame, row: int, heading: str | None = None, index=False, note: str | None = None) -> int:
    if index:
        df = df.reset_index()
    if heading:
        ws.cell(row, 1, heading).font = F(bold=True, size=11); row += 1
    for j, col in enumerate(df.columns, 1):
        c = ws.cell(row, j, str(col)); c.font = F(bold=True, color="FFFFFF"); c.fill = HDR_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for i, rec in enumerate(df.itertuples(index=False), row + 1):
        for j, v in enumerate(rec, 1):
            v = v.item() if hasattr(v, "item") else v
            if isinstance(v, float) and not np.isfinite(v):
                v = None
            c = ws.cell(i, j, v); c.font = F()
            f = _fmt_for(df.columns[j - 1], df.iloc[:, j - 1])
            if f:
                c.number_format = f
    row += len(df) + 1
    if note:
        ws.cell(row, 1, note).font = F(italic=True, color="595959"); row += 1
    return row + 1


def _widths(ws, default=16, first=30):
    for i in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(i)].width = first if i == 1 else default


def write_workbook(path: Path, res: dict) -> Path:
    wb = Workbook()
    n = len(res["scored"])
    # ---------------------------------------------------------------- README
    ws = wb.active; ws.title = "README"
    _title(ws, "STEM Workshop Participant Performance Model - Results Workbook")
    lines = [
        "Purpose: portfolio demonstration of a reproducible Python scoring/analysis pipeline (IT Technical Assistant application).",
        "All participant records are SYNTHETIC. Every figure, rank and finding is illustrative and must not be read as actual workshop outcomes.",
        "",
        "Sheets",
        "  Weights - editable weights, scoring parameters and band thresholds (blue = input). Changing them recalculates the Scores sheet.",
        "  Validation - data quality checks run before any analysis.",
        "  Scores - raw inputs, live Excel formulas for indicators/composite/rank/band, reconciled against the Python results.",
        "  Descriptives, Bands - distribution statistics and classification frequencies.",
        "  Correlation, PCA, Clusters - Pearson analysis, principal components, K-means profiles.",
        "  Sensitivity, Robustness - weight scenarios, Monte Carlo perturbation, noise and boundary analysis.",
        "",
        f"Composite is rounded to {C.SCORE_DECIMALS} decimals in both Excel and Python so floating-point noise cannot break ties or band boundaries.",
        "Colour convention: blue text = input/assumption; black = formula or Python-computed output.",
        "Static analysis sheets hold values computed in Python (seed %d); only the Scores sheet is formula-driven." % res["seed"],
        f"Run timestamp (UTC): {res['manifest']['run_utc']}",
    ]
    for i, t in enumerate(lines, 4):
        ws.cell(i, 1, t).font = F(bold=(t == "Sheets"))
    ws.column_dimensions["A"].width = 140

    # --------------------------------------------------------------- Weights
    ws = wb.create_sheet("Weights"); _title(ws, "Model parameters (editable)")
    for j, h in enumerate(["Indicator", "Definition", "Weight"], 1):
        c = ws.cell(4, j, h); c.font = F(bold=True, color="FFFFFF"); c.fill = HDR_FILL
    defs = {"attendance_punctuality": "100 x (sub-weight x attendance rate + (1-sub-weight) x punctuality rate)",
            "knowledge_gain": "Normalised gain: max(0,(post-pre)/(100-pre)) x 100",
            "participation": "In-session participation score (0-100)",
            "teamwork": "Teamwork & collaboration score (0-100)",
            "group_discussion": "Group discussions score (0-100)"}
    for i, k in enumerate(C.INDICATORS, 5):
        ws.cell(i, 1, C.LABELS[k]).font = F(); ws.cell(i, 2, defs[k]).font = F()
        c = ws.cell(i, 3, C.BASELINE_WEIGHTS[k]); c.font = BLUE; c.number_format = "0%"
    ws["A10"] = "Total"; ws["A10"].font = F(bold=True)
    ws["C10"] = "=SUM(C5:C9)"; ws["C10"].number_format = "0%"; ws["C10"].font = F(bold=True)
    ws["D10"] = '=IF(ABS(C10-1)<0.000000001,"OK - weights sum to 100%","ERROR - weights must sum to 100%")'
    ws["D10"].font = F(bold=True)
    params = [(13, "Total sessions", C.TOTAL_SESSIONS, "0", "Assumption: workshop ran 10 sessions (from source data header)"),
              (14, "Attendance sub-weight", C.ATTENDANCE_SUBWEIGHT, "0%", "Assumption taken from the source workbook formula (60% attendance)"),
              (15, "Punctuality sub-weight", "=1-C14", "0%", "Formula: remainder (40% punctuality)")]
    for r, lab, v, nf, note in params:
        ws.cell(r, 1, lab).font = F(); c = ws.cell(r, 3, v); c.number_format = nf
        c.font = F() if isinstance(v, str) else BLUE; ws.cell(r, 4, note).font = F(italic=True, color="595959")
    ws["A12"] = "Scoring parameters"; ws["A12"].font = F(bold=True)
    ws["A17"] = "Band thresholds (composite >= threshold)"; ws["A17"].font = F(bold=True)
    for r, (lo, lab) in zip(range(18, 22), C.BANDS[:-1]):
        ws.cell(r, 1, lab).font = F(); c = ws.cell(r, 3, lo); c.font = BLUE
    ws["A22"] = "Needs Improvement"; ws["B22"] = "below the Fair threshold"
    ws.column_dimensions["A"].width = 34; ws.column_dimensions["B"].width = 70
    ws.column_dimensions["C"].width = 12; ws.column_dimensions["D"].width = 62

    # ------------------------------------------------------------ Validation
    ws = wb.create_sheet("Validation"); _title(ws, "Data validation and quality assurance")
    r = put_df(ws, res["checks"], 5)
    for row in ws.iter_rows(min_row=6, max_row=5 + len(res["checks"])):
        if row[1].value == "WARN":
            for c in row: c.fill = WARN_FILL
    _widths(ws, 60, 42); ws.column_dimensions["B"].width = 10

    # ---------------------------------------------------------------- Scores
    ws = wb.create_sheet("Scores"); _title(ws, "Participant scores - live formulas")
    heads = ["Participant ID", "Team ID", "Sessions Attended", "Sessions On Time", "Pre-test", "Post-test",
             "Participation", "Teamwork", "Group Discussions", "Attendance & Punctuality", "Knowledge Gain",
             "Composite Score", "Rank", "Band", "Python Composite", "Reconciliation", "K-means Cluster"]
    for j, h in enumerate(heads, 1):
        c = ws.cell(4, j, h); c.font = F(bold=True, color="FFFFFF"); c.fill = HDR_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[4].height = 32
    raw = res["clean"].set_index("participant_id"); sc = res["scored"].set_index("participant_id")
    first, last = 5, 4 + n
    for i, pid in enumerate(res["clean"].participant_id, first):
        a = raw.loc[pid]
        vals = [pid, a.team_id, int(a.sessions_attended), int(a.sessions_on_time), int(a.pre_test), int(a.post_test),
                int(a.participation), int(a.teamwork), int(a.group_discussion)]
        for j, v in enumerate(vals, 1):
            c = ws.cell(i, j, v); c.font = BLUE if j > 2 else F()
        ws.cell(i, 10, f"=100*(Weights!$C$14*C{i}/Weights!$C$13+Weights!$C$15*IF(C{i}>0,D{i}/C{i},0))")
        ws.cell(i, 11, f"=IF(E{i}>=100,0,MAX(0,(F{i}-E{i})/(100-E{i})*100))")
        ws.cell(i, 12, f"=ROUND(J{i}*Weights!$C$5+K{i}*Weights!$C$6+G{i}*Weights!$C$7+H{i}*Weights!$C$8+I{i}*Weights!$C$9,{C.SCORE_DECIMALS})")
        ws.cell(i, 13, f"=RANK(L{i},$L${first}:$L${last},0)")
        ws.cell(i, 14, f'=IF(L{i}>=Weights!$C$18,"Excellent",IF(L{i}>=Weights!$C$19,"Very Good",'
                       f'IF(L{i}>=Weights!$C$20,"Good",IF(L{i}>=Weights!$C$21,"Fair","Needs Improvement"))))')
        ws.cell(i, 15, float(sc.loc[pid, "composite"]))
        ws.cell(i, 16, f'=IF(ABS(L{i}-O{i})<0.000001,"OK","CHECK")')
        ws.cell(i, 17, int(res["cluster_map"][pid]))
        for j in (10, 11, 12, 15):
            ws.cell(i, j).number_format = "0.00"
        for j in range(10, 18):
            ws.cell(i, j).font = F()
    ws["J2"] = "Reconciliation mismatches (Excel vs Python):"; ws["J2"].font = F(bold=True)
    ws["N2"] = f'=COUNTIF(P{first}:P{last},"CHECK")'; ws["N2"].font = F(bold=True)
    ws["A2"] = C.DISCLAIMER
    ws["O3"] = "Python value is static; a mismatch is expected if Weights are edited."
    ws["O3"].font = F(italic=True, color="595959")
    ws.freeze_panes = "C5"; ws.auto_filter.ref = f"A4:Q{last}"
    for j in range(1, 18):
        ws.column_dimensions[get_column_letter(j)].width = 13
    ws.column_dimensions["N"].width = 20

    # ---------------------------------------------------- analysis sheets
    ws = wb.create_sheet("Descriptives"); _title(ws, "Descriptive statistics")
    put_df(ws, res["desc"], 5, note="Shapiro-Wilk is very sensitive at n=450; read alongside the Q-Q plot.")
    _widths(ws, 13, 26)
    ws = wb.create_sheet("Bands"); _title(ws, "Performance classification")
    r = put_df(ws, res["bands"], 5, note="Thresholds: Excellent >=80, Very Good 70-79.99, Good 60-69.99, Fair 50-59.99, Needs Improvement <50.")
    _widths(ws, 14, 24)

    ws = wb.create_sheet("Correlation"); _title(ws, "Pearson correlation analysis")
    r = put_df(ws, res["r_mat"].rename(index=C.LABELS, columns=C.LABELS), 5, "Correlation matrix", index=True)
    put_df(ws, res["pairs"], r, "Pairwise tests (Holm-adjusted)",
           note="Composite-vs-indicator correlations are excluded: they are part-whole and inflated by construction.")
    _widths(ws, 16, 26)

    ws = wb.create_sheet("PCA"); _title(ws, "Principal Component Analysis (standardised indicators)")
    suit = pd.DataFrame({"Statistic": list(res["pca"]["suitability"]), "Value": [float(v) for v in res["pca"]["suitability"].values()]})
    r = put_df(ws, suit, 5, "Suitability (Bartlett sphericity, KMO)")
    r = put_df(ws, res["pca"]["variance"], r, "Variance explained")
    put_df(ws, res["pca"]["loadings"], r, "Component loadings (correlation of indicator with component)", index=True)
    _widths(ws, 16, 28)

    ws = wb.create_sheet("Clusters"); _title(ws, "K-means cluster analysis")
    r = put_df(ws, res["k_eval"], 5, f"Cluster-number evaluation (selected k = {res['cl']['k']}, highest silhouette)")
    for tag, cl in res["cluster_solutions"].items():
        r = put_df(ws, cl["profile"], r, f"{tag}: profile means (k={cl['k']}, silhouette {cl['silhouette']:.3f}, "
                   f"bootstrap ARI mean {cl['ari'].mean():.2f})", index=True)
    _widths(ws, 16, 26)

    ws = wb.create_sheet("Sensitivity"); _title(ws, "Weight sensitivity analysis")
    r = put_df(ws, res["scen"], 5, "Alternative scoring models vs baseline (Spearman / Kendall)")
    r = put_df(ws, res["oat"], r, "One-at-a-time weight perturbation (other weights rescaled proportionally)")
    r = put_df(ws, res["mc"]["summary"], r, f"Monte Carlo ({C.MC_DRAWS} Dirichlet-perturbed weight sets, concentration {C.MC_CONCENTRATION:g})")
    _widths(ws, 16, 44)

    ws = wb.create_sheet("Robustness"); _title(ws, "Model consistency and robustness")
    rep = res["repro"]
    r = put_df(ws, pd.DataFrame({"Test": ["Reproducibility (identical output hash across repeated runs)",
                                           "Output hash (first 16 hex)", "Excel-formula reconciliation"],
                                 "Result": ["PASS" if rep["identical"] else "FAIL", rep["hash"], res["recon_detail"]]}), 5)
    r = put_df(ws, res["noise"], r, "Measurement-noise robustness (Gaussian noise added to indicator scores)")
    put_df(ws, res["fragile"], r, "Classification fragility near band boundaries")
    _widths(ws, 24, 58)
    wb.save(path); log.info("Workbook written: %s", path)
    return path


# ----------------------------------------------------------------- Markdown
def _cell(v):
    if isinstance(v, (float, np.floating)):
        return "<0.001" if 0 < abs(v) < 0.001 and False else f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def md_table(df: pd.DataFrame, index=False) -> str:
    if index:
        df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "|".join(["---"] * len(df.columns)) + "|\n"
    body = "\n".join("| " + " | ".join(_cell(v) for v in r) + " |" for r in df.itertuples(index=False))
    return head + body + "\n"


def _silhouette_label(s):
    return ("strong" if s >= .71 else "reasonable" if s >= .51 else "weak" if s >= .26 else "no substantial")


def write_report(path: Path, res: dict, fig_names: dict) -> Path:
    d, v, pcs, cl = res["desc"].set_index("Variable"), res["pca"]["variance"], res["pca"], res["cl"]
    comp = d.loc["Composite score"]; band = res["bands"]
    suit = pcs["suitability"]; pairs = res["pairs"]; scen = res["scen"].iloc[1:]
    mc = res["mc"]["summary"].set_index("Metric")["Value"]; noise = res["noise"]; frag = res["fragile"]
    n_k = int((v["Retain (Kaiser >1)"] == "Yes").sum()); n_pa = int((v["Retain (parallel)"] == "Yes").sum())
    fig = lambda key, cap: f"![{cap}](figures/{fig_names[key]})\n\n*{cap}*\n"
    top5 = res["scored"].head(5)[["participant_id", "composite", "rank", "band"]].rename(
        columns={"participant_id": "Participant", "composite": "Composite", "rank": "Rank", "band": "Band"})
    md = f"""# Modelling of STEM Workshop Participant Performance

**Technical report - computational model, statistical analysis and IT competency demonstration**

> **{C.DISCLAIMER}** The workshop has ended and no actual assessment records are available; a synthetic dataset of {len(res['scored'])} participants is used throughout. Every score, rank, cluster and finding below is illustrative of the *method*, not a result about real people.

## 1. Purpose and framework

A reproducible Python model that scores, ranks and classifies participants from five indicators with fixed weights:

{md_table(pd.DataFrame({'Indicator': [C.LABELS[k] for k in C.INDICATORS], 'Weight': [f'{C.BASELINE_WEIGHTS[k]:.0%}' for k in C.INDICATORS]}))}
**Composite index.** For participant *i*, S_i = sum_k w_k x_ik, where x_ik is indicator *k* on a 0-100 scale, w_k >= 0 and sum w_k = 1. Indicator definitions (taken from the supplied workbook's own formulas):

- *Attendance & Punctuality* = 100 x (0.6 x sessions attended / 10 + 0.4 x sessions on time / sessions attended).
- *Knowledge Gain* = normalised gain = max(0, (post - pre) / (100 - pre)) x 100, which adjusts for the room each participant had to improve. Negative gains are floored at 0.
- *Participation, Teamwork, Group Discussions* are used as recorded (0-100).

Bands: Excellent >= 80; Very Good 70-79.99; Good 60-69.99; Fair 50-59.99; Needs Improvement < 50. Ranking uses competition ranking (ties share the best rank).

## 2. Data validation and quality assurance

{md_table(res['checks'])}
The dataset passed every blocking check. Two warnings were handled explicitly rather than silently: {int(res['clean'].eval('post_test < pre_test').sum())} participants scored lower on the post-test than the pre-test (gain floored at zero by the normalised-gain definition), and a small number of valid but extreme values were retained. {res['recon_detail']}.

## 3. Distribution of performance

The composite score has mean {comp['Mean']:.1f} (SD {comp['SD']:.1f}), median {comp['Median']:.1f}, range {comp['Min']:.1f}-{comp['Max']:.1f}, skewness {comp['Skewness']:.2f} and excess kurtosis {comp['Excess kurtosis']:.2f}.

{md_table(res['desc'][['Variable','Mean','SD','Median','Min','Max','Skewness','Excess kurtosis']])}
{fig('dist', 'Figure 1. Composite score distribution and normal Q-Q plot')}
{fig('box', 'Figure 2. Indicator score distributions')}
### Classification

{md_table(band)}
{fig('bands', 'Figure 3. Participants by performance band')}
Top five (illustrative): 

{md_table(top5)}
## 4. Correlation analysis

All {len(pairs)} pairwise Pearson correlations between indicators are positive (r from {pairs.r.min():.2f} to {pairs.r.max():.2f}); {int((pairs['Significant (Holm, 0.05)']=='Yes').sum())} of {len(pairs)} remain significant after Holm correction. The strongest link is {pairs.loc[pairs.r.idxmax(),'Indicator A']} with {pairs.loc[pairs.r.idxmax(),'Indicator B']}.

{fig('corr', 'Figure 4. Pearson correlation matrix')}
Composite-vs-indicator correlations are deliberately not interpreted: they are part-whole and inflated by construction. Because this dataset is synthetic, the correlation structure reflects the simulation design, not a property of real participants.

## 5. Principal Component Analysis

PCA on the standardised indicators is justified because the correlations are moderate and uniformly positive (Bartlett chi-square = {suit['Bartlett chi-square']:.1f}, df = {suit['Bartlett df']:.0f}, p < 0.001; KMO = {suit['KMO (overall)']:.2f}).

{md_table(v)}
PC1 explains {v['Variance %'][0]:.1f}% of the variance and is the only component retained by Kaiser ({n_k}) and parallel analysis ({n_pa}). All indicators load positively on PC1 ({pcs['loadings']['PC1'].min():.2f}-{pcs['loadings']['PC1'].max():.2f}), so PC1 behaves as a **general performance factor**: participants who do well on one indicator tend to do well on the others. The remaining components are not distinguishable from noise, which also means the five indicators are not five independent dimensions. This supports reporting a single composite index, while recommending the weights be treated as a policy choice rather than a statistical finding.

{md_table(pcs['loadings'], index=True)}
{fig('scree', 'Figure 5. Scree plot with parallel analysis')}
{fig('biplot', 'Figure 6. PCA biplot coloured by cluster')}
## 6. Cluster analysis (K-means)

K-means was run on standardised indicators for k = {C.K_RANGE.start}-{C.K_RANGE.stop - 1} ({C.KMEANS_N_INIT} initialisations each) and k was chosen by highest mean silhouette, cross-checked against the elbow and Davies-Bouldin index. Selected **k = {cl['k']}** (silhouette {cl['silhouette']:.2f}).

{md_table(res['k_eval'])}
{fig('ksel', 'Figure 7. Cluster-number evaluation')}
{md_table(cl['profile'], index=True)}
{fig('prof', 'Figure 8. Standardised cluster profiles')}
Interpretation: the silhouette of {cl['silhouette']:.2f} indicates **{_silhouette_label(cl['silhouette'])}** cluster structure, and bootstrap stability is {'high' if cl['ari'].mean() >= .8 else 'moderate'} (mean adjusted Rand index {cl['ari'].mean():.2f}). Because PCA shows one dominant general factor, the clusters mainly separate participants by *overall performance level* rather than by distinct indicator mixes. They are useful as practical groupings (e.g. for targeted support) but should not be described as sharply separated types. A k = 3 solution is included in the workbook as a finer-grained alternative.

## 7. Sensitivity analysis of the weights

**Alternative scoring models.** {len(scen)} alternative weight sets were compared with the baseline using Spearman rank correlation.

{md_table(res['scen'][['Scenario','Spearman rho','Kendall tau','Top-10% overlap','Mean |rank shift|','Max |rank shift|','Band changed %']])}
{fig('scen', 'Figure 9. Agreement of alternative scoring models with the baseline')}
Rank order is highly consistent overall (Spearman rho {scen['Spearman rho'].min():.3f}-{scen['Spearman rho'].max():.3f}), yet individuals can move materially: mean rank shifts of {scen['Mean |rank shift|'].min():.0f}-{scen['Mean |rank shift|'].max():.0f} places, up to {scen['Max |rank shift|'].max():.0f} for single participants, and {scen['Band changed %'].min():.0f}-{scen['Band changed %'].max():.0f}% of participants change band. High overall correlation therefore does **not** guarantee stable individual placement.

{fig('oat', 'Figure 10. One-at-a-time weight perturbation')}
**Monte Carlo.** Across {C.MC_DRAWS} random weight sets centred on the baseline, mean Spearman rho = {mc['Mean Spearman rho']:.3f} (5th percentile {mc['5th percentile rho']:.3f}); the median 90% rank interval spans {mc['Median 90% rank-interval width']:.0f} places and the average participant keeps their band in {mc['Mean band stability %']:.0f}% of draws.

{fig('mc', 'Figure 11. Monte Carlo weight perturbation')}
## 8. Consistency, robustness and limitations

- **Reproducibility.** Repeated executions produced identical output hashes ({'confirmed' if res['repro']['identical'] else 'NOT confirmed'}; fixed seed {res['seed']}). {res['recon_detail']}.
- **Measurement noise.** Adding Gaussian noise (SD {noise['Noise SD (points)'].min()}-{noise['Noise SD (points)'].max()} points) to each indicator lowers mean rho from {noise['Mean Spearman rho'].iloc[0]:.3f} to {noise['Mean Spearman rho'].iloc[-1]:.3f}; band agreement falls from {noise['Mean band agreement %'].iloc[0]:.0f}% to {noise['Mean band agreement %'].iloc[-1]:.0f}%.

{md_table(noise)}
{fig('noise', 'Figure 12. Ranking stability under measurement noise')}
- **Boundary fragility.** {frag.Participants[0]} participants ({frag.Percent[0]:.0f}%) lie within 1 point of a band boundary and {frag.Participants[1]} ({frag.Percent[1]:.0f}%) within 2 points, so band labels for these participants are sensitive to small scoring changes. Bands should be read as approximate.

**Limitations.**

1. The data are synthetic; correlations, clusters and rankings reflect the simulation and carry no inference about real participants.
2. The weights are an assigned policy, not estimated from outcomes; the analysis shows how much the results depend on them but cannot say which weights are 'correct'.
3. The Attendance & Punctuality sub-weights (60/40) and normalised-gain definition are assumptions inherited from the supplied workbook.
4. Normalised gain is unstable when pre-test scores are high and is floored at zero, which discards information about decline.
5. K-means assumes roughly spherical clusters and was applied to bounded 0-100 scores; structure is weak.
6. Indicators are positively correlated, so the effective weight of 'general performance' exceeds any single nominal weight.
7. Team-level dependence (45 teams) was not modelled; teamwork scores may be clustered within teams.

## 9. IT competencies demonstrated

{md_table(pd.DataFrame({'Competency': ['Computational problem-solving', 'Python programming & statistical automation', 'Data validation & QA', 'Data processing & transformation', 'Algorithm development', 'Reproducible workflow', 'File handling & Excel integration', 'Visualisation & technical reporting', 'Documentation, error handling & testing'], 'Evidence in this project': ['Translating an organisational evaluation problem into a weighted index, ranking and segmentation model', 'Modular package (src/stem_model) driven by a single CLI command', f"{len(res['checks'])} automated checks with PASS/WARN/FAIL and blocking failures (validation.py)", 'Derived indicators, normalised gain, ranking and banding (scoring.py)', 'Own implementations of Bartlett/KMO, parallel analysis, Holm correction, Dirichlet Monte Carlo (analysis.py, sensitivity.py)', 'Fixed seed, pinned requirements, SHA-256 input hash, run manifest, repeat-run hash check', 'Reads the source workbook; writes a formula-driven results workbook and reconciles it with Python', '12 labelled figures; this auto-generated report', 'Docstrings, logging, custom exceptions, argparse CLI, unit tests (tests/)']}))}
## 10. Reproducing this analysis

```
pip install -r requirements.txt
python run_analysis.py --input data/STEM_Participant_Data_450_Synthetic.xlsx --output outputs
python -m unittest discover -s tests -v
```

Input SHA-256: `{res['manifest']['input_sha256'][:16]}...`; run manifest: `run_manifest.json`.
"""
    path.write_text(md, encoding="utf-8"); log.info("Report written: %s", path)
    return path

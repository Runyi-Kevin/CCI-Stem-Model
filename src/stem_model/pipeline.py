"""End-to-end orchestration: load -> validate -> score -> analyse -> report."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, analysis as A, config as C, plotting as P, reporting as R
from . import scoring as S, sensitivity as X, validation as V

log = logging.getLogger("stem_model")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(input_path, output_dir, seed: int = C.SEED, k: int | None = None, excel_reference=None) -> dict:
    t0 = time.time()
    input_path, out = Path(input_path), Path(output_dir)
    figs = out / "figures"; figs.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    # 1 load + validate ----------------------------------------------------
    raw = V.load_raw(input_path)
    clean, checks = V.validate(raw)

    # 2 score ----------------------------------------------------------------
    scored = S.build_scored_table(clean)
    recon_detail = "No independent Excel reference supplied"
    if excel_reference:
        chk = V.reconcile_with_excel(scored, excel_reference)
        checks = pd.concat([checks, pd.DataFrame([{"Check": chk.name, "Status": chk.status, "Detail": chk.detail}])],
                           ignore_index=True)
        recon_detail = f"Cross-check against the source workbook's Excel formulas: {chk.status} ({chk.detail})"
        log.info(recon_detail)
        if chk.status == "FAIL":
            raise V.DataValidationError(chk.detail)
    repro = X.reproducibility_check(lambda: S.build_scored_table(clean))
    log.info("Reproducibility: %s", repro)

    # 3 describe / correlate --------------------------------------------------
    desc, bands = A.descriptives(scored), A.band_frequencies(scored)
    r_mat, pairs = A.pearson_analysis(scored)

    # 4 PCA + clustering --------------------------------------------------------
    pca = A.pca_analysis(scored, seed)
    k_eval = A.evaluate_k(pca["Z"], seed)
    k_sel = k or A.choose_k(k_eval)
    cl = A.fit_clusters(scored, pca["Z"], k_sel, seed)
    solutions = {"Selected solution": cl}
    alt_k = 3 if k_sel != 3 else 4
    solutions[f"Alternative solution"] = A.fit_clusters(scored, pca["Z"], alt_k, seed)
    cluster_map = dict(zip(scored.participant_id, cl["labels"]))
    scored["cluster"] = cl["labels"]

    # 5 sensitivity + robustness ---------------------------------------------------
    scen, oat = X.scenario_analysis(scored), X.oat_analysis(scored)
    mc, noise, fragile = X.monte_carlo(scored, seed), X.noise_robustness(scored, seed), X.boundary_fragility(scored)

    manifest = {"project": "STEM workshop performance model (synthetic)", "version": __version__,
                "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "seed": seed,
                "input_file": input_path.name, "input_sha256": _sha256(input_path),
                "python": platform.python_version(),
                "libraries": {m: sys.modules[m].__version__ for m in
                              ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "seaborn", "openpyxl")},
                "parameters": {"weights": C.BASELINE_WEIGHTS, "k_selected": k_sel, "mc_draws": C.MC_DRAWS,
                               "mc_concentration": C.MC_CONCENTRATION, "noise_reps": C.NOISE_REPS},
                "disclaimer": C.DISCLAIMER}
    res = dict(raw=raw, clean=clean, checks=checks, scored=scored, desc=desc, bands=bands, r_mat=r_mat, pairs=pairs,
               pca=pca, k_eval=k_eval, cl=cl, cluster_solutions=solutions, cluster_map=cluster_map, scen=scen, oat=oat,
               mc=mc, noise=noise, fragile=fragile, repro=repro, recon_detail=recon_detail, seed=seed, manifest=manifest)

    # 6 outputs -------------------------------------------------------------------
    paths = {"dist": P.distribution(scored, figs), "box": P.indicator_boxplots(scored, figs),
             "bands": P.band_bars(bands, figs), "corr": P.correlation_heatmap(r_mat, figs),
             "scree": P.scree(pca, figs), "biplot": P.biplot(pca, cl["labels"], figs),
             "ksel": P.cluster_selection(k_eval, k_sel, figs), "prof": P.cluster_profiles(cl, figs),
             "scen": P.sensitivity_scenarios(scen, figs), "oat": P.oat_lines(oat, figs),
             "mc": P.monte_carlo(mc, figs), "noise": P.noise_curve(noise, figs)}
    fig_names = {key: p.name for key, p in paths.items()}
    wb = R.write_workbook(out / "STEM_Model_Results.xlsx", res)
    rep = R.write_report(out / "Technical_Report.md", res, fig_names)
    scored.to_csv(out / "scored_participants.csv", index=False)
    manifest["runtime_seconds"] = round(time.time() - t0, 1)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Done in %.1fs", manifest["runtime_seconds"])
    res["paths"] = {"workbook": wb, "report": rep}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="STEM workshop participant performance model (synthetic data)")
    ap.add_argument("--input", default="data/STEM_Participant_Data_450_Synthetic.xlsx")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--k", type=int, default=None, help="override the silhouette-selected number of clusters")
    ap.add_argument("--excel-reference", default=None,
                    help="workbook with recalculated formulas to reconcile against (optional)")
    a = ap.parse_args(argv)
    Path(a.output).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(Path(a.output) / "run.log", mode="w")])
    log.warning(C.DISCLAIMER)
    try:
        run(a.input, a.output, a.seed, a.k, a.excel_reference)
    except (FileNotFoundError, V.DataValidationError, ValueError) as exc:
        log.error("Run aborted: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

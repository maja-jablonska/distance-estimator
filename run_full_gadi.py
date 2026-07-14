#!/usr/bin/env python3
"""
run_full_gadi.py — full-dataset linear spectrophotometric-parallax run (Hogg+18
style) for the ~800k-star APOGEE bulge sample, in a single Gadi batch job.

Thin CLI driver: the library code lives in the spphot package (spphot.data for
sample assembly, spphot.linear for the model, spphot.eval for metrics). This
module re-exports the historical `run_full_gadi` API so existing imports
(`import run_full_gadi as R`) keep working.

Pipeline: prepare_sample() -> fit fold-A / fold-B / all-train (Gauss-Newton, L2)
-> cross-validated predictions for every kept star -> constant fractional err_sp
from the hi-S/N probe -> results parquet + self-contained model npz + eval report.

Usage (see run_full_gadi.pbs):
  python run_full_gadi.py --parquet <spectra.parquet> --allstar <allStar.fits> \
                          --out results_full.parquet [--lam 0.1] [--batch-rows 20000]
"""
from __future__ import annotations
import os, argparse
import numpy as np

# re-exports: the pre-package API of this module (consumers do `import run_full_gadi as R`)
from spphot.data import (LABEL_COLS, META_COLS, TELESCOPES, log,           # noqa: F401
                         load_pixel_masks, load_apogee_windows, load_metadata,
                         build_lnflux_streaming, prepare_sample)
from spphot.datasets import REGISTRY, get_dataset
from spphot.linear import (design, fit_parallax_model, predict,            # noqa: F401
                           cv_fold_scatter, save_model, load_model)
import spphot.eval as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--allstar", required=True)
    ap.add_argument("--out", default="results_full.parquet")
    ap.add_argument("--model-out", default=None,
                    help="path for the saved model npz (default: <out>_model.npz)")
    ap.add_argument("--lam", type=float, default=0.1, help="ridge strength")
    ap.add_argument("--lam-scan", default=None,
                    help="comma-separated ridge values to scan, e.g. '0.003,0.01,0.03,0.1,0.3'; "
                         "picks the lam with the lowest cross-validated fold scatter, then "
                         "runs the full pipeline with it")
    ap.add_argument("--snr-min", type=float, default=100.0, help="training S/N cut")
    ap.add_argument("--bad-frac", type=float, default=0.01, help="shared good-pixel threshold")
    ap.add_argument("--clip-sigma", type=float, default=4.0,
                    help="iterative training-set sigma-clip on the parallax residual "
                         "(default 4; 0 = off). Removes high-chi outliers (binaries, "
                         "bad astrometry); empirically improves scatter AND bias here, "
                         "but it is a parallax cut — watch the reported bias")
    ap.add_argument("--batch-rows", type=int, default=20000)
    ap.add_argument("--pixel-mask-dir", default=None,
                    help="dir with per-telescope pixel_lit_mask_<tel>.npy; keeps only "
                         "lit pixels (intersected with the data-quality mask)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="dr17", choices=sorted(REGISTRY),
                    help="photometry dataset spec (spphot.datasets registry)")
    ap.add_argument("--aux-phot", default=None,
                    help="auxiliary photometry parquet (dataset must declare aux_phot)")
    args = ap.parse_args()
    import pandas as pd

    spec_ds = get_dataset(args.dataset)
    S = prepare_sample(args.parquet, args.allstar, dataset=spec_ds,
                       aux_phot_path=args.aux_phot, snr_min=args.snr_min,
                       bad_frac=args.bad_frac, batch_rows=args.batch_rows,
                       pixel_mask_dir=args.pixel_mask_dir, seed=args.seed)
    phot_k, X_spec = S["phot"], S["spec"]
    plx_k, err_k = S["plx"], S["err"]                 # plx_k is zero-point corrected
    plx_raw_k, zpt_k = S["plx_raw"], S["zeropoint"]
    samp_k, train_k = S["sample"], S["train"]
    ids_k, dist_bj_k = S["ids"], S["dist_bj"]
    star_bad, good, tel_masks = S["star_bad"], S["good"], S["tel_masks"]
    if args.clip_sigma > 0:
        log(f"training-set sigma-clip ENABLED at {args.clip_sigma:g}sigma on the parallax "
            f"residual (may bias the estimator — compare the bias line vs a clip-off run)")

    # ---- fit three models on the training subset ----
    phot_tr, spec_tr = phot_k[train_k], X_spec[train_k]
    plx_tr, err_tr, fold_tr = plx_k[train_k], err_k[train_k], samp_k[train_k]

    def fit_on(mask, name, lam):
        Xf, st = design(phot_tr[mask], spec_tr[mask])
        th, res = fit_parallax_model(Xf, plx_tr[mask], err_tr[mask], lam=lam,
                                     clip_sigma=args.clip_sigma)
        clip_msg = f" clipped={res.n_clipped}/{mask.sum()}" if args.clip_sigma > 0 else ""
        log(f"  fit {name}: {mask.sum()} stars | lam={lam:g} converged={res.success} "
            f"iters={res.nit} obj={res.fun:.4f}{clip_msg}")
        return th, st

    # ---- optional ridge scan: choose lam by cross-validated fold scatter ----
    fold_models = None
    if args.lam_scan:
        import re
        lams = [float(x) for x in re.split(r"[,;\s]+", args.lam_scan.strip()) if x]
        log(f"lambda scan over {lams} "
            f"(cross-validated robust fractional scatter on the hi-S/N probe) ...")
        scan = {}
        for lam in lams:
            sc, fa, fb = cv_fold_scatter(phot_tr, spec_tr, plx_tr, err_tr, fold_tr, lam,
                                         clip_sigma=args.clip_sigma)
            scan[lam] = (sc, fa, fb)
            log(f"  lam={lam:<8g} CV fold scatter = {100*sc:.2f}%")
        best_lam = min(scan, key=lambda L: scan[L][0])
        log("  --- lambda scan summary (lower is better) ---")
        for lam in lams:
            mark = "   <-- selected" if lam == best_lam else ""
            log(f"    lam={lam:<8g} {100*scan[lam][0]:.2f}%{mark}")
        args.lam = best_lam
        _, (theta_A, stats_A), (theta_B, stats_B) = scan[best_lam]   # reuse winning fits
        fold_models = (theta_A, stats_A, theta_B, stats_B)

    log("fitting models ...")
    if fold_models is not None:
        theta_A, stats_A, theta_B, stats_B = fold_models      # already fit at best lam
        log(f"  reusing fold-A/B fits from the lambda scan (lam={args.lam:g})")
    else:
        theta_A, stats_A = fit_on(fold_tr == "A", "fold-A", args.lam)
        theta_B, stats_B = fit_on(fold_tr == "B", "fold-B", args.lam)
    theta_all, stats_all = fit_on(np.ones(len(plx_tr), bool), "all-train", args.lam)

    # ---- predict every kept star ----
    log("predicting spec parallaxes ...")
    plx_sp = np.empty(len(plx_k))
    A_tr = train_k & (samp_k == "A")          # training, fold A -> predicted by B
    B_tr = train_k & (samp_k == "B")          # training, fold B -> predicted by A
    rest = ~train_k                           # non-training -> all-train model
    plx_sp[A_tr] = predict(theta_B, stats_B, phot_k[A_tr], X_spec[A_tr])
    plx_sp[B_tr] = predict(theta_A, stats_A, phot_k[B_tr], X_spec[B_tr])
    plx_sp[rest] = predict(theta_all, stats_all, phot_k[rest], X_spec[rest])

    # ---- constant fractional err_sp from hi-S/N probe out-of-sample scatter ----
    probe = (train_k & np.isfinite(plx_k) & (plx_k > 0)
             & (plx_k / err_k >= 20.0))
    frac_sigma = E.robust_scatter(np.log(plx_sp[probe]) - np.log(plx_k[probe]))
    err_sp = frac_sigma * plx_sp
    dist_kpc = 1.0 / plx_sp
    log(f"adopted fractional spec error: {100*frac_sigma:.1f}% "
        f"(probe N={probe.sum()})")

    # ---- write results ----
    out = pd.DataFrame({
        "sdss_id": ids_k,
        "plx": plx_k, "e_plx": err_k,            # plx is zero-point corrected (plx_raw - zeropoint)
        "plx_raw": plx_raw_k, "zeropoint": zpt_k,
        "plx_sp": plx_sp, "err_sp": err_sp,
        "dist_sp_kpc": dist_kpc,
        "r_med_photogeo_pc": dist_bj_k,
        "sample": samp_k, "train": train_k,
        "spec_bad_frac": star_bad,
    })
    out.to_parquet(args.out, index=False)

    model_out = args.model_out or (os.path.splitext(args.out)[0] + "_model.npz")
    save_model(model_out,
               theta_all=theta_all, stats_all=stats_all,
               theta_A=theta_A, stats_A=stats_A, theta_B=theta_B, stats_B=stats_B,
               good=good, frac_sigma=frac_sigma, tel_masks=tel_masks,
               dataset=spec_ds,
               config={"lam": args.lam, "f_max": 2.0, "bad_frac": args.bad_frac,
                       "snr_min": args.snr_min, "clip_sigma": args.clip_sigma,
                       "seed": args.seed})
    log(f"wrote {args.out}  ({len(out)} stars)")
    log(f"saved model -> {model_out}")

    # ---- evaluation report on the training subset (cross-validated) ----
    cat = {"plx_a": plx_tr, "err_a": err_tr,
           "plx_sp": plx_sp[train_k], "err_sp": err_sp[train_k],
           "train": np.ones(train_k.sum(), bool), "sample": fold_tr,
           "id": ids_k[train_k]}
    print()
    E.print_report(E.evaluate(cat, label="full-data linear fit"))
    print()
    for f in ("A", "B"):
        E.print_report(E.evaluate(cat, fold=f, label="full-data linear fit"))
        print()


if __name__ == "__main__":
    main()

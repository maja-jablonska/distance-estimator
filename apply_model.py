#!/usr/bin/env python3
"""
apply_model.py — predict spectrophotometric parallaxes/distances for NEW stars
using a model saved by run_full_gadi.py, without refitting.

It reuses run_full_gadi's metadata loader, streaming spectral builder (projected
onto the SAVED good-pixel mask), and batched predictor — so apply-time
normalization is identical to training time.

Usage:
  python apply_model.py --model spphot_results_full_model.npz \
                        --parquet new_spectra.parquet \
                        --allstar astraAllStarASPCAP-0.8.0.fits \
                        --out new_distances.parquet
"""
from __future__ import annotations
import os, argparse
import numpy as np

import run_full_gadi as R   # load_model, load_metadata, build_lnflux_streaming, predict, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="npz saved by run_full_gadi.py")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--allstar", required=True)
    ap.add_argument("--out", default="applied_distances.parquet")
    ap.add_argument("--batch-rows", type=int, default=20000)
    args = ap.parse_args()
    import pandas as pd

    m = R.load_model(args.model)
    n_tel = len(m["tel_masks"]) if m["tel_masks"] else 0
    R.log(f"loaded model: {m['good_pixel_mask'].sum()} good pixels, "
          f"frac_sigma={100*m['frac_sigma']:.1f}%, telescope masks={n_tel}")

    merged, n_parquet = R.load_metadata(args.parquet, args.allstar)

    phot_all = merged[m["label_cols"]].to_numpy(float)
    keep = np.isfinite(phot_all).all(axis=1)
    R.log(f"keep (complete photometry): {keep.sum()} / {n_parquet}")

    # build spectra on the SAVED pixel mask (fixed_good) + per-telescope lit masks,
    # same normalization and imputation as training
    X_spec, _, star_bad = R.build_lnflux_streaming(
        args.parquet, keep, f_max=m["f_max"], batch_rows=args.batch_rows,
        fixed_good=m["good_pixel_mask"], tel_masks=m["tel_masks"])

    phot_k = phot_all[keep]
    ids_k = merged["sdss_id"].to_numpy()[keep]

    R.log("predicting with the all-train model ...")
    plx_sp = R.predict(m["theta_all"], m["stats_all"], phot_k, X_spec)
    err_sp = m["frac_sigma"] * plx_sp
    dist_kpc = 1.0 / plx_sp

    pd.DataFrame({
        "sdss_id": ids_k,
        "plx_sp": plx_sp, "err_sp": err_sp, "dist_sp_kpc": dist_kpc,
        "spec_bad_frac": star_bad,
    }).to_parquet(args.out, index=False)
    R.log(f"wrote {args.out}  ({keep.sum()} stars)")


if __name__ == "__main__":
    main()

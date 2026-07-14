#!/usr/bin/env python3
"""
apply_nn.py — predict spectrophotometric parallaxes/distances for NEW stars with
a heteroscedastic-NN checkpoint saved by train_nn.py, WITHOUT refitting (driver).

The library lives in spphot.nn: load_bundle() reads the checkpoint + telescope
lit-masks once, apply_to_parquet() streams the spectra onto the SAVED good-pixel
mask with the saved standardization, so apply-time features are identical to
training.

Built for the cluster test: it carries a cluster label through to the output, so
the result feeds straight into spphot.clusters. Two ways to label:
  * one file per cluster -> --cluster-name NGC6791
  * a combined file with a per-star name column -> --cluster-col <colname>
    (read from the parquet scalar columns, else from the allStar table)

Gaia parallaxes are written zero-point corrected (plx = plx_raw - zeropoint), the
same convention the model was trained on, so the cluster test compares like with
like (use offset=0.0 there). If the cluster parquet has no 'zeropoint' column,
plx falls back to plx_raw and you should pass the global offset to the test.

Usage:
  python apply_nn.py --model spphot_nn_results_model.pt \
      --parquet ngc6791_spectra.parquet --allstar astraAllStarASPCAP-0.8.0.fits \
      --pixel-mask-dir <dir> --cluster-name NGC6791 --out ngc6791_nn.parquet
"""
from __future__ import annotations
import argparse

# re-exports: run_clusters.py (and old notebooks) import these from this module
from spphot.nn import load_bundle, apply_to_parquet          # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".pt checkpoint saved by train_nn.py")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--allstar", required=True)
    ap.add_argument("--pixel-mask-dir", default=None,
                    help="same pixel-mask dir used in training (telescope lit masks)")
    ap.add_argument("--out", default="applied_nn.parquet")
    ap.add_argument("--cluster-name", default=None,
                    help="label every star with this cluster name (file == one cluster)")
    ap.add_argument("--cluster-col", default=None,
                    help="per-star cluster-name column to carry through (combined file)")
    ap.add_argument("--f-max", type=float, default=2.0,
                    help="bad-pixel flux clip; must match training (default 2.0)")
    ap.add_argument("--batch-rows", type=int, default=20000)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    bundle = load_bundle(args.model, args.pixel_mask_dir, args.device)
    apply_to_parquet(bundle, args.parquet, args.allstar, out=args.out,
                     cluster_name=args.cluster_name, cluster_col=args.cluster_col,
                     f_max=args.f_max, batch_rows=args.batch_rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_clusters.py — one command, end to end: prepare the per-cluster spectra
datasets, apply a chosen NN checkpoint to ALL clusters, and run the Hogg+18
Fig-4 cluster test on the result.

This chains the three existing pieces in a single process:
  1. prepare_cluster_spectra.prepare  — cluster.ipynb's matched-astra metadata
       parquets + the big spectra parquet -> one clean spectra parquet per cluster
       (data-quality cuts: spectrum_flags==0, one best-snr spectrum per star).
  2. apply_nn.load_bundle / apply_to_parquet — load the checkpoint + pixel masks
       ONCE, then infer plx_sp/err_sp for every cluster (features built on the
       saved good-pixel mask, identical to training).
  3. cluster_test.run_cluster_test — IVW spec-vs-Gaia mean, member tightness, and
       the Gaia-free internal-chi2 calibration check, on the concatenation.

Loading the checkpoint once (vs the per-file python invocation in apply_nn.pbs)
is the main reason to use this driver over the shell loop.

Usage
-----
    python run_clusters.py \
        --cluster-dir   /scratch/.../clusters/matched_astra_parquets \
        --spectra       /scratch/.../data/spectra_infer_parallax_zpt.parquet \
        --allstar       /scratch/.../data/astraAllStarASPCAP-0.8.0.fits \
        --model         /scratch/.../spphot_nn_results_model.pt \
        --pixel-mask-dir /scratch/.../pixel_mask \
        --work-dir      /scratch/.../clusters/run

    # re-score without re-preparing / re-inferring (tweak err-scale, offset):
    python run_clusters.py ... --skip-prep --skip-apply --err-scale 1.49

Outputs (under --work-dir):
    spectra_per_cluster/<cluster>.parquet   prepared model-ready spectra
    cluster_nn/<cluster>_nn.parquet         per-cluster inferred plx_sp/err_sp
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path

import run_full_gadi as R          # log()
import prepare_cluster_spectra as P
import apply_nn as A
import cluster_test as C


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # inputs
    ap.add_argument("--cluster-dir", required=True,
                    help="dir of <cluster>_astra_matched.parquet (cluster.ipynb output)")
    ap.add_argument("--spectra", required=True,
                    help="big spectra parquet with flux/continuum/ivar")
    ap.add_argument("--allstar", required=True,
                    help="astra allStar FITS (Gaia plx + photometry for every star)")
    ap.add_argument("--model", required=True, help=".pt checkpoint to apply")
    ap.add_argument("--pixel-mask-dir", default=None,
                    help="per-telescope pixel masks dir (MUST match training)")
    ap.add_argument("--work-dir", required=True,
                    help="working dir for prepared spectra + inferred catalogs")
    # prep options (forwarded to prepare_cluster_spectra)
    ap.add_argument("--id-col", default="sdss_id",
                    help="join key shared by cluster parquets and the spectra parquet")
    ap.add_argument("--keep-flagged", action="store_true",
                    help="skip the spectrum_flags==0 clean-spectrum cut")
    ap.add_argument("--snr-min", type=float, default=0.0,
                    help="optional snr floor at prep (default 0 = off)")
    ap.add_argument("--batch-rows", type=int, default=20000)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    # cluster-test scoring
    ap.add_argument("--offset", type=float, default=0.0,
                    help="Gaia zero-point added to plx (use if spectra lack 'zeropoint')")
    ap.add_argument("--err-scale", type=float, default=1.0,
                    help="multiply err_sp before scoring (preview your recal factor c)")
    ap.add_argument("--plx-g", default="plx", help="Gaia column to compare (plx|plx_raw)")
    ap.add_argument("--min-members", type=int, default=5)
    # stage toggles (resume / re-score without recompute)
    ap.add_argument("--skip-prep", action="store_true",
                    help="reuse existing prepared spectra in <work-dir>/spectra_per_cluster")
    ap.add_argument("--skip-apply", action="store_true",
                    help="reuse existing inferred catalogs in <work-dir>/cluster_nn")
    args = ap.parse_args()
    import pandas as pd

    work = Path(args.work_dir)
    spectra_dir = work / "spectra_per_cluster"
    nn_dir = work / "cluster_nn"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    nn_dir.mkdir(parents=True, exist_ok=True)

    # 1) prepare per-cluster spectra (cuts + one clean spectrum per star) ----------
    if args.skip_prep:
        R.log(f"[1/3] skip-prep: reusing {spectra_dir}")
    else:
        R.log("[1/3] preparing per-cluster spectra datasets ...")
        P.prepare(args.cluster_dir, args.spectra, spectra_dir, id_col=args.id_col,
                  require_clean=not args.keep_flagged, snr_min=args.snr_min,
                  batch_rows=args.batch_rows)

    import pyarrow.parquet as pq
    cluster_files = sorted(glob.glob(str(spectra_dir / "*.parquet")))
    # drop empty parquets (clusters with no member spectra): apply_to_parquet can't
    # infer the spectral grid from zero rows and would trip the saved-mask assertion
    nonempty = [f for f in cluster_files if pq.ParquetFile(f).metadata.num_rows > 0]
    for f in cluster_files:
        if f not in nonempty:
            R.log(f"skipping {Path(f).stem}: 0 spectra")
    cluster_files = nonempty
    if not cluster_files:
        raise SystemExit(f"no non-empty prepared spectra parquets in {spectra_dir}")

    # 2) load the checkpoint ONCE, infer every cluster -----------------------------
    if args.skip_apply:
        R.log(f"[2/3] skip-apply: reusing {nn_dir}")
    else:
        R.log(f"[2/3] applying {Path(args.model).name} to {len(cluster_files)} clusters ...")
        bundle = A.load_bundle(args.model, args.pixel_mask_dir, args.device)
        for f in cluster_files:
            name = Path(f).stem
            R.log(f"--- {name} ---")
            A.apply_to_parquet(bundle, f, args.allstar,
                               out=str(nn_dir / f"{name}_nn.parquet"),
                               cluster_name=name, f_max=2.0, batch_rows=args.batch_rows)

    nn_files = sorted(glob.glob(str(nn_dir / "*_nn.parquet")))
    if not nn_files:
        raise SystemExit(f"no inferred catalogs in {nn_dir}")

    # 3) cluster test on the concatenation -----------------------------------------
    R.log(f"[3/3] cluster test on {len(nn_files)} catalogs ...")
    df = pd.concat([pd.read_parquet(f) for f in nn_files], ignore_index=True)
    R.log(f"loaded {len(df)} stars from {len(nn_files)} cluster catalogs")
    members = C.members_from_labels(df, "cluster")
    rep = C.run_cluster_test(df, members, offset=args.offset, err_scale=args.err_scale,
                             plx_g=args.plx_g, min_members=args.min_members)
    C.print_cluster_test(rep)
    R.log(f"done. prepared spectra -> {spectra_dir}; inferred catalogs -> {nn_dir}")


if __name__ == "__main__":
    main()

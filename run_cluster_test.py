#!/usr/bin/env python3
"""
run_cluster_test.py — load applied per-cluster catalogs (apply_nn.py output),
group by the cluster label, and print the Hogg+18 Fig-4 cluster test.

  python run_cluster_test.py cluster_nn/*_nn.parquet
  python run_cluster_test.py cluster_nn/*_nn.parquet --err-scale 1.49   # preview recal
  python run_cluster_test.py one_combined.parquet --plx-g plx_raw --offset 0.048
"""
from __future__ import annotations
import argparse, glob
import pandas as pd
import spphot.clusters as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="applied parquet file(s) or glob(s)")
    ap.add_argument("--cluster-col", default="cluster")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="Gaia zero-point to add to plx (use if plx wasn't corrected)")
    ap.add_argument("--err-scale", type=float, default=1.0,
                    help="multiply err_sp before scoring (preview your recal factor c)")
    ap.add_argument("--plx-g", default="plx",
                    help="Gaia parallax column to compare against (plx or plx_raw)")
    ap.add_argument("--min-members", type=int, default=5)
    args = ap.parse_args()

    files = []
    for pat in args.inputs:
        files += sorted(glob.glob(pat)) or [pat]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"loaded {len(df)} stars from {len(files)} file(s)")

    members = C.members_from_labels(df, args.cluster_col)
    rep = C.run_cluster_test(df, members, offset=args.offset, err_scale=args.err_scale,
                             plx_g=args.plx_g, min_members=args.min_members)
    C.print_cluster_test(rep)


if __name__ == "__main__":
    main()

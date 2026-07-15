#!/usr/bin/env python3
"""
Combine the VVV/VIRAC2 (vvv.py) and GLIMPSE (glimpse.py) crossmatch outputs
with the original APOGEE target parquet into one master table, joined on the
shared id column. One row per APOGEE target; aux-catalogue columns are NaN
where that catalogue has no match, plus boolean in_vvv / in_glimpse flags.

Usage:
    python xmatch_combine.py apogee.parquet virac2_xmatch.parquet \
        glimpse_matched.parquet combined.parquet --id-col sdss_id

Notes:
- Both upstream scripts key their output on the SAME id as the APOGEE input
  (vvv.py --id-col, glimpse.py ID_COL), so this is an id join, not a sky
  match — the sky matching already happened server-side in each script.
- By default only the scalar columns of the APOGEE parquet are carried
  through: the bulge parquets hold spectra list-columns that decompress to
  tens of GB (same trap vvv.py sidesteps). Pass --apogee-cols to pick
  columns explicitly, or --keep-nested if you really want everything.
- Each aux table is deduped to the nearest match per id (vvv.py already
  does this; glimpse.py output can in principle repeat an id across chunk
  boundaries), and aux column names that collide with an APOGEE column get
  a _vvv / _glimpse suffix. Aux copies of the input id/ra/dec columns are
  dropped rather than suffixed — they are just the APOGEE values echoed back.
- The output is --aux-phot / spphot.datasets.AuxPhot friendly: keyed by
  --id-col, aux bands as plain columns (e.g. phot_ks_mean_mag, 4.5mag).
"""

import argparse
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_apogee(path, id_col, explicit_cols, keep_nested):
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    avail = set(schema.names)
    if id_col not in avail:
        sys.exit(f"id column '{id_col}' not in {path}. "
                 f"Available: {sorted(avail)}")
    if explicit_cols:
        missing = [c for c in explicit_cols if c not in avail]
        if missing:
            sys.exit(f"--apogee-cols {missing} not in {path}")
        cols = list(dict.fromkeys([id_col] + explicit_cols))
    elif keep_nested:
        cols = schema.names
    else:
        cols = [n for n in schema.names
                if not pa.types.is_nested(schema.field(n).type)]
        dropped = sorted(avail - set(cols))
        if dropped:
            print(f"APOGEE: skipping {len(dropped)} nested (list/struct) "
                  f"column(s): {dropped[:8]}{' ...' if len(dropped) > 8 else ''} "
                  f"(--keep-nested to include)")
    df = pq.read_table(path, columns=cols).to_pandas()
    df[id_col] = df[id_col].astype(str)
    n0 = len(df)
    df = df.drop_duplicates(subset=id_col)
    print(f"APOGEE: {len(df):,} targets"
          + (f" ({n0 - len(df):,} duplicate ids dropped)" if len(df) < n0 else ""))
    return df


def load_aux(path, name, id_col, sep_col, apogee_cols, ra_col, dec_col):
    """Read a crossmatch output: cast the id, keep matched rows only, dedupe
    to the nearest match per id, drop echoed input coords, suffix collisions."""
    df = pd.read_parquet(path)
    if id_col not in df.columns:
        sys.exit(f"id column '{id_col}' not in {path}. "
                 f"Columns: {sorted(df.columns)}")
    if sep_col not in df.columns:
        sys.exit(f"expected separation column '{sep_col}' in {path} "
                 f"(is this really the {name} output?). "
                 f"Columns: {sorted(df.columns)}")
    df[id_col] = df[id_col].astype(str)

    # vvv.py writes one row per INPUT target with NaN separation where
    # unmatched — those rows carry no aux data, so drop them here and let the
    # left join reintroduce the NaNs.
    df = df[df[sep_col].notna()]
    df = (df.sort_values(sep_col)
            .drop_duplicates(subset=id_col, keep="first"))

    echoed = [c for c in (ra_col, dec_col) if c in df.columns]
    df = df.drop(columns=echoed)

    collisions = [c for c in df.columns
                  if c != id_col and c in apogee_cols]
    df = df.rename(columns={c: f"{c}_{name}" for c in collisions})
    if collisions:
        print(f"{name}: suffixed colliding column(s) "
              f"{collisions[:8]}{' ...' if len(collisions) > 8 else ''} -> _{name}")
    print(f"{name}: {len(df):,} matched sources from {path}")
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Join APOGEE targets with their VVV/VIRAC2 and GLIMPSE "
                    "crossmatch results into one parquet.")
    ap.add_argument("apogee_in", help="original APOGEE target parquet")
    ap.add_argument("vvv_in", help="vvv.py output parquet")
    ap.add_argument("glimpse_in", help="glimpse.py output parquet")
    ap.add_argument("parquet_out")
    ap.add_argument("--id-col", default="sdss_id",
                    help="shared id column (default: sdss_id)")
    ap.add_argument("--ra-col", default="ra")
    ap.add_argument("--dec-col", default="dec",
                    help="APOGEE coordinate column names — aux copies of "
                         "these are dropped as redundant")
    ap.add_argument("--apogee-cols", nargs="+", default=None, metavar="COL",
                    help="carry only these APOGEE columns (id always kept); "
                         "default: all scalar columns")
    ap.add_argument("--keep-nested", action="store_true",
                    help="also carry nested list/struct APOGEE columns "
                         "(spectra!) — expect a huge output")
    ap.add_argument("--require", choices=["none", "any", "both"],
                    default="none",
                    help="keep all targets (none, default), only those "
                         "matched in at least one catalogue (any), or in "
                         "both (both)")
    args = ap.parse_args()

    apogee = load_apogee(args.apogee_in, args.id_col,
                         args.apogee_cols, args.keep_nested)
    vvv = load_aux(args.vvv_in, "vvv", args.id_col, "sep_arcsec",
                   set(apogee.columns), args.ra_col, args.dec_col)
    glimpse = load_aux(args.glimpse_in, "glimpse", args.id_col, "angDist",
                       set(apogee.columns), args.ra_col, args.dec_col)

    out = (apogee
           .merge(vvv, on=args.id_col, how="left")
           .merge(glimpse, on=args.id_col, how="left",
                  suffixes=("", "_glimpse")))
    sep_vvv = "sep_arcsec_vvv" if "sep_arcsec_vvv" in out.columns else "sep_arcsec"
    sep_gli = "angDist_glimpse" if "angDist_glimpse" in out.columns else "angDist"
    out["in_vvv"] = out[sep_vvv].notna()
    out["in_glimpse"] = out[sep_gli].notna()

    n = len(out)
    n_v, n_g = out["in_vvv"].sum(), out["in_glimpse"].sum()
    n_both = (out["in_vvv"] & out["in_glimpse"]).sum()
    n_any = (out["in_vvv"] | out["in_glimpse"]).sum()
    print(f"\n{n:,} APOGEE targets")
    print(f"  VVV/VIRAC2 : {n_v:,} ({100 * n_v / n:.1f}%)")
    print(f"  GLIMPSE    : {n_g:,} ({100 * n_g / n:.1f}%)")
    print(f"  both       : {n_both:,} ({100 * n_both / n:.1f}%)")
    print(f"  either     : {n_any:,} ({100 * n_any / n:.1f}%)")

    if args.require == "any":
        out = out[out["in_vvv"] | out["in_glimpse"]]
    elif args.require == "both":
        out = out[out["in_vvv"] & out["in_glimpse"]]

    out.to_parquet(args.parquet_out, index=False)
    print(f"{len(out):,} rows x {out.shape[1]} cols -> {args.parquet_out}")


if __name__ == "__main__":
    main()

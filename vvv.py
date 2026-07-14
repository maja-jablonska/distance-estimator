#!/usr/bin/env python3
"""
Cross-match stars in a parquet file against VIRAC2 via the ESO TAP service.

Usage:
    python virac2_xmatch.py targets.parquet out_virac2.parquet \
        --id-col APOGEE_ID --ra-col RA --dec-col DEC --radius 0.3

Notes:
- Positions are uploaded in chunks (TAP upload limits + politeness).
- Run discover mode first to confirm the VIRAC2 table/column names in
  the ESO tabular TAP schema, then set VIRAC2_TABLE below:
      python virac2_xmatch.py --discover
- If your RA/DEC are Gaia DR3 (epoch J2016.0) you are already on the
  VIRAC2 reference frame; 0.3 arcsec is a sensible radius. If they are
  2MASS positions (epoch ~2000), widen to ~1 arcsec or propagate first.
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
import pyvo
from astropy.table import Table

TAP_URL = "https://archive.eso.org/tap_cat"
VIRAC2_TABLE = None  # set after running --discover, e.g. "VVVX.VIRAC2_sources"
CHUNK = 20_000       # rows per upload; reduce if the service complains

# Columns to pull from VIRAC2. Trim/extend after --discover shows the schema.
# Keep the quality/crowding bookkeeping — you want it as covariates later.
VIRAC2_COLS = "v.*"


def discover(tap):
    q = """
        SELECT table_name, description
        FROM TAP_SCHEMA.tables
        WHERE table_name LIKE '%VIRAC%' OR description LIKE '%VIRAC%'
    """
    print(tap.run_sync(q).to_table())
    if VIRAC2_TABLE:
        q2 = f"""
            SELECT column_name, datatype, description
            FROM TAP_SCHEMA.columns
            WHERE table_name = '{VIRAC2_TABLE}'
        """
        print(tap.run_sync(q2).to_table())


def xmatch_chunk(tap, chunk_df, radius_arcsec):
    upload = Table.from_pandas(chunk_df[["_uid", "_ra", "_dec"]])
    adql = f"""
        SELECT a._uid, {VIRAC2_COLS},
               DISTANCE(POINT('ICRS', a._ra, a._dec),
                        POINT('ICRS', v.ra, v.dec)) * 3600.0 AS sep_arcsec
        FROM TAP_UPLOAD.targets AS a
        JOIN {VIRAC2_TABLE} AS v
          ON 1 = CONTAINS(POINT('ICRS', v.ra, v.dec),
                          CIRCLE('ICRS', a._ra, a._dec, {radius_arcsec}/3600.0))
    """
    job = tap.run_async(adql, uploads={"targets": upload})
    return job.to_table().to_pandas()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_in", nargs="?")
    ap.add_argument("parquet_out", nargs="?")
    ap.add_argument("--id-col", default="APOGEE_ID")
    ap.add_argument("--ra-col", default="RA")
    ap.add_argument("--dec-col", default="DEC")
    ap.add_argument("--radius", type=float, default=0.3,
                    help="match radius in arcsec")
    ap.add_argument("--head", type=int, default=None, metavar="X",
                    help="test mode: only cross-match the first X rows "
                         "of parquet_in")
    ap.add_argument("--discover", action="store_true",
                    help="list VIRAC2 tables/columns in the TAP schema and exit")
    args = ap.parse_args()

    tap = pyvo.dal.TAPService(TAP_URL)

    if args.discover:
        discover(tap)
        return

    if VIRAC2_TABLE is None:
        sys.exit("Set VIRAC2_TABLE at the top of this script "
                 "(run with --discover to find the name).")
    if not (args.parquet_in and args.parquet_out):
        sys.exit("Provide input and output parquet paths.")

    df = pd.read_parquet(args.parquet_in)
    if args.head is not None:
        df = df.head(args.head)
        print(f"TEST MODE: using first {len(df)} rows of {args.parquet_in}")
    for col in (args.id_col, args.ra_col, args.dec_col):
        if col not in df.columns:
            sys.exit(f"Column '{col}' not in {args.parquet_in}. "
                     f"Available: {list(df.columns)}")

    work = pd.DataFrame({
        "_uid": df[args.id_col].astype(str),
        "_ra": df[args.ra_col].astype(float),
        "_dec": df[args.dec_col].astype(float),
    }).dropna()
    # VIRAC2 footprint pre-filter: rough inner-Galaxy box to avoid
    # uploading stars with no chance of a match. Comment out if unsure.
    # (Galactic cut done properly:)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    gal = SkyCoord(work["_ra"].values * u.deg,
                   work["_dec"].values * u.deg).galactic
    l = gal.l.wrap_at(180 * u.deg).deg
    b = gal.b.deg
    in_box = (np.abs(b) < 15) & ((l > -70) & (l < 25))
    print(f"{in_box.sum()}/{len(work)} targets inside rough VVV/VVVX box; "
          f"querying those only.")
    work = work[in_box].reset_index(drop=True)

    results = []
    for i in range(0, len(work), CHUNK):
        chunk = work.iloc[i:i + CHUNK]
        t0 = time.time()
        res = xmatch_chunk(tap, chunk, args.radius)
        print(f"chunk {i // CHUNK + 1}: {len(chunk)} uploaded, "
              f"{len(res)} matches, {time.time() - t0:.0f}s")
        results.append(res)

    matched = pd.concat(results, ignore_index=True)

    # Resolve multiple matches: keep nearest, but KEEP a flag — a second
    # source inside the radius is itself a crowding covariate.
    matched["n_in_radius"] = matched.groupby("_uid")["_uid"].transform("size")
    matched = (matched.sort_values("sep_arcsec")
                      .drop_duplicates("_uid", keep="first"))

    out = df.merge(matched.rename(columns={"_uid": args.id_col}),
                   on=args.id_col, how="left")
    out.to_parquet(args.parquet_out, index=False)
    n_match = out["sep_arcsec"].notna().sum()
    print(f"{n_match}/{len(df)} targets matched -> {args.parquet_out}")


if __name__ == "__main__":
    main()
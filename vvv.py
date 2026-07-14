#!/usr/bin/env python3
"""
Cross-match stars in a parquet file against VIRAC2 via the ESO TAP service.

Usage:
    python vvv.py targets.parquet out_virac2.parquet \
        --id-col APOGEE_ID --ra-col RA --dec-col DEC --radius 0.3

Notes:
- Only the id/ra/dec columns are read from targets.parquet, and the OUTPUT is
  the slim crossmatch table (id, ra, dec, VIRAC2 columns, sep_arcsec,
  n_in_radius) keyed by --id-col — i.e. exactly the aux-photometry table
  spphot.datasets.AuxPhot consumes. Join it onto anything else by id later;
  never haul the spectra columns through this script.
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

TAP_URL = "https://archive.eso.org/tap_cat"
VIRAC2_TABLE = None  # set after running --discover, e.g. "VVVX.VIRAC2_sources"
# ESO tap_cat has TAP_UPLOAD *disabled*, so the crossmatch is an OR-of-circles
# WHERE clause (one CONTAINS per target) instead of an upload-join. Each circle
# adds ~110 chars of ADQL, so keep chunks small enough for the query to parse.
CHUNK = 500

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
    """Pull every VIRAC2 source within `radius_arcsec` of ANY target in the
    chunk (no upload — ESO tap_cat forbids TAP_UPLOAD), then assign each
    returned source to its nearest target locally and cut to the radius.

    A source sitting within the radius of two different targets is assigned
    only to the nearer one — impossible in practice at r ~ 0.3" unless two
    targets are themselves closer than the match radius."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    r_deg = radius_arcsec / 3600.0
    circles = " OR ".join(
        f"1=CONTAINS(POINT('ICRS', v.ra, v.dec), "
        f"CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {r_deg:.8f}))"
        for ra, dec in zip(chunk_df["_ra"], chunk_df["_dec"]))
    adql = f"SELECT {VIRAC2_COLS} FROM {VIRAC2_TABLE} AS v WHERE {circles}"
    res = tap.run_sync(adql).to_table().to_pandas()
    if res.empty:
        return res.assign(_uid=pd.Series(dtype=str),
                          sep_arcsec=pd.Series(dtype=float))

    src = SkyCoord(res["ra"].values * u.deg, res["dec"].values * u.deg)
    tgt = SkyCoord(chunk_df["_ra"].values * u.deg, chunk_df["_dec"].values * u.deg)
    idx, sep, _ = src.match_to_catalog_sky(tgt)
    res["_uid"] = chunk_df["_uid"].values[idx]
    res["sep_arcsec"] = sep.arcsec
    return res[res["sep_arcsec"] <= radius_arcsec]


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
    ap.add_argument("--retries", type=int, default=3,
                    help="attempts per chunk before skipping it (skipped "
                         "chunks -> <out>.failed.parquet for a re-run)")
    ap.add_argument("--table", default=None,
                    help="VIRAC2 table name in the TAP schema (overrides the "
                         "VIRAC2_TABLE constant; find it with --discover)")
    ap.add_argument("--discover", action="store_true",
                    help="list VIRAC2 tables/columns in the TAP schema and exit")
    args = ap.parse_args()

    global VIRAC2_TABLE
    if args.table:
        VIRAC2_TABLE = args.table

    tap = pyvo.dal.TAPService(TAP_URL)

    if args.discover:
        discover(tap)
        return

    if VIRAC2_TABLE is None:
        sys.exit("Set VIRAC2_TABLE at the top of this script "
                 "(run with --discover to find the name).")
    if not (args.parquet_in and args.parquet_out):
        sys.exit("Provide input and output parquet paths.")

    # Read ONLY the three columns the crossmatch needs — the bulge parquets
    # carry spectra list-columns that decompress to tens of GB, and a full
    # pd.read_parquet gets OOM-killed on a login node before --head applies.
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(args.parquet_in)
    avail = set(pf.schema_arrow.names)
    cols = [args.id_col, args.ra_col, args.dec_col]
    missing = [c for c in cols if c not in avail]
    if missing:
        sys.exit(f"Column(s) {missing} not in {args.parquet_in}. "
                 f"Available: {sorted(avail)}")
    if args.head is not None:
        batch = next(pf.iter_batches(batch_size=args.head, columns=cols))
        df = batch.to_pandas()
        print(f"TEST MODE: using first {len(df)} rows of {args.parquet_in}")
    else:
        df = pq.read_table(args.parquet_in, columns=cols).to_pandas()

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

    # A failed chunk (TAP timeout, 5xx, dropped connection) is retried, then
    # SKIPPED — its stars come out unmatched (NaN) instead of killing the run,
    # and are written to <out>.failed.parquet for a targeted re-run.
    results, failed = [], []
    n_chunks = (len(work) + CHUNK - 1) // CHUNK
    for i in range(0, len(work), CHUNK):
        chunk = work.iloc[i:i + CHUNK]
        tag = f"chunk {i // CHUNK + 1}/{n_chunks}"
        for attempt in range(1, args.retries + 1):
            t0 = time.time()
            try:
                res = xmatch_chunk(tap, chunk, args.radius)
            except Exception as e:
                wait = 30 * attempt
                print(f"{tag}: attempt {attempt}/{args.retries} failed "
                      f"({type(e).__name__}: {e}); retrying in {wait}s")
                time.sleep(wait)
                continue
            print(f"{tag}: {len(chunk)} uploaded, {len(res)} matches, "
                  f"{time.time() - t0:.0f}s")
            results.append(res)
            break
        else:
            print(f"{tag}: GIVING UP after {args.retries} attempts — "
                  f"{len(chunk)} stars carried through unmatched")
            failed.append(chunk)

    if not results:
        sys.exit("every chunk failed — is the TAP service down? "
                 "Nothing written.")
    matched = pd.concat(results, ignore_index=True)

    # Resolve multiple matches: keep nearest, but KEEP a flag — a second
    # source inside the radius is itself a crowding covariate.
    matched["n_in_radius"] = matched.groupby("_uid")["_uid"].transform("size")
    matched = (matched.sort_values("sep_arcsec")
                      .drop_duplicates("_uid", keep="first"))

    # slim aux-phot table: one row per input target, VIRAC2 cols NaN where unmatched
    df[args.id_col] = df[args.id_col].astype(str)
    out = df.merge(matched.rename(columns={"_uid": args.id_col}),
                   on=args.id_col, how="left", suffixes=("", "_virac"))
    out.to_parquet(args.parquet_out, index=False)
    n_match = out["sep_arcsec"].notna().sum()
    print(f"{n_match}/{len(df)} targets matched -> {args.parquet_out}")

    if failed:
        # sidecar with the ORIGINAL column names, so it feeds straight back in:
        #   python vvv.py <out>.failed.parquet retry.parquet --id-col ... ;
        # then combine:  out.fillna(retry)  or concat matched rows by id.
        fdf = pd.concat(failed, ignore_index=True).rename(columns={
            "_uid": args.id_col, "_ra": args.ra_col, "_dec": args.dec_col})
        fpath = args.parquet_out + ".failed.parquet"
        fdf.to_parquet(fpath, index=False)
        print(f"WARNING: {len(fdf)} stars in {len(failed)} failed chunk(s) were "
              f"never queried (unmatched in the output, NOT confirmed absent) "
              f"-> re-run them from {fpath}")


if __name__ == "__main__":
    main()
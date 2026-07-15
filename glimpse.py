"""
Cross-match APOGEE targets (from parquet) against the GLIMPSE catalogue.

Primary route: CDS X-Match against VizieR II/293/glimpse (merged GLIMPSE
I+II+3D catalogue) -- fast server-side match, handles ~10^5-10^6 rows easily.

Fallback route: IRSA TAP with table upload against glimpse2_v2cat, if you
need the native IRSA columns/flags rather than the VizieR mirror.

Usage:
    python apogee_glimpse_xmatch.py apogee.parquet glimpse_matched.parquet
"""

import sys

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table
from astroquery.xmatch import XMatch

# ---------------------------------------------------------------- config ---
RA_COL = "ra"            # adjust to your parquet schema (e.g. "ra", "RA_ICRS")
DEC_COL = "dec"
ID_COL = "sdss_id"
MATCH_RADIUS = 1.0 * u.arcsec   # IRAC PSF ~2"; 1" is standard for bright stars
CHUNK = 200_000                 # CDS upload limit is generous, but chunk anyway

GLIMPSE_VIZIER = "vizier:II/293/glimpse"


def load_apogee(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=[ID_COL, RA_COL, DEC_COL])
    df = df.dropna(subset=[RA_COL, DEC_COL]).drop_duplicates(subset=ID_COL)
    print(f"Loaded {len(df):,} unique APOGEE sources with coordinates")
    return df


def xmatch_cds(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-match via CDS X-Match service, chunked."""
    results = []
    for start in range(0, len(df), CHUNK):
        chunk = df.iloc[start:start + CHUNK]
        t = Table.from_pandas(chunk[[ID_COL, RA_COL, DEC_COL]])
        matched = XMatch.query(
            cat1=t,
            cat2=GLIMPSE_VIZIER,
            max_distance=MATCH_RADIUS,
            colRA1=RA_COL,
            colDec1=DEC_COL,
        )
        results.append(matched.to_pandas())
        print(f"  chunk {start // CHUNK + 1}: "
              f"{len(matched):,} matches from {len(chunk):,} sources")
    out = pd.concat(results, ignore_index=True)

    # Keep only the nearest match per APOGEE_ID (X-Match returns all within radius)
    out = (out.sort_values("angDist")
              .drop_duplicates(subset=ID_COL, keep="first")
              .reset_index(drop=True))
    return out


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    df = load_apogee(in_path)
    matched = xmatch_cds(df)

    n = len(matched)
    print(f"\n{n:,} matched ({100 * n / len(df):.1f}% of input)")
    print(f"median separation: {matched['angDist'].median():.3f} arcsec")

    # Useful GLIMPSE columns in II/293: 3.6mag, 4.5mag, 5.8mag, 8.0mag
    # with errors e_3.6mag etc. NaN = no detection in that band.
    matched.to_parquet(out_path, index=False)
    print(f"written -> {out_path}")


if __name__ == "__main__":
    main()

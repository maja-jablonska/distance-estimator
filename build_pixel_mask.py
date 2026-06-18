#!/usr/bin/env python3
"""
build_pixel_mask.py — from the per-spectrum APOGEE parquet, build a "lit anywhere"
pixel mask and attach the Gaia parallax zero-point.

Two products, both derived from
  scr_mk27/bulge-ages-and-orbits/data/spectra_infer_parallax.parquet
(one row == one spectrum; `pixel_flags` is an array-valued column, one bitmask per
pixel on the spectral grid):

1. Lit-anywhere pixel mask
   A pixel is "lit" in a spectrum when its `pixel_flags` value is nonzero. ORing
   across every spectrum gives a per-pixel mask that is True wherever *any* spectrum
   flagged that pixel. Telescopes are kept separate (their grids can differ). We also
   keep the per-pixel hit count (how many spectra lit each pixel) and report its
   histogram.

   Saved per telescope to <out-dir>:
     pixel_lit_mask_<telescope>.npy    bool (Lfull,)   True == lit in >=1 spectrum
     pixel_hit_count_<telescope>.npy   int  (Lfull,)   #spectra lighting each pixel
     pixel_lit_hist_<telescope>.png    histogram of the hit count

2. Gaia parallax zero-point
   zpt.get_zpt(...) on the 5-/6-parameter sources (the notebook's `zero_points`),
   one value per spectrum, written next to the source id.

     parallax_zeropoint.parquet        gaia_dr3_source_id, zeropoint
     parallax_zeropoint_hist.png       histogram of the zero-point

Streams the parquet in batches (pyarrow) reading only the columns it needs, so it
does not blow up memory the way collecting the whole frame does.

Usage:
  python build_pixel_mask.py [PARQUET] [--out-dir DIR] [--batch-size N]
                             [--skip-mask] [--skip-zpt]
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless / cluster
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

DEFAULT_PARQUET = "scr_mk27/bulge-ages-and-orbits/data/spectra_infer_parallax.parquet"

# zpt is only defined for 5- and 6-parameter astrometric solutions.
VALID_ASTROMETRIC_PARAMS = (31, 95)


def _lit_counts(rows) -> np.ndarray:
    """Per-pixel count of spectra with a nonzero flag, over a list of 1-D arrays.

    Stacks only same-grid rows (one telescope at a time), so APO/LCO grids of
    different length never get vstacked together.
    """
    flags = np.vstack([np.asarray(v) for v in rows])  # (rows, L)
    return (flags != 0).sum(axis=0).astype(np.int64)


# --------------------------------------------------------------------------- #
# 1. lit-anywhere pixel mask
# --------------------------------------------------------------------------- #
def build_lit_mask(path: str, out_dir: str, batch_size: int) -> None:
    pf = pq.ParquetFile(path)

    # per telescope: running per-pixel hit count and spectrum tally
    hit_count: dict[str, np.ndarray] = {}
    n_spectra: dict[str, int] = {}

    for batch in pf.iter_batches(batch_size=batch_size,
                                 columns=["telescope", "pixel_flags"]):
        telescopes = batch.column("telescope").to_pylist()
        flags = batch.column("pixel_flags").to_pylist()

        # group this batch's rows by telescope, then stack each group on its own
        # grid (APO and LCO can have different pixel-grid lengths)
        by_tel: dict[str, list] = {}
        for tel, row in zip(telescopes, flags):
            by_tel.setdefault(tel, []).append(row)

        for tel, rows in by_tel.items():
            counts = _lit_counts(rows)
            if tel not in hit_count:
                hit_count[tel] = counts
                n_spectra[tel] = len(rows)
            else:
                hit_count[tel] += counts
                n_spectra[tel] += len(rows)

    for tel, counts in hit_count.items():
        lit_mask = counts > 0
        n = n_spectra[tel]
        tag = str(tel)

        np.save(os.path.join(out_dir, f"pixel_lit_mask_{tag}.npy"), lit_mask)
        np.save(os.path.join(out_dir, f"pixel_hit_count_{tag}.npy"), counts)

        n_lit = int(lit_mask.sum())
        print(f"[mask] telescope={tag}: {n} spectra, grid={lit_mask.size} pixels, "
              f"lit-anywhere={n_lit} ({n_lit / lit_mask.size:.3%})")

        # --- histogram of the per-pixel hit count -------------------------- #
        # report the fraction of spectra lighting each pixel (0..1)
        frac = counts / max(n, 1)
        fig, ax = plt.subplots()
        ax.hist(frac, bins=50)
        ax.set_yscale("log")
        ax.set_xlabel("fraction of spectra lighting the pixel")
        ax.set_ylabel("number of pixels")
        ax.set_title(f"lit-pixel histogram — {tag} (N={n})")
        fig.tight_layout()
        hist_path = os.path.join(out_dir, f"pixel_lit_hist_{tag}.png")
        fig.savefig(hist_path, dpi=120)
        plt.close(fig)
        print(f"[mask] histogram -> {hist_path}")

        # text histogram so it shows up in the run log too
        edges = np.linspace(0.0, 1.0, 11)
        hist, _ = np.histogram(frac, bins=edges)
        for lo, hi, c in zip(edges[:-1], edges[1:], hist):
            print(f"        [{lo:0.1f}, {hi:0.1f})  {c}")


# --------------------------------------------------------------------------- #
# 2. Gaia parallax zero-point
# --------------------------------------------------------------------------- #
def build_zeropoint(path: str, out_dir: str, batch_size: int) -> None:
    import polars as pl
    from zero_point import zpt

    zpt.load_tables()

    cols = [
        "gaia_dr3_source_id",
        "phot_g_mean_mag",
        "nu_eff_used_in_astrometry",
        "pseudocolour",
        "ecl_lat",
        "astrometric_params_solved",
    ]
    pf = pq.ParquetFile(path)
    frames = []

    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        df = pl.from_arrow(batch)
        params = df["astrometric_params_solved"].to_numpy()
        ok = np.isin(params, VALID_ASTROMETRIC_PARAMS)
        if not ok.any():
            continue
        df = df.filter(pl.Series(ok))

        zp = zpt.get_zpt(
            df["phot_g_mean_mag"].to_numpy(),
            df["nu_eff_used_in_astrometry"].to_numpy(),
            df["pseudocolour"].to_numpy(),
            df["ecl_lat"].to_numpy(),
            df["astrometric_params_solved"].to_numpy(),
            _warnings=False,
        )
        frames.append(
            df.select("gaia_dr3_source_id").with_columns(
                pl.Series("zeropoint", zp)
            )
        )

    if not frames:
        print("[zpt] no 5-/6-parameter sources found; nothing written")
        return

    out = pl.concat(frames)
    out_path = os.path.join(out_dir, "parallax_zeropoint.parquet")
    out.write_parquet(out_path)

    zp = out["zeropoint"].to_numpy()
    finite = zp[np.isfinite(zp)]
    print(f"[zpt] {out.height} sources -> {out_path}")
    print(f"[zpt] zeropoint  median={np.nanmedian(zp):+.4f} mas  "
          f"mean={np.nanmean(zp):+.4f}  "
          f"[{np.nanmin(zp):+.4f}, {np.nanmax(zp):+.4f}]  "
          f"NaN={np.isnan(zp).sum()}")

    fig, ax = plt.subplots()
    ax.hist(finite, bins=60)
    ax.set_xlabel("Gaia parallax zero-point [mas]")
    ax.set_ylabel("number of sources")
    ax.set_title("parallax zero-point")
    fig.tight_layout()
    hist_path = os.path.join(out_dir, "parallax_zeropoint_hist.png")
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)
    print(f"[zpt] histogram -> {hist_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet", nargs="?", default=DEFAULT_PARQUET)
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: parquet's directory)")
    ap.add_argument("--batch-size", type=int, default=20_000)
    ap.add_argument("--skip-mask", action="store_true")
    ap.add_argument("--skip-zpt", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.parquet) or "."
    os.makedirs(out_dir, exist_ok=True)
    print(f"parquet  : {args.parquet}")
    print(f"out-dir  : {out_dir}")

    if not args.skip_mask:
        build_lit_mask(args.parquet, out_dir, args.batch_size)
    if not args.skip_zpt:
        build_zeropoint(args.parquet, out_dir, args.batch_size)


if __name__ == "__main__":
    main()

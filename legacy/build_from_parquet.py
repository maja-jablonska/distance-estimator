#!/usr/bin/env python3
"""
build_from_parquet.py — assemble the continuum-normalized spectral matrix for the
Hogg+18 spectrophotometric-parallax model from APOGEE parquet files.

Inputs (per star, on the 8575-pixel apStar grid), as array-valued parquet columns:
  raw        : the un-normalized spectrum S
  continuum  : the pseudo-continuum C
  ivar       : inverse variance of `raw` (APOGEE uses ivar==0 as its masked/no-data
               sentinel, so it stands in for APOGEE_PIXMASK until you add the bitmask)

Recipe:
  f  = S / C                normalized flux (~1 in continuum); feature is ln(f)
  iv = ivar * C**2          ivar propagated through the division
  bad per pixel/star, shared good-pixel mask = pixels good in >=(1-bad_frac_max)
  of stars (Hogg+18 keep ~7405 of 8575). Locally-bad survivors imputed to f=1
  (ln 0) with iv=0 so an inverse-variance-weighted fit ignores them.

Returns ln(flux) on the good-pixel subset plus the matching ivar (keep it — the
weighted fit / sigma_int head needs it).
"""
from __future__ import annotations
import numpy as np


def _stack(df, col):
    """Vstack an array-valued parquet column into (N, Lfull)."""
    return np.vstack([np.asarray(v, float) for v in df[col].to_numpy()])


def build_sample_from_parquet(
    parquet,
    spectrum_col: str = "raw",
    continuum_col: str = "continuum",
    ivar_col: str = "ivar",
    id_col: str | None = None,
    f_max: float = 2.0,
    bad_frac_max: float = 0.01,
):
    """Build (X_spec, IV_spec, good_pixel_mask, ids) from one or more parquet files.

    Parameters
    ----------
    parquet : str | Path | list
        A parquet path, a glob-expanded list of paths, or a pandas DataFrame.
        Multiple files are concatenated row-wise (one row == one star).
    spectrum_col, continuum_col, ivar_col : str
        Array-valued columns. `ivar` must be the inverse variance of `spectrum_col`
        (same array) — if your `ivar` was computed for an already-normalized flux,
        pass that flux as spectrum_col and a continuum of ones.
    id_col : str | None
        Optional per-star identifier column carried through to `ids`.
    f_max : float
        Upper sanity clip on normalized flux; >f_max flags a normalization failure.
    bad_frac_max : float
        A pixel is kept in the shared mask if it is bad in < bad_frac_max of stars.

    Returns
    -------
    dict with:
        X_spec          (N, L)   ln(normalized flux) on good pixels
        IV_spec         (N, L)   matching ivar (0 where imputed)
        good_pixel_mask (Lfull,) bool, True == kept
        ids             (N,)     ids if id_col given else integer index
    """
    import pandas as pd

    if isinstance(parquet, pd.DataFrame):
        df = parquet
    elif isinstance(parquet, (list, tuple)):
        df = pd.concat([pd.read_parquet(p) for p in parquet], ignore_index=True)
    else:
        df = pd.read_parquet(parquet)

    S  = _stack(df, spectrum_col)        # (N, Lfull) un-normalized spectrum
    C  = _stack(df, continuum_col)
    iv = _stack(df, ivar_col)
    N, Lfull = S.shape

    # --- normalize + propagate ivar ---
    C_safe = np.where(C > 0, C, np.nan)
    f  = S / C_safe
    iv = iv * C ** 2

    # --- per-pixel, per-star bad mask (ivar==0 is APOGEE's masked sentinel) ---
    bad = (
        ~np.isfinite(f) | ~np.isfinite(iv) | (iv <= 0)
        | (C <= 0) | (f <= 0) | (f > f_max)
    )

    # --- shared good-pixel mask: good in >=(1-bad_frac_max) of stars ---
    good = bad.mean(axis=0) < bad_frac_max
    L = int(good.sum())
    print(f"[parquet] stars={N}  full grid={Lfull}  good pixels kept={L} "
          f"(Hogg+18 kept 7405)")

    # --- impute locally-bad survivors: f=1 (ln 0), iv=0 (zero weight) ---
    f  = np.where(bad, 1.0, f)
    iv = np.where(bad, 0.0, iv)

    X_spec  = np.log(f[:, good])
    IV_spec = iv[:, good]

    ids = (df[id_col].to_numpy() if id_col is not None
           else np.arange(N))

    return {
        "X_spec": X_spec,
        "IV_spec": IV_spec,
        "good_pixel_mask": good,
        "ids": ids,
    }


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:] or ["spectra.parquet"]
    out = build_sample_from_parquet(paths if len(paths) > 1 else paths[0])
    print(f"X_spec  {out['X_spec'].shape}")
    print(f"IV_spec {out['IV_spec'].shape}")
    print(f"kept    {int(out['good_pixel_mask'].sum())} pixels")

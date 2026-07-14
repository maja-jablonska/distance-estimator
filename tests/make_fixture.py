#!/usr/bin/env python3
"""
make_fixture.py — generate a tiny synthetic dataset that exercises every input
schema of the pipeline (spectra parquet, allStar FITS, pixel masks, cluster
spectra, anchors table), so the whole chain can run locally in seconds:

    run_full_gadi.py -> train_nn.py -> train_v2.py -> apply_nn.py
    -> run_cluster_test.py -> prepare_cluster_spectra.py

Everything is drawn from one seeded rng, so the fixture is byte-reproducible —
tests/smoke_test.sh relies on that to compare outputs across refactor commits.

Design notes (kept in sync with tests/checks.py EXPECTED_* constants):
  * grid L=300; columns 280:300 have ivar=0 for ALL stars (a synthetic chip gap),
    and each telescope lit-mask kills 3 more distinct pixels -> the shared
    good-pixel mask is deterministically 300 - 20 - 3 - 3 = 274 pixels.
  * ln(flux/continuum) = 0.1 * z * w + noise with z = ln(plx_true) and a fixed
    per-pixel weight vector w clipped to |w|<=2, so |lnf| <= 0.22 and the f>f_max
    bad-pixel branch never fires (keeps the good mask deterministic) while the
    linear model can genuinely recover the parallax.
  * allStar has unmatched rows, duplicate sdss_id rows with NaN mags (exercises
    the completeness dedup), 5% NaN j_mag (completeness cut), 5% NaN zeropoint
    (the "plx but no zeropoint -> dropped from training" branch), and is written
    big-endian by astropy (exercises the byteswap path).
  * one fake cluster FAKE1: 30 stars at exactly d=1.5 kpc, with memberprob and
    an anchors table (mu = 5 log10(1500) - 5), for apply_nn/cluster-test/anchors.
"""
from __future__ import annotations
import argparse
import os

import numpy as np

L = 300                      # spectral grid width
GAP = slice(280, 300)        # ivar==0 chip gap, all stars
N_FIELD = 220                # field stars (in spectra parquet + allStar)
N_CLUSTER = 30               # FAKE1 members (separate spectra parquet)
CLUSTER_D_KPC = 1.5
BANDS = ["g_mag", "bp_mag", "rp_mag", "j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag"]
BASE_MAG = [14.0, 15.0, 14.5, 12.0, 11.5, 11.3, 11.2, 11.1]   # per-band absolute-ish base


def spectra_table(rng, ids, d_kpc, tel, w, snr, flags, zeropoint, memberprob=None):
    """pyarrow Table with the spectra-parquet schema (list<float32> arrays)."""
    import pyarrow as pa

    n = len(ids)
    z = -np.log(d_kpc)                                     # ln(plx_true), plx in mas @ kpc
    lnf = 0.1 * z[:, None] * w[None, :] + rng.normal(0, 0.01, (n, L))
    cont = rng.uniform(0.8, 1.2, n)[:, None] * np.ones((n, L))
    flux = cont * np.exp(lnf)
    ivar = np.full((n, L), 1.0)
    ivar[:, GAP] = 0.0                                     # chip gap: bad for everyone

    f32 = pa.list_(pa.float32())
    cols = {
        "sdss_id": pa.array(ids, pa.int64()),
        "telescope": pa.array(tel),
        "snr": pa.array(snr, pa.float64()),
        "spectrum_flags": pa.array(flags, pa.int64()),
        "zeropoint": pa.array(zeropoint, pa.float64()),
        "flux": pa.array(flux.astype(np.float32).tolist(), f32),
        "continuum": pa.array(cont.astype(np.float32).tolist(), f32),
        "ivar": pa.array(ivar.astype(np.float32).tolist(), f32),
    }
    if memberprob is not None:
        cols["memberprob"] = pa.array(memberprob, pa.float64())
    return pa.table(cols)


def allstar_rows(rng, ids, d_kpc, zeropoint):
    """dict of allStar columns for these stars (plx consistent with d and zpt)."""
    n = len(ids)
    e_plx = rng.uniform(0.01, 0.04, n)
    zpt = np.where(np.isfinite(zeropoint), zeropoint, -0.03)
    plx_raw = 1.0 / d_kpc + rng.normal(0, 1, n) * e_plx + zpt   # so plx_raw - zpt ~ 1/d
    mags = {b: base + 5 * np.log10(d_kpc) + rng.normal(0, 0.02, n)
            for b, base in zip(BANDS, BASE_MAG)}
    return {"sdss_id": ids, "plx": plx_raw, "e_plx": e_plx, **mags,
            "r_med_photogeo": 1000.0 * d_kpc}


def main():
    import pyarrow.parquet as pq
    from astropy.io import fits

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "fixture"))
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "pixel_masks"), exist_ok=True)
    os.makedirs(os.path.join(out, "cluster_meta"), exist_ok=True)

    rng = np.random.default_rng(0)
    w = np.clip(rng.normal(0, 1, L), -2, 2)               # fixed spectral weight vector

    # ---------------- field stars ----------------
    ids_f = np.arange(1000, 1000 + N_FIELD, dtype=np.int64)
    d_f = rng.uniform(0.5, 3.0, N_FIELD)
    tel_f = rng.choice(["apo25m", "lco25m"], N_FIELD, p=[0.7, 0.3])
    snr_f = rng.uniform(40, 300, N_FIELD)
    flags_f = np.where(rng.random(N_FIELD) < 0.10, 1024, 0).astype(np.int64)
    zpt_f = rng.normal(-0.03, 0.01, N_FIELD)
    zpt_f[rng.random(N_FIELD) < 0.05] = np.nan            # plx-but-no-zeropoint branch
    pq.write_table(spectra_table(rng, ids_f, d_f, tel_f, w, snr_f, flags_f, zpt_f),
                   os.path.join(out, "spectra.parquet"))

    # ---------------- cluster stars (one true distance) ----------------
    ids_c = np.arange(2000, 2000 + N_CLUSTER, dtype=np.int64)
    d_c = np.full(N_CLUSTER, CLUSTER_D_KPC)
    tel_c = np.array(["apo25m"] * N_CLUSTER)
    snr_c = rng.uniform(150, 300, N_CLUSTER)
    flags_c = np.zeros(N_CLUSTER, np.int64)
    zpt_c = np.full(N_CLUSTER, -0.03)
    prob_c = rng.uniform(0.92, 1.0, N_CLUSTER)
    pq.write_table(
        spectra_table(rng, ids_c, d_c, tel_c, w, snr_c, flags_c, zpt_c, memberprob=prob_c),
        os.path.join(out, "cluster_spectra.parquet"))

    # ---------------- allStar FITS (HDU 2 holds the table) ----------------
    rows = allstar_rows(rng, np.concatenate([ids_f, ids_c]),
                        np.concatenate([d_f, d_c]),
                        np.concatenate([zpt_f, zpt_c]))
    # 5% NaN j_mag among field stars -> completeness cut
    nan_j = rng.random(N_FIELD) < 0.05
    rows["j_mag"][:N_FIELD][nan_j] = np.nan
    # 10 unmatched stars (in allStar, not in any parquet)
    extra = allstar_rows(rng, np.arange(5000, 5010, dtype=np.int64),
                         rng.uniform(0.5, 3.0, 10), np.full(10, -0.03))
    # 5 duplicate sdss_id rows with all-NaN mags (dedup must keep the complete row)
    dup_ids = ids_f[:5]
    dup = allstar_rows(rng, dup_ids, d_f[:5], zpt_f[:5])
    for b in BANDS:
        dup[b][:] = np.nan
    table = {k: np.concatenate([rows[k], extra[k], dup[k]]) for k in rows}
    cols = [fits.Column(name="sdss_id", format="K", array=table["sdss_id"])]
    cols += [fits.Column(name=c, format="D", array=table[c])
             for c in ["plx", "e_plx", *BANDS, "r_med_photogeo"]]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(),
                  fits.BinTableHDU.from_columns(cols)]
                 ).writeto(os.path.join(out, "allStar.fits"), overwrite=True)

    # ---------------- per-telescope lit-pixel masks ----------------
    for tel, dead in [("apo25m", [10, 60, 110]), ("lco25m", [20, 70, 120])]:
        m = np.ones(L, bool)
        m[dead] = False
        np.save(os.path.join(out, "pixel_masks", f"pixel_lit_mask_{tel}.npy"), m)

    # ---------------- cluster metadata + anchors ----------------
    import pandas as pd
    pd.DataFrame({"sdss_id": ids_c,
                  "gaia_dr3_source_id": ids_c + 900000}).to_parquet(
        os.path.join(out, "cluster_meta", "FAKE1_astra_matched.parquet"), index=False)
    mu_true = 5 * np.log10(1000.0 * CLUSTER_D_KPC) - 5     # 10.88 mag at 1.5 kpc
    pd.DataFrame({"sdss_id": ids_c[:10], "mu": np.full(10, mu_true),
                  "mu_err": np.full(10, 0.03), "group": ["FAKE1"] * 10}
                 ).to_csv(os.path.join(out, "anchors.csv"), index=False)

    print(f"fixture written to {out}: {N_FIELD} field + {N_CLUSTER} cluster stars, "
          f"grid L={L}, expected shared good pixels = {L - 20 - 6}")


if __name__ == "__main__":
    main()

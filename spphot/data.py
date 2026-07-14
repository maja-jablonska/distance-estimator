"""
spphot.data — sample assembly shared by every model (linear, NN, v2).

The memory-critical data layer of the pipeline (formerly the top half of
run_full_gadi.py):
  1. load allStar FITS metadata (plx, e_plx, magnitudes, Bailer-Jones distance)
  2. merge onto the spectra parquet by sdss_id (left join -> preserves parquet row order)
  3. build the continuum-normalized ln-flux matrix in ONE streaming pass over the
     parquet (chunked, float32). Shared good-pixel mask from the ivar==0 masked
     sentinel + flux/continuum sanity. Peak RAM ~ N_keep * 8575 * 4 bytes.
  4. quality-cut training set + reproducible A/B split (seeded)

Everything funnels through prepare_sample(); the streaming builder replaced the
vstack-everything approach (~165 GB at 800k stars — see legacy/build_from_parquet.py).
"""
from __future__ import annotations
import os, time
import numpy as np

LABEL_COLS = ["g_mag", "bp_mag", "rp_mag", "j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag"]
META_COLS  = ["sdss_id", "plx", "e_plx", *LABEL_COLS, "r_med_photogeo"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------
# per-telescope "lit-anywhere" pixel masks (build_pixel_mask.py products)
# ----------------------------------------------------------------------
# A pixel is "lit" (True) when at least one spectrum from that telescope flagged
# it; the never-lit pixels are chip gaps / no-coverage. We KEEP only lit pixels
# (good_telescope = lit_mask) and intersect that, per star, with the existing
# per-pixel data-quality bad detection (ivar==0 / continuum sanity). Telescopes
# carry their own grid, so masks are stored one .npy per telescope.
TELESCOPES = ("apo1m", "apo25m", "lco25m")


def load_pixel_masks(mask_dir, telescopes=TELESCOPES):
    """{telescope: bool lit-mask} from <mask_dir>/pixel_lit_mask_<tel>.npy.

    Missing files are skipped (those telescopes get no extra masking). Returns
    None if mask_dir is falsy, so the caller stays backward-compatible."""
    if not mask_dir:
        return None
    masks = {}
    for t in telescopes:
        p = os.path.join(mask_dir, f"pixel_lit_mask_{t}.npy")
        if os.path.exists(p):
            m = np.load(p).astype(bool)
            masks[t] = m
            log(f"pixel mask {t}: keep {int(m.sum())}/{m.size} lit pixels ({p})")
        else:
            log(f"pixel mask {t}: MISSING ({p}) -> no telescope masking for {t}")
    return masks or None


def load_apogee_windows(path, width):
    """HOOK (not yet wired in): build a (width,) bool mask that is True on the union
    of the APOGEE element windows, to restrict the fit to spectral-line regions.

    To enable: load the per-element window definitions (e.g. the apogee/aspcap line
    list or the global_mask / element-window FITS) into a boolean array of length
    `width`, then AND it into each telescope's lit mask in load_pixel_masks (so
    `good_telescope = lit_mask & window_mask`). Left unimplemented on purpose — wire
    it up once the window file path/format is settled."""
    raise NotImplementedError("APOGEE element-window masking not wired up yet")


# ----------------------------------------------------------------------
# metadata: allStar FITS  +  parquet scalar columns, merged on sdss_id
# ----------------------------------------------------------------------
def load_metadata(parquet_path, allstar_path):
    import pandas as pd
    import pyarrow.parquet as pq
    from astropy.io import fits

    log("reading allStar metadata table ...")
    a = fits.open(allstar_path)[2].data            # extension 2 holds the table
    allstar = pd.DataFrame({c: np.asarray(a[c]) for c in META_COLS})
    for c in allstar.columns:                      # FITS is big-endian; pandas wants native
        arr = allstar[c].values
        if getattr(arr.dtype, "byteorder", "=") == ">":
            allstar[c] = arr.astype(arr.dtype.newbyteorder("="))

    # astra allStar has one row per spectrum/reduction, so a star (sdss_id) can
    # appear several times; META_COLS are per-star quantities, so keep one row
    # per star — preferring the most complete one in case duplicates carry NaNs.
    n_dup = int(allstar["sdss_id"].duplicated().sum())
    if n_dup:
        completeness = allstar[["plx", "e_plx", *LABEL_COLS]].notna().sum(axis=1)
        allstar = (allstar.assign(_complete=completeness)
                   .sort_values("_complete", ascending=False, kind="stable")
                   .drop_duplicates("sdss_id")
                   .drop(columns="_complete"))
        log(f"allStar: collapsed {n_dup} duplicate sdss_id rows "
            f"-> {len(allstar)} unique stars")

    log("reading parquet scalar columns (sdss_id, snr, spectrum_flags, zeropoint) ...")
    avail = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    scalar_cols = ["sdss_id", "snr", "spectrum_flags"]
    has_zpt = "zeropoint" in avail
    if has_zpt:
        scalar_cols.append("zeropoint")
    meta = pq.read_table(parquet_path, columns=scalar_cols).to_pandas()
    if not has_zpt:
        meta["zeropoint"] = np.nan
        log("parquet has no 'zeropoint' column -> parallax zero-point correction disabled")
    n_parquet = len(meta)

    # left join keeps parquet row order, so it stays aligned to the streamed spectra
    merged = meta.merge(allstar, on="sdss_id", how="left")
    assert len(merged) == n_parquet, "merge changed row count (duplicate sdss_id in allStar?)"
    log(f"metadata: {n_parquet} parquet rows, "
        f"{merged['plx'].notna().sum()} with allStar match")
    return merged, n_parquet


# ----------------------------------------------------------------------
# streaming spectral builder (the memory-critical part)
# ----------------------------------------------------------------------
def _list_col_2d(arr, width):
    """ListArray of fixed-length float sublists -> (n, width) float64, fast path."""
    flat = arr.values.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    assert flat.size % width == 0, "ragged spectral column (non-fixed length)"
    return flat.reshape(-1, width)


def build_lnflux_streaming(parquet_path, keep_mask, f_max=2.0, bad_frac_max=0.01,
                           batch_rows=20000, fixed_good=None, tel_masks=None):
    """One streaming pass. Returns:
        X_spec    (n_keep, L) float32  ln(normalized flux) on shared good pixels
        good      (Lfull,)    bool     kept-pixel mask
        star_bad  (n_keep,)   float32  per-star bad-pixel fraction (data-quality flag)
    keep_mask is in PARQUET ROW ORDER (same order as load_metadata's merged frame).
    If fixed_good is given (a saved model's mask), it is used verbatim instead of
    recomputing — so new spectra land on exactly the pixels the model was fit on.
    If tel_masks (a {telescope: lit-mask} dict from load_pixel_masks) is given, a
    pixel not lit by a star's telescope is treated as bad for that star (intersected
    with the data-quality bad detection), so the feature only ever uses lit pixels."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    n_keep = int(keep_mask.sum())
    lnfull = None
    bad_count = None
    star_bad = np.empty(n_keep, np.float32)
    width = None
    out = 0          # next free row in the kept-output arrays
    row0 = 0         # running offset into the full parquet
    t0 = time.time()
    unknown_tels = set()

    columns = ["flux", "continuum", "ivar"]
    if tel_masks is not None:
        columns.append("telescope")

    for bi, batch in enumerate(pf.iter_batches(batch_size=batch_rows,
                                               columns=columns)):
        bsz = batch.num_rows
        sel = keep_mask[row0:row0 + bsz]
        row0 += bsz
        k = int(sel.sum())
        if k == 0:
            continue
        if width is None:
            width = batch.column("flux").values.to_numpy(zero_copy_only=False).size // bsz
            lnfull = np.zeros((n_keep, width), np.float32)
            bad_count = np.zeros(width, np.int64)
            if tel_masks is not None:
                for t, m in tel_masks.items():
                    if m.size != width:
                        raise ValueError(f"pixel mask {t} length {m.size} != grid {width}")

        flux = _list_col_2d(batch.column("flux"), width)[sel]
        cont = _list_col_2d(batch.column("continuum"), width)[sel]
        ivar = _list_col_2d(batch.column("ivar"), width)[sel]

        C = np.where(cont > 0, cont, np.nan)
        f = flux / C                                              # normalized flux
        bad = (~np.isfinite(f) | ~np.isfinite(ivar) | (ivar <= 0)
               | (cont <= 0) | (f <= 0) | (f > f_max))            # ivar==0 is APOGEE's mask sentinel
        star_bad[out:out + k] = bad.mean(axis=1)                  # quality flag, telescope-independent

        if tel_masks is not None:                                 # keep only telescope-lit pixels (AND)
            tels = np.asarray(batch.column("telescope").to_pylist())[sel]
            lit = np.ones((k, width), bool)
            for t in np.unique(tels):
                m = tel_masks.get(str(t))
                if m is None:
                    unknown_tels.add(str(t))
                    continue                                      # unknown telescope -> no extra masking
                lit[tels == t] = m
            bad |= ~lit

        f = np.where(bad, 1.0, f)                                 # impute bad -> continuum (ln 0)
        lnfull[out:out + k] = np.log(f).astype(np.float32)
        bad_count += bad.sum(axis=0)
        out += k

        if bi % 20 == 0:
            log(f"  spectra: {out}/{n_keep} kept rows "
                f"({(time.time()-t0):.0f}s)")

    if unknown_tels:
        log(f"WARNING: no pixel mask for telescopes {sorted(unknown_tels)} "
            f"-> those spectra kept all data-quality-good pixels")

    assert out == n_keep, f"filled {out} rows, expected {n_keep}"
    if fixed_good is not None:
        assert fixed_good.size == width, "saved mask length != spectral grid"
        good = fixed_good
        log(f"spectra done: grid={width}, applying saved mask L={int(good.sum())}")
    else:
        good = bad_count < bad_frac_max * n_keep
        log(f"spectra done: grid={width}, good pixels kept={int(good.sum())} "
            f"(Hogg+18 kept 7405)")
    X_spec = lnfull[:, good].copy()                              # (n_keep, L) float32
    del lnfull                                                   # free the full grid
    return X_spec, good, star_bad


# ----------------------------------------------------------------------
# sample assembly — shared by the linear fit and the NN workflow
# ----------------------------------------------------------------------
def prepare_sample(parquet, allstar, *, snr_min=100.0, bad_frac=0.01,
                   batch_rows=20000, pixel_mask_dir=None, seed=42):
    """Load + assemble the modelling sample once, so every model (linear or NN) is
    trained and scored on identical features/targets/splits: allStar+parquet
    metadata, the Gaia parallax zero-point (plx_corr = plx - zeropoint), the
    per-telescope lit-pixel masks, the quality-cut training set (no parallax/S-N
    cut), a reproducible A/B split, and the streamed ln-flux matrix.

    Returns a dict of kept-row arrays (parquet row order, restricted to stars with
    complete photometry):
        phot (n,8)  spec (n,L)  plx err  plx_raw zeropoint  sample('A'/'B')
        train(bool)  ids  dist_bj  star_bad   plus good(Lfull,), tel_masks, n_parquet
    """
    tel_masks = load_pixel_masks(pixel_mask_dir)
    merged, n_parquet = load_metadata(parquet, allstar)

    phot_all = merged[LABEL_COLS].to_numpy(float)
    keep = np.isfinite(phot_all).all(axis=1)
    log(f"keep (complete photometry): {keep.sum()} / {n_parquet}")

    # zero-point: plx_corr = plx - zeropoint (NaN for non-5/6-param sources -> the
    # isfinite(plx_corr) cut below drops them from training)
    plx_raw = merged["plx"].to_numpy(float)
    zpt_all = merged["zeropoint"].to_numpy(float)
    plx_all = plx_raw - zpt_all
    err_all = merged["e_plx"].to_numpy(float)
    snr_ok  = merged["snr"].to_numpy(float)
    flags   = merged["spectrum_flags"].to_numpy()
    n_zpt = int(np.isfinite(zpt_all).sum())
    n_no_zpt = int((np.isfinite(plx_raw) & ~np.isfinite(zpt_all)).sum())
    log(f"zero-point: applied to {n_zpt} sources; "
        f"{n_no_zpt} stars have plx but no zeropoint (dropped from training)")
    train = (keep & (snr_ok > snr_min) & (flags == 0)
             & np.isfinite(plx_all) & np.isfinite(err_all) & (err_all > 0))

    # reproducible 50/50 A/B split, stratified on train / non-train
    rng = np.random.default_rng(seed)
    sample = np.full(n_parquet, "B")
    for mask in (train, keep & ~train):
        idx = np.where(mask)[0]
        sample[idx[rng.permutation(len(idx))[:len(idx) // 2]]] = "A"
    log(f"training stars: {train.sum()}  | negative plx kept: "
        f"{100*(plx_all[train] < 0).mean():.1f}%")

    X_spec, good, star_bad = build_lnflux_streaming(
        parquet, keep, bad_frac_max=bad_frac, batch_rows=batch_rows, tel_masks=tel_masks)

    return {
        "phot": phot_all[keep], "spec": X_spec,
        "plx": plx_all[keep], "err": err_all[keep],
        "plx_raw": plx_raw[keep], "zeropoint": zpt_all[keep],
        "sample": sample[keep], "train": train[keep],
        "ids": merged["sdss_id"].to_numpy()[keep],
        "dist_bj": merged["r_med_photogeo"].to_numpy(float)[keep],
        "star_bad": star_bad, "good": good, "tel_masks": tel_masks,
        "n_parquet": n_parquet,
    }

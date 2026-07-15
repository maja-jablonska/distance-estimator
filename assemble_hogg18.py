#!/usr/bin/env python
"""assemble_hogg18.py — rebuild the Hogg+18 (Paper I) INPUT dataset for verification.

The repo's hogg2018.fits (= Zenodo 1468053) is the paper's OUTPUT catalog: the
exact 44784-star Parent Sample with their training_set flag and A/B split. This
driver rebuilds the model INPUTS for those stars, in the two artifacts
spphot.data.prepare_sample() consumes:

    data/hogg18/hogg18_allstar.fits      allStar-style table in HDU 2:
        sdss_id (=APOGEE_ID), plx, e_plx (Gaia DR2, raw), g_mag..w2_mag,
        r_med_photogeo (NaN), + hogg_training_set / hogg_sample / logg /
        Gaia astrometric-quality columns for paper-cut replication.
    data/hogg18/hogg18_spectra.parquet   one row per star, same order as the
        meta parquet: sdss_id, snr, spectrum_flags(=0), zeropoint(=-0.048),
        telescope, flux / continuum / ivar list columns on the 8575-pixel grid.

Sources (paper §4): APOGEE DR14 allStar-l31c.2 (J,H,K + spectra pointers),
Gaia DR2 archive joins tmass_best_neighbour -> gaia_source (G, BP, RP,
parallax) and allwise_best_neighbour -> allwise_original_valid (w1mpro,
w2mpro). Spectra are aspcapStar-r8 files, pseudo-continuum-normalized exactly
as Eilers' normalize_all_spectra.py (Chebyshev deg-2 per chip, ivar-weighted
on the Cannon continuum pixels pixtest8_dr13.txt, sigma>0.3 / sigma>1 + grow-1
bad masks). Bad pixels get ivar=0 (the builder's mask sentinel), so
build_lnflux_streaming reproduces Hogg's ln-flux features (bad -> ln f = 0).

zeropoint = -0.048 mas: the pipeline computes plx_corr = plx - zeropoint,
which equals Hogg's adjusted parallax plx + 0.048.

NOTE on training selection: prepare_sample's default train mask
(snr>100 & flags==0) is NOT the paper's (parallax_error<0.1 &
visibility_periods_used>=8 & chi2 criterion). For an exact Paper-I
replication, override train/sample with hogg_training_set / hogg_sample.

Stages (resumable; default: all):
    meta     download allStar DR14 (~500 MB, cached), query the Gaia archive,
             write hogg18_allstar.fits + hogg18_meta.parquet
    spectra  download 44784 aspcapStar files (~10 GB traffic, nothing kept on
             disk), normalize, write spectra_chunks/chunk_XXXX.parquet
    concat   merge chunks -> hogg18_spectra.parquet, delete chunks
    verify   counts + color cuts vs the paper, spot-check normalization

Run:  python assemble_hogg18.py [--stage meta|spectra|concat|verify|all]
"""
from __future__ import annotations
import argparse
import io
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZENODO_FITS = os.path.join(HERE, "hogg2018.fits")
ALLSTAR_URL = ("https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/"
               "stars/l31c/l31c.2/allStar-l31c.2.fits")
SPEC_BASE = ("https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/"
             "stars/l31c/l31c.2")
PIXTEST_URL = ("https://raw.githubusercontent.com/aceilers/"
               "spectroscopic_parallax/master/data/pixtest8_dr13.txt")
ZENODO_URL = ("https://zenodo.org/records/1468053/files/"
              "data_HoggEilersRix2018.fits?download=1")
ZEROPOINT = -0.048          # plx - (-0.048) = plx + 0.048 (Hogg's offset)
NPIX = 8575
LARGE = 3.0                 # Eilers' bad-pixel sigma sentinel


def set_out(out_dir):
    """Bind all output paths under out_dir (on Gadi: somewhere in /scratch)."""
    global OUT, ALLSTAR_LOCAL, PIXLIST, META_PARQUET, ALLSTAR_OUT
    global CHUNK_DIR, SPEC_PARQUET
    OUT = os.path.abspath(out_dir)
    ALLSTAR_LOCAL = os.path.join(OUT, "allStar-l31c.2.fits")
    PIXLIST = os.path.join(OUT, "pixtest8_dr13.txt")
    META_PARQUET = os.path.join(OUT, "hogg18_meta.parquet")
    ALLSTAR_OUT = os.path.join(OUT, "hogg18_allstar.fits")
    CHUNK_DIR = os.path.join(OUT, "spectra_chunks")
    SPEC_PARQUET = os.path.join(OUT, "hogg18_spectra.parquet")


set_out(os.path.join(HERE, "data", "hogg18"))


def _ensure_small_inputs():
    """Fetch the tiny support files if absent (repo checkout usually has both)."""
    global ZENODO_FITS
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(PIXLIST):
        log(f"fetching pixtest8_dr13.txt -> {PIXLIST}")
        urllib.request.urlretrieve(PIXTEST_URL, PIXLIST)
    if not os.path.exists(ZENODO_FITS):
        ZENODO_FITS = os.path.join(OUT, "data_HoggEilersRix2018.fits")
        if not os.path.exists(ZENODO_FITS):
            log(f"fetching Zenodo catalog -> {ZENODO_FITS}")
            urllib.request.urlretrieve(ZENODO_URL, ZENODO_FITS)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------
# stage: meta
# ----------------------------------------------------------------------
def stage_meta():
    import pandas as pd
    from astropy.io import fits
    from astropy.table import Table

    if os.path.exists(META_PARQUET) and os.path.exists(ALLSTAR_OUT):
        log(f"meta outputs already present in {OUT} -> skipping stage")
        return
    _ensure_small_inputs()
    if not os.path.exists(ALLSTAR_LOCAL):
        log(f"downloading {ALLSTAR_URL} (resumable via curl) ...")
        rc = os.system(f"curl -L -C - -o '{ALLSTAR_LOCAL}.part' '{ALLSTAR_URL}' "
                       f"&& mv '{ALLSTAR_LOCAL}.part' '{ALLSTAR_LOCAL}'")
        if rc != 0:
            sys.exit("allStar download failed")

    log("reading Zenodo catalog (the definitive 44784-star list) ...")
    z = fits.open(ZENODO_FITS)[1].data
    zc = pd.DataFrame({
        "sdss_id": np.char.strip(z["2MASS_ID"].astype(str)),
        "hogg_gaia_parallax": z["Gaia_parallax"].astype(float),
        "hogg_gaia_parallax_err": z["Gaia_parallax_err"].astype(float),
        "spec_parallax": z["spec_parallax"].astype(float),
        "spec_parallax_err": z["spec_parallax_err"].astype(float),
        "hogg_training_set": z["training_set"].astype(int),
        "hogg_sample": np.char.strip(z["sample"].astype(str)),
    })
    assert len(zc) == 44784 and zc["sdss_id"].is_unique

    log("reading allStar DR14 ...")
    a = fits.open(ALLSTAR_LOCAL)[1].data
    cols = {
        "sdss_id": np.char.strip(a["APOGEE_ID"].astype(str)),
        "ra": a["RA"].astype(float), "dec": a["DEC"].astype(float),
        "j_mag": a["J"].astype(float), "h_mag": a["H"].astype(float),
        "k_mag": a["K"].astype(float),
        "snr": a["SNR"].astype(float), "logg": a["LOGG"].astype(float),
        "teff": a["TEFF"].astype(float), "fe_h": a["FE_H"].astype(float),
        "location_id": a["LOCATION_ID"].astype(int),
        "field": np.char.strip(a["FIELD"].astype(str)),
        "file": np.char.strip(a["FILE"].astype(str)),
        "telescope": np.char.strip(a["TELESCOPE"].astype(str)),
    }
    allstar = pd.DataFrame(cols)
    allstar = allstar[allstar["sdss_id"].isin(set(zc["sdss_id"]))]
    # a star re-observed in several fields has several allStar rows (and
    # spectra); like Eilers' parent_sample.py, keep the first occurrence
    n_dup = int(allstar["sdss_id"].duplicated().sum())
    allstar = allstar.drop_duplicates("sdss_id", keep="first")
    log(f"allStar: matched {len(allstar)}/44784 stars ({n_dup} duplicate rows dropped)")

    # ---- Gaia DR2: the archive kills anonymous uploads (quota) and string
    # IN-lists on tmass_best_neighbour (no index -> 408), so instead:
    # (1) CDS XMatch on 2MASS positions -> candidate Gaia source_ids,
    # (2) ESA archive numeric-IN on tmass_best_neighbour to keep exactly the
    #     archive's official 2MASS assignment (what the paper used),
    # (3) ESA archive numeric-IN join for photometry/astrometry/AllWISE.
    from astroquery.gaia import Gaia
    import astropy.units as u
    Gaia.ROW_LIMIT = -1

    def esa_numeric_in(template, ids, step=500, tag=""):
        # step < 2000 so the anonymous sync-row cap can never truncate a chunk.
        # Each chunk is cached to disk: the archive drops connections at random
        # (evening flakiness / rate limiting), and reruns must cost nothing.
        # step is part of the cache key — a different step must never reuse files.
        cache_dir = os.path.join(OUT, "esa_cache")
        os.makedirs(cache_dir, exist_ok=True)
        parts = []
        n_chunks = (len(ids) + step - 1) // step
        for i in range(0, len(ids), step):
            cpath = os.path.join(cache_dir, f"{tag}_s{step}_{i//step:04d}.parquet")
            if os.path.exists(cpath):
                parts.append(pd.read_parquet(cpath))
                continue
            inlist = ",".join(map(str, ids[i:i + step]))
            for attempt in range(7):
                try:
                    job = Gaia.launch_job(template.format(inlist=inlist))
                    chunk = job.get_results().to_pandas()
                    chunk.columns = [c.lower() for c in chunk.columns]
                    for c in chunk.columns:      # bytes -> str for parquet
                        if chunk[c].dtype == object:
                            chunk[c] = chunk[c].apply(
                                lambda s: s.decode() if isinstance(s, bytes) else s)
                    chunk.to_parquet(cpath, index=False)
                    parts.append(chunk)
                    break
                except Exception as e:
                    if attempt == 6:
                        raise
                    wait = 15 * 2 ** attempt      # 15s .. 16 min
                    log(f"  {tag} chunk {i//step}: {type(e).__name__}, "
                        f"retry in {wait}s ...")
                    time.sleep(wait)
            log(f"  {tag} chunk {i//step + 1}/{n_chunks}: {len(parts[-1])} rows")
        return pd.concat(parts, ignore_index=True)

    match_cache = os.path.join(OUT, "gaia_match_cache.parquet")
    if os.path.exists(match_cache):
        good = pd.read_parquet(match_cache)
        log(f"Gaia: loaded {len(good)} archive-confirmed matches from cache")
    else:
        log("CDS XMatch: 2MASS positions -> Gaia DR2 candidates ...")
        from astroquery.xmatch import XMatch
        pos = zc[["sdss_id"]].merge(allstar[["sdss_id", "ra", "dec"]],
                                    on="sdss_id", how="left")
        cand = []
        for radius in (2.0, 5.0):                # widen only for the stragglers
            todo = pos if radius == 2.0 else pos[~pos["sdss_id"].isin(
                {c for t in cand for c in t["sdss_id"]})]
            if not len(todo):
                break
            xm = XMatch.query(cat1=Table.from_pandas(todo),
                              cat2="vizier:I/345/gaia2",
                              max_distance=radius * u.arcsec,
                              colRA1="ra", colDec1="dec").to_pandas()
            cand.append(xm[["sdss_id", "source_id"]])
            log(f"  XMatch r={radius}\": {len(xm)} candidate pairs")
        cand = pd.concat(cand, ignore_index=True).drop_duplicates()
        cand["source_id"] = cand["source_id"].astype(np.int64)

        log("ESA archive: official tmass_best_neighbour assignment ...")
        sids = np.unique(cand["source_id"].to_numpy())
        arch = esa_numeric_in(
            "SELECT xm.source_id, xm.original_ext_source_id AS tmass, "
            "xm.angular_distance AS tmass_dist "
            "FROM gaiadr2.tmass_best_neighbour xm WHERE xm.source_id IN ({inlist})",
            sids, tag="tmass")
        arch["tmass"] = arch["tmass"].apply(
            lambda s: s.decode() if isinstance(s, bytes) else s)
        arch["sdss_id"] = "2M" + arch["tmass"].astype(str)
        # keep only candidates whose archive 2MASS assignment IS our star
        good = cand.merge(arch.drop(columns=["tmass"]), on=["sdss_id", "source_id"])
        # a 2MASS source can be best neighbour of >1 Gaia source: keep the closest
        n_multi = int(good["sdss_id"].duplicated().sum())
        good = (good.sort_values("tmass_dist", kind="stable")
                    .drop_duplicates("sdss_id", keep="first"))
        log(f"Gaia: {len(good)}/44784 archive-confirmed matches "
            f"({n_multi} multi-match resolved by distance)")
        good.to_parquet(match_cache, index=False)

    # the triple join gaia_source x allwise_best_neighbour x allwise_original_valid
    # exceeds the sync timeout even on indexed numeric INs -> three flat queries,
    # cached so a crash after the ~45-min query pass costs nothing to rerun
    phot_cache = os.path.join(OUT, "gaia_phot_cache.parquet")
    if os.path.exists(phot_cache):
        gtab = pd.read_parquet(phot_cache)
        log(f"Gaia: loaded photometry for {len(gtab)} stars from cache")
        return _write_meta(zc, allstar, gtab)

    log("ESA archive: gaia_source photometry + astrometry ...")
    sids = good["source_id"].to_numpy()
    gphot = esa_numeric_in(
        "SELECT g.source_id, g.phot_g_mean_mag AS g_mag, "
        "g.phot_bp_mean_mag AS bp_mag, g.phot_rp_mean_mag AS rp_mag, "
        "g.parallax AS plx, g.parallax_error AS e_plx, "
        "g.visibility_periods_used, g.astrometric_chi2_al, "
        "g.astrometric_n_good_obs_al, g.phot_variable_flag "
        "FROM gaiadr2.gaia_source g WHERE g.source_id IN ({inlist})",
        sids, tag="gaia")
    gphot["phot_variable_flag"] = gphot["phot_variable_flag"].apply(
        lambda s: s.decode() if isinstance(s, bytes) else s)

    log("ESA archive: allwise_best_neighbour ...")
    wx = esa_numeric_in(
        "SELECT xm.source_id, xm.allwise_oid "
        "FROM gaiadr2.allwise_best_neighbour xm WHERE xm.source_id IN ({inlist})",
        sids, tag="wise-x")

    log("ESA archive: AllWISE photometry ...")
    wphot = esa_numeric_in(
        "SELECT w.allwise_oid, w.w1mpro AS w1_mag, w.w2mpro AS w2_mag "
        "FROM gaiadr1.allwise_original_valid w WHERE w.allwise_oid IN ({inlist})",
        np.unique(wx["allwise_oid"].to_numpy()), tag="wise")

    gtab = (good.merge(gphot, on="source_id", how="left")
                .merge(wx, on="source_id", how="left")
                .merge(wphot, on="allwise_oid", how="left")
                .drop(columns=["allwise_oid"]))
    log(f"Gaia: {len(gtab)} rows, {int(gtab['w2_mag'].notna().sum())} with WISE")
    gtab.to_parquet(phot_cache, index=False)
    _write_meta(zc, allstar, gtab)


def _write_meta(zc, allstar, gtab):
    import pandas as pd
    from astropy.table import Table

    meta = zc.merge(allstar, on="sdss_id", how="left").merge(
        gtab, on="sdss_id", how="left")
    assert len(meta) == 44784
    meta["zeropoint"] = ZEROPOINT
    meta["spectrum_flags"] = 0
    meta["r_med_photogeo"] = np.nan

    for col in ("plx", "j_mag", "g_mag", "w2_mag"):
        n = int(meta[col].isna().sum())
        if n:
            log(f"WARNING: {n} stars missing {col}")
    dplx = (meta["plx"] - meta["hogg_gaia_parallax"]).abs()
    log(f"parallax vs Zenodo catalog: median |diff| = {np.nanmedian(dplx):.2e} mas, "
        f"max = {np.nanmax(dplx):.2e} mas")

    meta.to_parquet(META_PARQUET, index=False)
    log(f"wrote {META_PARQUET}")

    fits_cols = ["sdss_id", "plx", "e_plx", "g_mag", "bp_mag", "rp_mag",
                 "j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag",
                 "r_med_photogeo", "snr", "logg", "teff", "fe_h",
                 "hogg_training_set", "visibility_periods_used",
                 "astrometric_chi2_al", "astrometric_n_good_obs_al"]
    t = Table.from_pandas(meta[fits_cols].assign(
        hogg_sample=meta["hogg_sample"].values))
    from astropy.io import fits as afits
    hdul = afits.HDUList([afits.PrimaryHDU(), afits.ImageHDU(),
                          afits.BinTableHDU(t)])   # table in HDU 2, as the loader expects
    hdul.writeto(ALLSTAR_OUT, overwrite=True)
    log(f"wrote {ALLSTAR_OUT} (table in HDU 2)")


# ----------------------------------------------------------------------
# stage: spectra
# ----------------------------------------------------------------------
def _fetch(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(2.0 * (i + 1))


def _read_spectrum(buf):
    """aspcapStar: HDU1 = flux, HDU2 = sigma (Eilers' reader)."""
    from astropy.io import fits
    with fits.open(io.BytesIO(buf), memmap=False) as h:
        flux = np.array(h[1].data, float)
        sig = np.array(h[2].data, float)
        if flux.ndim > 1:
            flux, sig = flux[0], sig[0]
        hdr = h[1].header
        wl = 10.0 ** (hdr["CRVAL1"] + hdr["CDELT1"] * np.arange(len(flux)))
    return flux, sig, wl


class Normalizer:
    """Eilers' NormalizeData(), restructured per star on a fixed grid."""

    def __init__(self, wl):
        self.wl = wl
        pix = np.loadtxt(PIXLIST, usecols=(0,)).astype(int)
        self.var_array = np.full(NPIX, LARGE ** 2)
        self.var_array[pix] = 0.0
        self.takes = [(wl > 15150) & (wl < 15800),
                      (wl > 15890) & (wl < 16430),
                      (wl > 16490) & (wl < 16950)]

    def __call__(self, flux, sig):
        """-> (flux_imputed, continuum, ivar) with ivar=0 marking bad pixels."""
        flux, sig = flux.copy(), sig.copy()
        bad0 = ~np.isfinite(flux) | ~np.isfinite(sig) | (sig <= 0)
        flux[bad0], sig[bad0] = 1.0, LARGE
        w = 1.0 / (sig ** 2 + self.var_array)
        cont = np.zeros(NPIX)
        fnorm = np.ones(NPIX)
        snorm = np.full(NPIX, LARGE)
        for take in self.takes:
            fit = np.polynomial.chebyshev.Chebyshev.fit(
                self.wl[take], flux[take], w=w[take], deg=2)
            c = fit(self.wl[take])
            cont[take] = c
            fnorm[take] = flux[take] / c
            snorm[take] = sig[take] / c
        bad = snorm > 0.3                       # Eilers' first magic cut
        fnorm[bad], snorm[bad] = 1.0, LARGE
        bad_a = (~np.isfinite(fnorm) | ~np.isfinite(snorm)
                 | (snorm <= 0) | (snorm > 1.0))
        grow = bad_a | np.roll(bad_a, 1) | np.roll(bad_a, -1)
        grow[0] |= bad_a[1]                     # np.roll wraps; edges via neighbours only
        grow[-1] |= bad_a[-2]
        in_window = self.takes[0] | self.takes[1] | self.takes[2]
        dead = grow | bad | bad0 | ~in_window | (cont <= 0)
        ivar = np.where(dead, 0.0, 1.0 / sig ** 2)   # ivar of the RAW flux
        return flux, cont, ivar


def _spectrum_urls(row):
    fname = f"aspcapStar-r8-l31c.2-{row.sdss_id}.fits"
    urls = [f"{SPEC_BASE}/{row.location_id}/{fname}"]
    if row.location_id == 1:                    # apo1m lives under the field name
        urls = [f"{SPEC_BASE}/{row.field}/{fname}"] + urls
    if isinstance(row.file, str) and row.file and row.file != fname:
        urls.append(f"{SPEC_BASE}/{row.location_id}/{row.file}")
    return urls


def stage_spectra(chunk_size=512, workers=16):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    if os.path.exists(SPEC_PARQUET):
        log(f"{SPEC_PARQUET} already exists -> skipping stage")
        return
    _ensure_small_inputs()
    meta = pd.read_parquet(META_PARQUET)
    os.makedirs(CHUNK_DIR, exist_ok=True)
    norm = [None]                               # built from the first spectrum's grid

    def process(row):
        for url in _spectrum_urls(row):
            buf = _fetch(url)
            if buf:
                break
        if not buf:
            return None
        flux, sig, wl = _read_spectrum(buf)
        if len(flux) != NPIX:
            return None
        if norm[0] is None:
            norm[0] = Normalizer(wl)
        return norm[0](flux, sig)

    n_chunks = (len(meta) + chunk_size - 1) // chunk_size
    t0, n_missing = time.time(), 0
    for ci in range(n_chunks):
        out = os.path.join(CHUNK_DIR, f"chunk_{ci:04d}.parquet")
        if os.path.exists(out):
            continue
        rows = meta.iloc[ci * chunk_size:(ci + 1) * chunk_size]
        with ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(process, rows.itertuples()))
        flux = np.ones((len(rows), NPIX), np.float32)
        cont = np.zeros((len(rows), NPIX), np.float32)
        ivar = np.zeros((len(rows), NPIX), np.float32)
        ok = np.zeros(len(rows), bool)
        for i, r in enumerate(results):
            if r is not None:
                flux[i], cont[i], ivar[i] = r
                ok[i] = True
        n_missing += int((~ok).sum())
        tab = pa.table({
            "sdss_id": rows["sdss_id"].values,
            "snr": rows["snr"].values.astype(np.float32),
            "spectrum_flags": np.zeros(len(rows), np.int32),
            "zeropoint": np.full(len(rows), ZEROPOINT, np.float32),
            "telescope": rows["telescope"].fillna("apo25m").values,
            "spectrum_ok": ok,
            "flux": pa.FixedSizeListArray.from_arrays(flux.ravel(), NPIX),
            "continuum": pa.FixedSizeListArray.from_arrays(cont.ravel(), NPIX),
            "ivar": pa.FixedSizeListArray.from_arrays(ivar.ravel(), NPIX),
        })
        pq.write_table(tab, out, compression="zstd",
                       use_byte_stream_split=["flux", "continuum", "ivar"])
        done = min((ci + 1) * chunk_size, len(meta))
        rate = done / max(time.time() - t0, 1)
        log(f"chunk {ci + 1}/{n_chunks} done ({done}/{len(meta)} stars, "
            f"{rate:.0f}/s, {n_missing} missing so far)")
    log(f"spectra stage complete; {n_missing} spectra missing in this run")


def stage_concat(chunk_size=512):
    """Merge chunks into one parquet whose row groups are exactly 20000 rows —
    build_lnflux_streaming's default batch_rows — so its per-batch `.values`
    flatten (which assumes batches never slice into a row group) stays aligned."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if os.path.exists(SPEC_PARQUET):
        log(f"{SPEC_PARQUET} already exists -> skipping stage")
        return
    files = sorted(os.path.join(CHUNK_DIR, f) for f in os.listdir(CHUNK_DIR)
                   if f.endswith(".parquet"))
    n_expect = (44784 + chunk_size - 1) // chunk_size
    if len(files) != n_expect:
        sys.exit(f"only {len(files)}/{n_expect} chunks present — rerun --stage spectra")
    ROWG = 20000
    writer = None
    buf, total = [], 0

    def flush(tables, nrows):
        nonlocal writer
        t = pa.concat_tables(tables).combine_chunks()
        head, tail = t.slice(0, nrows), t.slice(nrows)
        if writer is None:
            writer = pq.ParquetWriter(SPEC_PARQUET, t.schema, compression="zstd",
                                      use_byte_stream_split=["flux", "continuum", "ivar"])
        writer.write_table(head, row_group_size=ROWG)
        return [tail] if len(tail) else []

    for f in files:
        buf.append(pq.read_table(f))
        total += len(buf[-1])
        while sum(len(t) for t in buf) >= ROWG:
            buf = flush(buf, ROWG)
    if buf and sum(len(t) for t in buf):
        buf = flush(buf, sum(len(t) for t in buf))
    writer.close()
    log(f"wrote {SPEC_PARQUET} ({total} rows, "
        f"{os.path.getsize(SPEC_PARQUET)/1e9:.2f} GB)")
    for f in files:
        os.remove(f)
    os.rmdir(CHUNK_DIR)
    log("chunk cache removed")


# ----------------------------------------------------------------------
# stage: verify
# ----------------------------------------------------------------------
def stage_verify():
    import pandas as pd
    meta = pd.read_parquet(META_PARQUET)
    print(f"\nParent Sample rows:            {len(meta)}   (paper: 44784)")
    print(f"training_set == 1:             {int(meta.hogg_training_set.sum())}   (paper: 28226)")
    print(f"sample A / B:                  {(meta.hogg_sample=='A').sum()} / "
          f"{(meta.hogg_sample=='B').sum()}")
    comp = meta[["g_mag", "bp_mag", "rp_mag", "j_mag", "h_mag", "k_mag",
                 "w1_mag", "w2_mag", "plx", "e_plx"]].notna().all(axis=1)
    print(f"complete photometry+astrom:    {int(comp.sum())}")
    jk = meta.j_mag - meta.k_mag
    bprp = meta.bp_mag - meta.rp_mag
    hw2 = meta.h_mag - meta.w2_mag
    print(f"color cut eq.6 (J-K):          {int((jk < 0.4 + 0.45*bprp).sum())} pass")
    print(f"color cut eq.7 (H-W2):         {int((hw2 > -0.05).sum())} pass")
    print(f"logg in (0, 2.2]:              {int(((meta.logg > 0) & (meta.logg <= 2.2)).sum())}")
    tr = (meta.e_plx < 0.1) & (meta.visibility_periods_used >= 8) & \
         (meta.astrometric_chi2_al / np.sqrt(meta.astrometric_n_good_obs_al - 5) <= 35)
    print(f"paper eq.8-10 re-derived:      {int(tr.sum())}   "
          f"(agreement with flag: {int((tr == (meta.hogg_training_set == 1)).sum())}/{len(meta)})")
    if os.path.exists(SPEC_PARQUET):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(SPEC_PARQUET)
        print(f"spectra parquet rows:          {pf.metadata.num_rows}  "
              f"(row groups: {pf.metadata.num_row_groups})")
        ok = np.asarray(pq.read_table(SPEC_PARQUET, columns=["spectrum_ok"])
                        ["spectrum_ok"])
        b = next(pf.iter_batches(batch_size=8,
                                 columns=["flux", "continuum", "ivar"]))
        f0 = np.asarray(b["flux"][0].values, float)
        c0 = np.asarray(b["continuum"][0].values, float)
        iv = np.asarray(b["ivar"][0].values, float)
        g = iv > 0
        print(f"spectra ok: {ok.sum()}/{len(ok)}; star 0: "
              f"{g.sum()} good pixels, median norm flux "
              f"{np.median(f0[g]/c0[g]):.4f} (expect ~1)")
    else:
        print("spectra parquet not built yet")
    print("\nnext: add nothing to REGISTRY — band columns match 'dr17'.")
    print("run:  python run_full_gadi.py --dataset dr17 \\")
    print(f"        --parquet {os.path.relpath(SPEC_PARQUET, HERE)} \\")
    print(f"        --allstar {os.path.relpath(ALLSTAR_OUT, HERE)}")
    print("for the exact paper split, use hogg_training_set / hogg_sample from the allstar table")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["meta", "spectra", "concat", "verify", "all"])
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "hogg18"),
                    help="output dir (on Gadi use /scratch, not $HOME)")
    args = ap.parse_args()
    set_out(args.out)
    if args.stage in ("meta", "all"):
        stage_meta()
    if args.stage in ("spectra", "all"):
        stage_spectra(args.chunk_size, args.workers)
    if args.stage in ("concat", "all"):
        stage_concat(args.chunk_size)
    if args.stage in ("verify", "all"):
        stage_verify()
